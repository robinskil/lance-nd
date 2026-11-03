//! Per-row min and max values of each variable, for pruning.
//!
//! A table with stats has the schema metadata key [`STATS_METADATA_KEY`] and
//! a `__stats__` column. The column has one child per variable with a stats
//! type. Each child is a struct of `min` and `max`, typed by the value type.
//! A null child means that the row does not have the variable. Null and NaN
//! values do not count. A variable without counted values has null `min`
//! and `max`.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, PrimitiveArray, RecordBatch, StringArray, StructArray, UInt64Array,
    downcast_primitive_array, make_comparator, new_empty_array, new_null_array,
};
use arrow::compute::kernels::aggregate::{
    max, max_string, max_string_view, min, min_string, min_string_view,
};
use arrow::compute::{SortOptions, cast, concat, take};
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, Field, Fields, Float16Type, Float32Type, Float64Type, Schema,
};
use nd_arrow_array::extension::{VALUES_FIELD, is_nd_field};

use crate::error::Result;

/// The name of the stats column. It is a reserved variable name.
pub const STATS_FIELD: &str = "__stats__";

/// The table schema metadata key that marks a table with stats.
pub const STATS_METADATA_KEY: &str = "lance_nd:stats";

/// `true` when the schema marks a table with stats.
pub fn has_stats(schema: &Schema) -> bool {
    schema
        .metadata()
        .get(STATS_METADATA_KEY)
        .is_some_and(|value| value == "true")
}

/// `true` when values of `data_type` get a min and a max.
pub fn is_stats_type(data_type: &DataType) -> bool {
    data_type.is_integer()
        || data_type.is_floating()
        || matches!(
            data_type,
            DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Utf8View
                | DataType::Timestamp(_, _)
                | DataType::Date32
                | DataType::Date64
                | DataType::Time32(_)
                | DataType::Time64(_)
                | DataType::Duration(_)
        )
}

/// `batch` with a `__stats__` column and the stats marker in its metadata.
///
/// `batch` is an encoded batch of nd columns. Without a variable of a stats
/// type, only the marker is added.
pub fn add_stats(batch: &RecordBatch) -> Result<RecordBatch> {
    let mut children: Vec<Field> = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
        if !is_nd_field(field) {
            continue;
        }
        let nd = column.as_struct();
        let values = nd
            .column_by_name(VALUES_FIELD)
            .expect("an nd column has values")
            .as_list::<i32>();
        let value_type = values.value_type();
        if !is_stats_type(&value_type) {
            continue;
        }
        let (min, max) = min_max(nd, values.offsets(), values.values().as_ref())?;
        let bounds = Fields::from(vec![
            Field::new("min", value_type.clone(), true),
            Field::new("max", value_type, true),
        ]);
        let child = StructArray::try_new(bounds.clone(), vec![min, max], nd.nulls().cloned())?;
        children.push(Field::new(field.name(), DataType::Struct(bounds), true));
        arrays.push(Arc::new(child));
    }

    let mut metadata = batch.schema().metadata().clone();
    metadata.insert(STATS_METADATA_KEY.to_string(), "true".to_string());
    let mut fields: Vec<Arc<Field>> = batch.schema().fields().iter().cloned().collect();
    let mut columns = batch.columns().to_vec();
    if !children.is_empty() {
        let children = Fields::from(children);
        fields.push(Arc::new(Field::new(
            STATS_FIELD,
            DataType::Struct(children.clone()),
            true,
        )));
        columns.push(Arc::new(StructArray::try_new(children, arrays, None)?));
    }
    let schema = Schema::new_with_metadata(fields, metadata);
    Ok(RecordBatch::try_new(Arc::new(schema), columns)?)
}

/// The min and the max of each row of an nd column. An absent row and a row
/// without counted values give nulls.
fn min_max(nd: &StructArray, offsets: &[i32], values: &dyn Array) -> Result<(ArrayRef, ArrayRef)> {
    let mut mins: Vec<ArrayRef> = Vec::with_capacity(nd.len());
    let mut maxs: Vec<ArrayRef> = Vec::with_capacity(nd.len());
    for row in 0..nd.len() {
        let (min, max) = if nd.is_valid(row) {
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            row_bounds(values.slice(start, end - start).as_ref())?
        } else {
            let null = new_null_array(values.data_type(), 1);
            (null.clone(), null)
        };
        mins.push(min);
        maxs.push(max);
    }
    let min = join(&mins, values.data_type())?;
    let max = join(&maxs, values.data_type())?;
    Ok((widen_zero(&min, -0.0), widen_zero(&max, 0.0)))
}

fn join(arrays: &[ArrayRef], data_type: &DataType) -> Result<ArrayRef> {
    if arrays.is_empty() {
        return Ok(new_empty_array(data_type));
    }
    let arrays: Vec<&dyn Array> = arrays.iter().map(|a| a.as_ref()).collect();
    Ok(concat(&arrays)?)
}

/// The min and the max of the values of one row, as arrays of one value.
fn row_bounds(values: &dyn Array) -> Result<(ArrayRef, ArrayRef)> {
    let data_type = values.data_type().clone();
    // The kernels skip nulls but count NaN, so a NaN result takes the slow path.
    let fast: Option<(ArrayRef, ArrayRef)> = downcast_primitive_array!(
        values => {
            let (low, high) = (min(values), max(values));
            (!is_nan(low) && !is_nan(high))
                .then(|| (one(values, low, &data_type), one(values, high, &data_type)))
        }
        DataType::Utf8 => {
            let values = values.as_string::<i32>();
            Some(strings(min_string(values), max_string(values)))
        }
        DataType::LargeUtf8 => {
            let values = values.as_string::<i64>();
            Some(strings(min_string(values), max_string(values)))
        }
        DataType::Utf8View => {
            let values = values.as_string_view();
            Some(strings(min_string_view(values), max_string_view(values)))
        }
        _ => None
    );
    match fast {
        Some(bounds) => {
            let (low, high) = bounds;
            Ok((cast(&low, &data_type)?, cast(&high, &data_type)?))
        }
        None => compared_bounds(values),
    }
}

/// `true` for a NaN. Only a NaN is not equal to itself.
#[allow(clippy::eq_op)]
fn is_nan<T: PartialEq>(value: Option<T>) -> bool {
    value.is_some_and(|v| v != v)
}

/// An array of `value`, of the type of `like`.
fn one<T: ArrowPrimitiveType>(
    _like: &PrimitiveArray<T>,
    value: Option<T::Native>,
    data_type: &DataType,
) -> ArrayRef {
    let array: PrimitiveArray<T> = [value].into_iter().collect();
    Arc::new(array.with_data_type(data_type.clone()))
}

fn strings(low: Option<&str>, high: Option<&str>) -> (ArrayRef, ArrayRef) {
    (
        Arc::new(StringArray::from(vec![low])),
        Arc::new(StringArray::from(vec![high])),
    )
}

/// The bounds of `values` by comparison, without null and NaN values.
fn compared_bounds(values: &dyn Array) -> Result<(ArrayRef, ArrayRef)> {
    let compare = make_comparator(values, values, SortOptions::default())?;
    let mut low: Option<usize> = None;
    let mut high: Option<usize> = None;
    for i in (0..values.len()).filter(|&i| values.is_valid(i) && !is_nan_at(values, i)) {
        if low.is_none_or(|l| compare(i, l) == Ordering::Less) {
            low = Some(i);
        }
        if high.is_none_or(|h| compare(i, h) == Ordering::Greater) {
            high = Some(i);
        }
    }
    let low = take(
        values,
        &UInt64Array::from(vec![low.map(|i| i as u64)]),
        None,
    )?;
    let high = take(
        values,
        &UInt64Array::from(vec![high.map(|i| i as u64)]),
        None,
    )?;
    Ok((low, high))
}

/// `bounds` with each zero as `zero`. Total order puts -0.0 below 0.0, but a
/// filter treats them as equal, so a min gets -0.0 and a max gets 0.0.
fn widen_zero(bounds: &ArrayRef, zero: f64) -> ArrayRef {
    match bounds.data_type() {
        DataType::Float16 => {
            let zero = <Float16Type as ArrowPrimitiveType>::Native::from_f64(zero);
            Arc::new(
                bounds
                    .as_primitive::<Float16Type>()
                    .unary::<_, Float16Type>(|v| if v.to_f64() == 0.0 { zero } else { v }),
            )
        }
        DataType::Float32 => Arc::new(
            bounds
                .as_primitive::<Float32Type>()
                .unary::<_, Float32Type>(|v| if v == 0.0 { zero as f32 } else { v }),
        ),
        DataType::Float64 => Arc::new(
            bounds
                .as_primitive::<Float64Type>()
                .unary::<_, Float64Type>(|v| if v == 0.0 { zero } else { v }),
        ),
        _ => bounds.clone(),
    }
}

fn is_nan_at(values: &dyn Array, i: usize) -> bool {
    match values.data_type() {
        DataType::Float16 => values.as_primitive::<Float16Type>().value(i).is_nan(),
        DataType::Float32 => values.as_primitive::<Float32Type>().value(i).is_nan(),
        DataType::Float64 => values.as_primitive::<Float64Type>().value(i).is_nan(),
        _ => false,
    }
}
