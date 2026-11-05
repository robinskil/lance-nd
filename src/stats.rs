//! Per-row min and max values of each variable, for pruning.
//!
//! A table with stats has the schema metadata key [`STATS_METADATA_KEY`]. Each
//! variable with a stats type has a stats column `__stats__:<variable>`, a
//! struct of `min` and `max` typed by the value type. A writer adds a variable
//! and its stats column together, so an older fragment lacks whole columns
//! only, which Lance reads as null. A null row means that the row does not
//! have the variable. Null and NaN values do not count. A variable without
//! counted values has null `min` and `max`.

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

/// The name prefix of the stats columns. Each variable name that starts with
/// it is reserved.
pub const STATS_PREFIX: &str = "__stats__";

/// The table schema metadata key that marks a table with stats.
pub const STATS_METADATA_KEY: &str = "lance_nd:stats";

/// The value of [`STATS_METADATA_KEY`] in a table with stats.
const STATS_ON: &str = "true";

/// `true` when the schema marks a table with stats.
pub fn has_stats(schema: &Schema) -> bool {
    schema
        .metadata()
        .get(STATS_METADATA_KEY)
        .is_some_and(|value| value == STATS_ON)
}

/// The name of the stats column of `variable`.
pub fn stats_column(variable: &str) -> String {
    // Lance 11 does not allow `.` in a top-level name.
    format!("{STATS_PREFIX}:{variable}")
}

/// The variable of the stats column `column`, or `None` for another name.
pub fn stats_variable(column: &str) -> Option<&str> {
    column.strip_prefix(STATS_PREFIX)?.strip_prefix(':')
}

/// `true` for a reserved name: a stats column or another name that starts
/// with [`STATS_PREFIX`].
pub fn is_stats_name(name: &str) -> bool {
    name.starts_with(STATS_PREFIX)
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

/// `batch` with a stats column for each variable of a stats type, and the
/// stats marker in its metadata.
///
/// `batch` is an encoded batch of nd columns. Without a variable of a stats
/// type, only the marker is added.
pub fn add_stats(batch: &RecordBatch) -> Result<RecordBatch> {
    let mut fields: Vec<Arc<Field>> = batch.schema().fields().iter().cloned().collect();
    let mut columns = batch.columns().to_vec();
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
        let stats = StructArray::try_new(bounds.clone(), vec![min, max], nd.nulls().cloned())?;
        let name = stats_column(field.name());
        fields.push(Arc::new(Field::new(name, DataType::Struct(bounds), true)));
        columns.push(Arc::new(stats));
    }

    let mut metadata = batch.schema().metadata().clone();
    metadata.insert(STATS_METADATA_KEY.to_string(), STATS_ON.to_string());
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

#[cfg(test)]
mod tests {
    use arrow::array::{
        Array, ArrayRef, AsArray, BooleanArray, Float64Array, Int32Array, StringArray,
        TimestampNanosecondArray,
    };
    use arrow::datatypes::{Float64Type, Int32Type, TimeUnit};
    use nd_arrow_array::{Dimensions, NdArrowArray, NdRecordBatch};

    use super::*;
    use crate::encode::encode_rows;

    fn row(variables: Vec<(&str, ArrayRef)>) -> NdRecordBatch {
        let fields: Vec<Field> = variables
            .iter()
            .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
            .collect();
        let arrays = variables
            .into_iter()
            .map(|(name, array)| {
                let dims = Dimensions::try_new([(format!("{name}_dim"), array.len())]).unwrap();
                NdArrowArray::try_new(array, dims).unwrap()
            })
            .collect();
        NdRecordBatch::try_new_inferred(Arc::new(Schema::new(fields)), arrays).unwrap()
    }

    fn stats_of(rows: &[NdRecordBatch]) -> RecordBatch {
        add_stats(&encode_rows(rows).unwrap()).unwrap()
    }

    fn child<'a>(batch: &'a RecordBatch, variable: &str) -> &'a StructArray {
        batch
            .column_by_name(&stats_column(variable))
            .expect("a stats column")
            .as_struct()
    }

    /// The names of the stats columns of `schema`.
    fn stats_names(schema: &Schema) -> Vec<String> {
        schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|name| is_stats_name(name))
            .collect()
    }

    fn min_max(child: &StructArray) -> (&ArrayRef, &ArrayRef) {
        (
            child.column_by_name("min").unwrap(),
            child.column_by_name("max").unwrap(),
        )
    }

    #[test]
    fn min_max_per_row() {
        let batch = stats_of(&[
            row(vec![
                (
                    "lat",
                    Arc::new(Float64Array::from(vec![-90.0, -89.0, -88.0])),
                ),
                (
                    "sst",
                    Arc::new(Float64Array::from(vec![Some(5.0), None, Some(1.0)])),
                ),
            ]),
            row(vec![
                ("lat", Arc::new(Float64Array::from(vec![11.0, 10.0]))),
                ("sst", Arc::new(Float64Array::from(vec![None, Some(7.0)]))),
            ]),
        ]);
        let (min, max) = min_max(child(&batch, "lat"));
        assert_eq!(min.as_primitive::<Float64Type>().values(), &[-90.0, 10.0]);
        assert_eq!(max.as_primitive::<Float64Type>().values(), &[-88.0, 11.0]);
        let (min, max) = min_max(child(&batch, "sst"));
        assert_eq!(min.as_primitive::<Float64Type>().values(), &[1.0, 7.0]);
        assert_eq!(max.as_primitive::<Float64Type>().values(), &[5.0, 7.0]);
        assert!(has_stats(batch.schema().as_ref()));
    }

    #[test]
    fn nan_and_all_null() {
        let batch = stats_of(&[
            row(vec![(
                "v",
                Arc::new(Float64Array::from(vec![f64::NAN, 2.0, f64::NAN])),
            )]),
            row(vec![(
                "v",
                Arc::new(Float64Array::from(vec![Some(f64::NAN), None])),
            )]),
        ]);
        let stats = child(&batch, "v");
        assert!(stats.is_valid(0) && stats.is_valid(1));
        let (min, max) = min_max(stats);
        assert_eq!(min.as_primitive::<Float64Type>().value(0), 2.0);
        assert_eq!(max.as_primitive::<Float64Type>().value(0), 2.0);
        assert!(min.is_null(1) && max.is_null(1));
    }

    #[test]
    fn negative_nan_and_large_rows() {
        let mut big: Vec<Option<f64>> = (0..10_000).map(|i| Some(i as f64)).collect();
        big[17] = Some(-f64::NAN);
        big[18] = None;
        let batch = stats_of(&[
            row(vec![(
                "v",
                Arc::new(Float64Array::from(vec![-f64::NAN, 1.0, 2.0])),
            )]),
            row(vec![("v", Arc::new(Float64Array::from(big)))]),
        ]);
        let (min, max) = min_max(child(&batch, "v"));
        assert_eq!(min.as_primitive::<Float64Type>().values(), &[1.0, 0.0]);
        assert_eq!(max.as_primitive::<Float64Type>().values(), &[2.0, 9999.0]);
    }

    #[test]
    fn absent_variable_is_null_child() {
        let batch = stats_of(&[
            row(vec![(
                "pressure",
                Arc::new(Int32Array::from(vec![3, 1, 2])),
            )]),
            row(vec![("other", Arc::new(Int32Array::from(vec![0])))]),
        ]);
        let stats = child(&batch, "pressure");
        assert!(stats.is_valid(0) && stats.is_null(1));
        let (min, max) = min_max(stats);
        assert_eq!(min.as_primitive::<Int32Type>().value(0), 1);
        assert_eq!(max.as_primitive::<Int32Type>().value(0), 3);
    }

    #[test]
    fn strings_and_timestamps() {
        let times = TimestampNanosecondArray::from(vec![30, 10, 20]).with_timezone("UTC");
        let batch = stats_of(&[row(vec![
            ("name", Arc::new(StringArray::from(vec!["b", "a", "c"]))),
            ("time", Arc::new(times)),
        ])]);
        let (min, max) = min_max(child(&batch, "name"));
        assert_eq!(min.as_string::<i32>().value(0), "a");
        assert_eq!(max.as_string::<i32>().value(0), "c");
        let (min, max) = min_max(child(&batch, "time"));
        assert_eq!(
            min.data_type(),
            &DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into()))
        );
        assert_eq!(
            min.as_primitive::<arrow::datatypes::TimestampNanosecondType>()
                .value(0),
            10
        );
        assert_eq!(
            max.as_primitive::<arrow::datatypes::TimestampNanosecondType>()
                .value(0),
            30
        );
    }

    #[test]
    fn other_types_get_no_child() {
        let batch = stats_of(&[row(vec![
            ("flag", Arc::new(BooleanArray::from(vec![true, false]))),
            ("v", Arc::new(Int32Array::from(vec![1]))),
        ])]);
        assert_eq!(stats_names(&batch.schema()), ["__stats__:v"]);
        assert_eq!(stats_column("v"), format!("{STATS_PREFIX}:v"));

        let only_flags = stats_of(&[row(vec![(
            "flag",
            Arc::new(BooleanArray::from(vec![true])),
        )])]);
        assert!(stats_names(&only_flags.schema()).is_empty());
        assert!(has_stats(only_flags.schema().as_ref()));
    }
}
