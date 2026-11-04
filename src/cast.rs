//! Cast new rows to the types of the table, see [`Casting`].

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, ListArray, RecordBatch, RecordBatchOptions, StructArray,
};
use arrow::compute::{CastOptions, cast_with_options};
use arrow::datatypes::{DataType, Field, FieldRef, Fields, Schema};

use crate::error::Result;

/// How a write treats a value or attribute type that differs from the table.
///
/// The rules follow the `casting` argument of numpy. The table type wins:
/// a cast changes the new rows, never the table.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Casting {
    /// Reject the write.
    #[default]
    No,
    /// Cast when no value can change, such as `float32` to `float64` or
    /// `uint16` to `int32`. Otherwise reject the write.
    Safe,
    /// Also cast within a kind or to a later kind in the order bool,
    /// unsigned, signed, float, such as `float64` to `float32`. An integer
    /// that does not fit fails the write.
    SameKind,
}

/// `batch` with each field cast to the type of the same field in `target`.
/// A field that the rule does not allow keeps its type, so the schema check
/// rejects it later.
pub(crate) fn cast_to(
    batch: RecordBatch,
    target: &Schema,
    casting: Casting,
) -> Result<RecordBatch> {
    if casting == Casting::No {
        return Ok(batch);
    }
    let mut fields = Vec::with_capacity(batch.num_columns());
    let mut columns = Vec::with_capacity(batch.num_columns());
    for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
        let (field, column) = match target.field_with_name(field.name()) {
            Ok(target) => cast_field(field, column, target, casting)?,
            Err(_) => (field.clone(), column.clone()),
        };
        fields.push(field);
        columns.push(column);
    }
    let schema = Schema::new_with_metadata(fields, batch.schema().metadata().clone());
    let options = RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
    Ok(RecordBatch::try_new_with_options(
        Arc::new(schema),
        columns,
        &options,
    )?)
}

fn cast_field(
    field: &FieldRef,
    array: &ArrayRef,
    target: &Field,
    casting: Casting,
) -> Result<(FieldRef, ArrayRef)> {
    let typed = |data_type: DataType| Arc::new(field.as_ref().clone().with_data_type(data_type));
    match (field.data_type(), target.data_type()) {
        (from, to) if from == to => Ok((field.clone(), array.clone())),
        (DataType::Struct(children), DataType::Struct(targets)) => {
            let source = array.as_struct();
            let mut fields = Vec::with_capacity(children.len());
            let mut columns = Vec::with_capacity(children.len());
            for (child, column) in children.iter().zip(source.columns()) {
                let (child, column) = match targets.find(child.name()) {
                    Some((_, target)) => cast_field(child, column, target, casting)?,
                    None => (child.clone(), column.clone()),
                };
                fields.push(child);
                columns.push(column);
            }
            let fields = Fields::from(fields);
            let array = StructArray::try_new(fields.clone(), columns, source.nulls().cloned())?;
            Ok((typed(DataType::Struct(fields)), Arc::new(array)))
        }
        (DataType::List(item), DataType::List(target)) => {
            let source = array.as_list::<i32>();
            let (item, values) = cast_field(item, source.values(), target, casting)?;
            let array = ListArray::try_new(
                item.clone(),
                source.offsets().clone(),
                values,
                source.nulls().cloned(),
            )?;
            Ok((typed(DataType::List(item)), Arc::new(array)))
        }
        (from, to) if allowed(from, to, casting) => {
            // Not `safe`: an integer that does not fit fails instead of becoming null.
            let options = CastOptions {
                safe: false,
                ..Default::default()
            };
            let array = cast_with_options(array, to, &options)?;
            Ok((typed(to.clone()), array))
        }
        _ => Ok((field.clone(), array.clone())),
    }
}

/// `true` when `casting` allows a cast of values from `from` to `to`.
fn allowed(from: &DataType, to: &DataType, casting: Casting) -> bool {
    let text =
        |t: &DataType| matches!(t, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View);
    match casting {
        Casting::No => false,
        _ if text(from) && text(to) => true,
        Casting::Safe => is_safe(from, to),
        Casting::SameKind => {
            is_safe(from, to)
                || kind(from)
                    .zip(kind(to))
                    .is_some_and(|(from, to)| from <= to)
        }
    }
}

/// The numpy kind order: bool, unsigned, signed, float.
fn kind(data_type: &DataType) -> Option<u8> {
    match data_type {
        DataType::Boolean => Some(0),
        t if t.is_unsigned_integer() => Some(1),
        t if t.is_signed_integer() => Some(2),
        t if t.is_floating() => Some(3),
        _ => None,
    }
}

/// `true` when each value of `from` has the same value in `to`, as numpy
/// `can_cast(from, to, "safe")`.
fn is_safe(from: &DataType, to: &DataType) -> bool {
    let bits = |t: &DataType| t.primitive_width().map_or(0, |bytes| bytes * 8);
    match (kind(from), kind(to)) {
        (Some(0), Some(_)) => true,
        (Some(f), Some(t)) if f == t => bits(to) > bits(from),
        // An unsigned integer fits a signed integer with more bits.
        (Some(1), Some(2)) => bits(to) > bits(from),
        // As numpy: up to 8 bits fit float16, 16 bits float32, all float64.
        (Some(1 | 2), Some(3)) => match bits(to) {
            16 => bits(from) <= 8,
            32 => bits(from) <= 16,
            _ => true,
        },
        _ => false,
    }
}
