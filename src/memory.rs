//! The memory column: the bytes in memory of the nd values of each row.
//!
//! A writer and a compaction size fragments by memory. Lance stores only
//! the size of a fragment on disk, so each row stores its own size in
//! [`MEMORY_COLUMN`]. A compaction then sums one small column to plan.

use std::ops::Range;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, UInt64Type};
use nd_arrow_array::extension::{VALUES_FIELD, is_nd_field};

use crate::error::Result;

/// The column with the bytes in memory of the nd values of each row. Its
/// name starts with [`crate::STATS_PREFIX`], so readers leave it out of the
/// nd record batches. A row without a value, as an older writer leaves it,
/// has no known size.
pub const MEMORY_COLUMN: &str = "__stats__memory";

/// The bytes in memory of the nd values of each row of `batch`: the values
/// times their width, and the bytes and offsets of strings and binary
/// values. Other columns do not count.
pub(crate) fn row_memory(batch: &RecordBatch) -> UInt64Array {
    let mut bytes = vec![0u64; batch.num_rows()];
    for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
        if !is_nd_field(field) {
            continue;
        }
        let nd = column.as_struct();
        let Some(values) = nd.column_by_name(VALUES_FIELD) else {
            continue;
        };
        let values = values.as_list::<i32>();
        let items = values.values();
        for (row, size) in bytes.iter_mut().enumerate() {
            if nd.is_null(row) {
                continue;
            }
            let start = values.value_offsets()[row] as usize;
            let end = values.value_offsets()[row + 1] as usize;
            *size += value_bytes(items, start..end);
        }
    }
    UInt64Array::from(bytes)
}

/// The bytes in memory of `items` at `range`.
fn value_bytes(items: &ArrayRef, range: Range<usize>) -> u64 {
    let count = range.len() as u64;
    let span = |start: i64, end: i64| (end - start) as u64;
    if let Some(strings) = items.as_string_opt::<i32>() {
        let offsets = strings.value_offsets();
        return span(offsets[range.start].into(), offsets[range.end].into()) + 4 * count;
    }
    if let Some(strings) = items.as_string_opt::<i64>() {
        let offsets = strings.value_offsets();
        return span(offsets[range.start], offsets[range.end]) + 8 * count;
    }
    if let Some(binary) = items.as_binary_opt::<i32>() {
        let offsets = binary.value_offsets();
        return span(offsets[range.start].into(), offsets[range.end].into()) + 4 * count;
    }
    if let Some(binary) = items.as_binary_opt::<i64>() {
        let offsets = binary.value_offsets();
        return span(offsets[range.start], offsets[range.end]) + 8 * count;
    }
    match items.data_type() {
        DataType::Boolean => count.div_ceil(8),
        data_type => data_type
            .primitive_width()
            .map_or(16 * count, |width| width as u64 * count),
    }
}

/// `batch` with [`MEMORY_COLUMN`] set to [`row_memory`], in place of an
/// existing column of that name.
pub(crate) fn add_memory(batch: &RecordBatch) -> Result<RecordBatch> {
    let memory: ArrayRef = Arc::new(row_memory(batch));
    let schema = batch.schema();
    let mut fields: Vec<Arc<Field>> = schema.fields().iter().cloned().collect();
    let mut columns = batch.columns().to_vec();
    match schema.index_of(MEMORY_COLUMN) {
        Ok(index) => columns[index] = memory,
        Err(_) => {
            fields.push(Arc::new(Field::new(MEMORY_COLUMN, DataType::UInt64, true)));
            columns.push(memory);
        }
    }
    let schema = Schema::new_with_metadata(fields, schema.metadata().clone());
    Ok(RecordBatch::try_new(Arc::new(schema), columns)?)
}

/// The bytes in memory of the rows of `batch`, from [`MEMORY_COLUMN`] when
/// it has a value in each row, else from [`row_memory`].
pub(crate) fn memory_sum(batch: &RecordBatch) -> u64 {
    if let Some(column) = batch.column_by_name(MEMORY_COLUMN)
        && column.null_count() == 0
        && let Some(memory) = column.as_primitive_opt::<UInt64Type>()
    {
        return memory.values().iter().sum();
    }
    row_memory(batch).values().iter().sum()
}
