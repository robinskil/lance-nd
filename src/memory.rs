//! The memory column: the bytes in memory of each row.
//!
//! A writer and a compaction size fragments by memory. Lance stores only
//! the size of a fragment on disk, so each row stores its own size in
//! [`MEMORY_COLUMN`]. A compaction then sums one small column to plan.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, UInt64Type};

use crate::error::Result;

/// The column with the bytes in memory of each row. Its name starts with
/// [`crate::STATS_PREFIX`], so readers leave it out of the nd record
/// batches. A row without a value, as an older writer leaves it, has no
/// known size.
pub const MEMORY_COLUMN: &str = "__stats__memory";

/// The bytes in memory of each row of `batch`, in all its columns: the
/// values, and the offsets and validity bits of each list and struct. A
/// null variable has no values, but its offsets and bits still count.
pub(crate) fn row_memory(batch: &RecordBatch) -> UInt64Array {
    let mut bits = vec![0u64; batch.num_rows()];
    for column in batch.columns() {
        add_bits(column.as_ref(), &mut bits);
    }
    UInt64Array::from_iter_values(bits.into_iter().map(|bits| bits.div_ceil(8)))
}

/// The bits in memory of one row of each field of `schema` when the row
/// lacks the field, as [`crate::schema::absent`] fills it.
pub(crate) fn absent_bits(schema: &Schema) -> Result<Vec<u64>> {
    schema
        .fields()
        .iter()
        .map(|field| {
            let mut bits = [0u64];
            add_bits(crate::schema::absent(field, 1)?.as_ref(), &mut bits);
            Ok(bits[0])
        })
        .collect()
}

/// Adds the bits in memory of each row of `array` to `rows`.
fn add_bits(array: &dyn Array, rows: &mut [u64]) {
    if array.nulls().is_some() {
        rows.iter_mut().for_each(|bits| *bits += 1);
    }
    let fixed = |rows: &mut [u64], width: u64| rows.iter_mut().for_each(|bits| *bits += width);
    match array.data_type() {
        DataType::Null => {}
        DataType::Boolean => fixed(rows, 1),
        DataType::Utf8 | DataType::Binary => {
            let offsets = match array.as_string_opt::<i32>() {
                Some(strings) => strings.value_offsets(),
                None => array.as_binary::<i32>().value_offsets(),
            };
            add_spans(rows, offsets, 32, |start, end| 8 * (end - start) as u64);
        }
        DataType::LargeUtf8 | DataType::LargeBinary => {
            let offsets = match array.as_string_opt::<i64>() {
                Some(strings) => strings.value_offsets(),
                None => array.as_binary::<i64>().value_offsets(),
            };
            add_spans(rows, offsets, 64, |start, end| 8 * (end - start) as u64);
        }
        DataType::List(_) => {
            let list = array.as_list::<i32>();
            let items = item_sums(list.values().as_ref());
            add_spans(rows, list.value_offsets(), 32, |start, end| {
                items[end as usize] - items[start as usize]
            });
        }
        DataType::LargeList(_) => {
            let list = array.as_list::<i64>();
            let items = item_sums(list.values().as_ref());
            add_spans(rows, list.value_offsets(), 64, |start, end| {
                items[end as usize] - items[start as usize]
            });
        }
        DataType::FixedSizeList(_, size) => {
            let list = array.as_fixed_size_list();
            let items = item_sums(list.values().as_ref());
            let size = *size as usize;
            for (row, bits) in rows.iter_mut().enumerate() {
                let start = list.value_offset(row) as usize;
                *bits += items[start + size] - items[start];
            }
        }
        DataType::Struct(_) => {
            for child in array.as_struct().columns() {
                add_bits(child.as_ref(), rows);
            }
        }
        data_type => match data_type.primitive_width() {
            Some(width) => fixed(rows, 8 * width as u64),
            // Other types share their buffers evenly over the rows.
            None => {
                let share = 8 * array.get_array_memory_size() as u64 / rows.len().max(1) as u64;
                fixed(rows, share);
            }
        },
    }
}

/// The running sums of the bits of the items of `items`, from 0.
fn item_sums(items: &dyn Array) -> Vec<u64> {
    let mut bits = vec![0u64; items.len()];
    add_bits(items, &mut bits);
    let mut sums = Vec::with_capacity(bits.len() + 1);
    let mut sum = 0;
    sums.push(0);
    for bits in bits {
        sum += bits;
        sums.push(sum);
    }
    sums
}

/// Adds to each row `width` bits for its offset and `span` of its range.
fn add_spans<O: Copy>(rows: &mut [u64], offsets: &[O], width: u64, span: impl Fn(O, O) -> u64) {
    for (row, bits) in rows.iter_mut().enumerate() {
        *bits += width + span(offsets[row], offsets[row + 1]);
    }
}

/// `batch` with [`MEMORY_COLUMN`] set to [`row_memory`], in place of an
/// existing column of that name.
pub(crate) fn add_memory(batch: &RecordBatch) -> Result<RecordBatch> {
    let schema = batch.schema();
    let mut fields: Vec<Arc<Field>> = schema.fields().iter().cloned().collect();
    let mut columns = batch.columns().to_vec();
    let index = match schema.index_of(MEMORY_COLUMN) {
        Ok(index) => index,
        Err(_) => {
            fields.push(Arc::new(Field::new(MEMORY_COLUMN, DataType::UInt64, true)));
            columns.push(Arc::new(UInt64Array::from(vec![0; batch.num_rows()])));
            columns.len() - 1
        }
    };
    // A column of 8 byte values without nulls, as it is written.
    columns[index] = Arc::new(UInt64Array::from(vec![0; batch.num_rows()]));
    let schema = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
    let sized = RecordBatch::try_new(schema.clone(), columns.clone())?;
    columns[index] = Arc::new(row_memory(&sized)) as ArrayRef;
    Ok(RecordBatch::try_new(schema, columns)?)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_lists_count_their_offsets() {
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let field = Field::new("v", DataType::List(item), true);
        let schema = Arc::new(Schema::new(vec![field.clone()]));
        let array = crate::schema::absent(&field, 3).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
        // A 4 byte offset and a validity bit per row, without values.
        assert_eq!(row_memory(&batch).values().to_vec(), [5, 5, 5]);
        assert_eq!(absent_bits(&schema).unwrap(), [33]);
    }
}
