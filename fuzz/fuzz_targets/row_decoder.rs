//! `RowDecoder` must return an error for a bad batch, never panic.
//!
//! The input becomes a record batch that Arrow accepts, with nd columns of
//! any content: value counts that do not match the dimension sizes, null
//! rows, columns without the nd extension and repeated names.

#![no_main]

use std::sync::Arc;

use arbitrary::Arbitrary;
use arrow::array::{
    ArrayRef, Float64Array, ListArray, RecordBatch, StringArray, StructArray, UInt32Array,
};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::datatypes::{DataType, Field, Fields, Schema};
use lance_nd::encode::RowDecoder;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    rows: u8,
    columns: Vec<Column>,
}

#[derive(Arbitrary, Debug)]
struct Column {
    name: u8,
    extension: bool,
    cells: Vec<Cell>,
}

#[derive(Arbitrary, Debug)]
struct Cell {
    valid: bool,
    values: Vec<Option<f64>>,
    dims: Vec<(u8, u32)>,
}

/// A list column with one list of `items` per length in `lengths`.
fn list(item: Field, items: ArrayRef, lengths: &[usize]) -> ArrayRef {
    let offsets = OffsetBuffer::from_lengths(lengths.iter().copied());
    Arc::new(ListArray::new(Arc::new(item), offsets, items, None))
}

fn nd_column(column: &Column, rows: usize) -> (Field, ArrayRef) {
    let empty = Cell {
        valid: false,
        values: Vec::new(),
        dims: Vec::new(),
    };
    let cells: Vec<&Cell> = (0..rows)
        .map(|row| column.cells.get(row).unwrap_or(&empty))
        .collect();
    let values: Vec<Option<f64>> = cells
        .iter()
        .flat_map(|c| c.values.iter().copied())
        .collect();
    let names: Vec<String> = cells
        .iter()
        .flat_map(|c| c.dims.iter().map(|(name, _)| format!("d{}", name % 4)))
        .collect();
    let sizes: Vec<u32> = cells
        .iter()
        .flat_map(|c| c.dims.iter().map(|(_, size)| size % 8))
        .collect();
    let value_lengths: Vec<usize> = cells.iter().map(|c| c.values.len()).collect();
    let dim_lengths: Vec<usize> = cells.iter().map(|c| c.dims.len()).collect();

    let item = Field::new("item", DataType::Float64, true);
    let name_item = Field::new("item", DataType::Utf8, false);
    let size_item = Field::new("item", DataType::UInt32, false);
    let children = vec![
        list(
            item.clone(),
            Arc::new(Float64Array::from(values)),
            &value_lengths,
        ),
        list(
            name_item.clone(),
            Arc::new(StringArray::from(names)),
            &dim_lengths,
        ),
        list(
            size_item.clone(),
            Arc::new(UInt32Array::from(sizes)),
            &dim_lengths,
        ),
    ];
    let fields = Fields::from(vec![
        Field::new("values", DataType::new_list(DataType::Float64, true), false)
            .with_data_type(DataType::List(Arc::new(item))),
        Field::new("dim_names", DataType::List(Arc::new(name_item)), false),
        Field::new("dim_sizes", DataType::List(Arc::new(size_item)), false),
    ]);
    let nulls = NullBuffer::from(cells.iter().map(|c| c.valid).collect::<Vec<_>>());
    let array = StructArray::new(fields.clone(), children, Some(nulls));
    let mut field = Field::new(
        format!("v{}", column.name % 4),
        DataType::Struct(fields),
        true,
    );
    if column.extension {
        field = field.with_metadata(
            [
                (
                    "ARROW:extension:name".to_string(),
                    "nd_arrow.array".to_string(),
                ),
                (
                    "ARROW:extension:metadata".to_string(),
                    r#"{"version":1}"#.to_string(),
                ),
            ]
            .into(),
        );
    }
    (field, Arc::new(array))
}

fuzz_target!(|input: Input| {
    let rows = usize::from(input.rows % 16);
    let (fields, columns): (Vec<Field>, Vec<ArrayRef>) = input
        .columns
        .iter()
        .take(6)
        .map(|column| nd_column(column, rows))
        .unzip();
    let options = arrow::array::RecordBatchOptions::new().with_row_count(Some(rows));
    let Ok(batch) =
        RecordBatch::try_new_with_options(Arc::new(Schema::new(fields)), columns, &options)
    else {
        return;
    };
    if let Ok(decoder) = RowDecoder::try_new(&batch) {
        for row in decoder.into_rows() {
            let _ = row;
        }
    }
});
