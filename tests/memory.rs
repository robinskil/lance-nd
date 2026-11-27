//! The memory column: each row stores the bytes in memory of its values.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Float32Array, RecordBatchIterator, StringArray};
use arrow::datatypes::{Field, Schema, UInt64Type};
use futures::stream;
use lance::Dataset;
use lance::dataset::WriteParams;
use lance_file::version::LanceFileVersion;
use lance_nd::{
    CompactOptions, MEMORY_COLUMN, NdWriter, WriteOptions, compact, encode_rows, open, scan,
};
use nd_arrow_array::{Dimensions, NdArrowArray, NdRecordBatch};

fn variable(name: &str, array: ArrayRef) -> (Field, NdArrowArray) {
    let dims = Dimensions::try_new([(format!("{name}_dim"), array.len())]).unwrap();
    let field = Field::new(name, array.data_type().clone(), true);
    (field, NdArrowArray::try_new(array, dims).unwrap())
}

/// A row with `values` Float32 values in `psal`, without `psal` for 0, and
/// a `name` string of 5 bytes.
fn row(values: usize) -> NdRecordBatch {
    let mut variables = vec![variable("name", Arc::new(StringArray::from(vec!["abcde"])))];
    if values > 0 {
        let psal = Float32Array::from_iter_values((0..values).map(|i| i as f32));
        variables.push(variable("psal", Arc::new(psal)));
    }
    let (fields, arrays): (Vec<_>, Vec<_>) = variables.into_iter().unzip();
    NdRecordBatch::try_new_inferred(Arc::new(Schema::new(fields)), arrays).unwrap()
}

fn uri(dir: &tempfile::TempDir) -> String {
    dir.path().join("table.lance").to_str().unwrap().to_string()
}

async fn memory(dataset: &Dataset) -> Vec<Option<u64>> {
    let batch = dataset.scan().try_into_batch().await.unwrap();
    let column = batch.column_by_name(MEMORY_COLUMN).unwrap();
    column.as_primitive::<UInt64Type>().iter().collect()
}

#[tokio::test]
async fn a_writer_stores_the_memory_of_each_row() {
    let dir = tempfile::tempdir().unwrap();
    let mut writer = NdWriter::create(&uri(&dir), &WriteOptions::default())
        .await
        .unwrap();
    let rows = [1000, 0, 10].map(|values| Ok(row(values)));
    writer.write(stream::iter(rows)).await.unwrap();
    let dataset = writer.finish().await.unwrap();
    // 4 bytes per float, and 5 bytes plus a 4 byte offset for the string.
    assert_eq!(memory(&dataset).await, [Some(4009), Some(9), Some(49)]);
    // Readers leave the column out of the rows.
    let rows = scan(&dataset).try_collect().await.unwrap();
    assert!(rows[0].column_by_name(MEMORY_COLUMN).is_none());
}

#[tokio::test]
async fn compaction_plans_without_the_column_and_then_writes_it() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    // A table of an older writer: one fragment per row, without the column.
    let target = 1_000_000;
    for (i, values) in [100_000, 100_000, 100_000, 100_000].into_iter().enumerate() {
        let batch = encode_rows(&[row(values)]).unwrap();
        let mut metadata = batch.schema().metadata().clone();
        metadata.insert("lance_nd:format".to_string(), "1".to_string());
        let schema = Arc::new(batch.schema().as_ref().clone().with_metadata(metadata));
        let batch = batch.with_schema(schema.clone()).unwrap();
        let params = WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            mode: match i {
                0 => lance::dataset::WriteMode::Create,
                _ => lance::dataset::WriteMode::Append,
            },
            ..Default::default()
        };
        let reader = RecordBatchIterator::new([Ok(batch)], schema);
        Dataset::write(reader, &uri, Some(params)).await.unwrap();
    }
    // A writer of this release adds the column to the table.
    let mut writer = NdWriter::append(&uri, &WriteOptions::default())
        .await
        .unwrap();
    writer.write(stream::iter([Ok(row(10))])).await.unwrap();
    writer.finish().await.unwrap();

    let mut dataset = open(&uri).await.unwrap();
    assert_eq!(memory(&dataset).await[..4], [None, None, None, None]);
    // Rows of 400 KB: 2 fragments of 2 and 3 rows.
    let options = CompactOptions {
        target_fragment_bytes: target,
        ..Default::default()
    };
    let stats = compact(&mut dataset, &options).await.unwrap();
    assert_eq!(stats.fragments_removed, 5);
    assert_eq!(dataset.get_fragments().len(), 2);
    let sizes = memory(&dataset).await;
    assert!(sizes.iter().all(Option::is_some));
    assert_eq!(sizes[0], Some(400_009));

    // The next compaction plans from the column and finds no work.
    let version = dataset.version().version;
    compact(&mut dataset, &options).await.unwrap();
    assert_eq!(dataset.version().version, version);
}
