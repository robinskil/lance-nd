use std::sync::Arc;

use arrow::array::{Float32Array, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use futures::stream;
use lance::Dataset;
use lance_nd::{Error, NdWriter, WriteMode, WriteOptions, scan, variables, write};
use nd_arrow_array::{Attributes, Dimensions, NdArrowArray, NdRecordBatch};

fn uri(dir: &tempfile::TempDir) -> String {
    dir.path().join("table.lance").to_str().unwrap().to_string()
}

fn rows(
    batches: Vec<NdRecordBatch>,
) -> impl futures::Stream<Item = lance_nd::Result<NdRecordBatch>> {
    stream::iter(batches.into_iter().map(Ok))
}

/// A `(time)` Float64 variable `sst` with a title.
fn sst(times: usize, title: &str) -> NdRecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "sst",
        DataType::Float64,
        true,
    )]));
    let sst = NdArrowArray::try_new(
        Arc::new(Float64Array::from_iter_values((0..times).map(|i| i as f64))),
        Dimensions::try_new([("time", times)]).unwrap(),
    )
    .unwrap()
    .with_attributes(Attributes::new().with("units", "K"));
    NdRecordBatch::try_new_inferred(schema, vec![sst])
        .unwrap()
        .with_attributes(Attributes::new().with("title", title))
}

/// One variable `name` of type Float32 on `(station=2)`.
fn station_var(name: &str) -> NdRecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Float32, true)]));
    let array = NdArrowArray::try_new(
        Arc::new(Float32Array::from(vec![1.0, 2.0])),
        Dimensions::try_new([("station", 2)]).unwrap(),
    )
    .unwrap();
    NdRecordBatch::try_new_inferred(schema, vec![array]).unwrap()
}

/// `sst` as Float32, a type change against [`sst`].
fn sst_f32() -> NdRecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "sst",
        DataType::Float32,
        true,
    )]));
    let sst = NdArrowArray::try_new(
        Arc::new(Float32Array::from(vec![1.0])),
        Dimensions::try_new([("time", 1)]).unwrap(),
    )
    .unwrap();
    NdRecordBatch::try_new_inferred(schema, vec![sst]).unwrap()
}

async fn table_with_one_row(uri: &str) -> Dataset {
    write(uri, rows(vec![sst(2, "base")]), &WriteOptions::default())
        .await
        .unwrap()
}

fn overwrite() -> WriteOptions {
    WriteOptions {
        mode: WriteMode::Overwrite,
        ..Default::default()
    }
}

#[tokio::test]
async fn create_is_invisible_until_finish() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let mut writer = NdWriter::create(&uri, &WriteOptions::default())
        .await
        .unwrap();
    writer
        .write(rows(vec![sst(2, "a"), sst(3, "b")]))
        .await
        .unwrap();
    assert!(Dataset::open(&uri).await.is_err());
    let dataset = writer.finish().await.unwrap();
    assert_eq!(dataset.version().version, 1);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 2);
}

#[tokio::test]
async fn append_is_one_version() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let version = table_with_one_row(&uri).await.version().version;
    let mut writer = NdWriter::append(&uri, &WriteOptions::default())
        .await
        .unwrap();
    writer.write(rows(vec![sst(2, "a")])).await.unwrap();
    writer.write(rows(vec![sst(4, "b")])).await.unwrap();
    assert_eq!(
        Dataset::open(&uri).await.unwrap().version().version,
        version
    );
    let dataset = writer.finish().await.unwrap();
    assert_eq!(dataset.version().version, version + 1);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 3);
    let read = scan(&dataset).try_collect().await.unwrap();
    assert_eq!(read[2].column_by_name("sst").unwrap().len(), 4);
}

#[tokio::test]
async fn new_variable_in_second_write() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let mut writer = NdWriter::create(&uri, &WriteOptions::default())
        .await
        .unwrap();
    writer.write(rows(vec![sst(2, "a")])).await.unwrap();
    writer
        .write(rows(vec![station_var("extra")]))
        .await
        .unwrap();
    let dataset = writer.finish().await.unwrap();
    let names: Vec<String> = variables(&dataset)
        .unwrap()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, vec!["sst", "extra"]);
    let read = scan(&dataset).try_collect().await.unwrap();
    assert!(read[0].column_by_name("extra").is_none());
    assert!(read[0].column_by_name("sst").is_some());
    assert!(read[1].column_by_name("extra").is_some());
    assert_eq!(read[0].attributes(), &Attributes::new().with("title", "a"));
}

#[tokio::test]
async fn abort_leaves_table() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let version = table_with_one_row(&uri).await.version().version;

    let mut writer = NdWriter::append(&uri, &WriteOptions::default())
        .await
        .unwrap();
    writer
        .write(rows(vec![station_var("extra")]))
        .await
        .unwrap();
    writer.abort();
    let mut dropped = NdWriter::append(&uri, &WriteOptions::default())
        .await
        .unwrap();
    dropped.write(rows(vec![sst(2, "x")])).await.unwrap();
    drop(dropped);

    let dataset = Dataset::open(&uri).await.unwrap();
    assert_eq!(dataset.version().version, version);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 1);
    assert_eq!(variables(&dataset).unwrap().len(), 1);
}

#[tokio::test]
async fn create_without_rows_fails() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let writer = NdWriter::create(&uri, &WriteOptions::default())
        .await
        .unwrap();
    assert!(matches!(
        writer.finish().await,
        Err(Error::InvalidArgument(_))
    ));
    assert!(Dataset::open(&uri).await.is_err());

    let version = table_with_one_row(&uri).await.version().version;
    let writer = NdWriter::create(&uri, &overwrite()).await.unwrap();
    assert!(matches!(
        writer.finish().await,
        Err(Error::InvalidArgument(_))
    ));
    let dataset = Dataset::open(&uri).await.unwrap();
    assert_eq!(dataset.version().version, version);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 1);
}

#[tokio::test]
async fn append_without_rows_is_no_commit() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let version = table_with_one_row(&uri).await.version().version;
    let writer = NdWriter::append(&uri, &WriteOptions::default())
        .await
        .unwrap();
    let dataset = writer.finish().await.unwrap();
    assert_eq!(dataset.version().version, version);
    assert_eq!(
        Dataset::open(&uri).await.unwrap().version().version,
        version
    );
}

#[tokio::test]
async fn type_change_across_writes_fails() {
    // New table: the type change fails, and no table appears.
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let mut writer = NdWriter::create(&uri, &WriteOptions::default())
        .await
        .unwrap();
    writer.write(rows(vec![sst(2, "a")])).await.unwrap();
    assert!(writer.write(rows(vec![sst_f32()])).await.is_err());
    assert!(writer.finish().await.is_err());
    assert!(Dataset::open(&uri).await.is_err());

    // Existing table: the type change fails, and the table does not change.
    let version = table_with_one_row(&uri).await.version().version;
    let mut writer = NdWriter::append(&uri, &WriteOptions::default())
        .await
        .unwrap();
    writer
        .write(rows(vec![station_var("extra")]))
        .await
        .unwrap();
    assert!(writer.write(rows(vec![sst_f32()])).await.is_err());
    assert!(writer.finish().await.is_err());
    let dataset = Dataset::open(&uri).await.unwrap();
    assert_eq!(dataset.version().version, version);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 1);
}

#[tokio::test]
async fn create_existing_fails() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    table_with_one_row(&uri).await;
    assert!(matches!(
        NdWriter::create(&uri, &WriteOptions::default()).await,
        Err(Error::InvalidArgument(_))
    ));
}

#[tokio::test]
async fn append_missing_fails() {
    let dir = tempfile::tempdir().unwrap();
    assert!(matches!(
        NdWriter::append(&uri(&dir), &WriteOptions::default()).await,
        Err(Error::InvalidArgument(_))
    ));
}

#[tokio::test]
async fn overwrite_replaces_in_one_commit() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let version = table_with_one_row(&uri).await.version().version;
    let mut writer = NdWriter::create(&uri, &overwrite()).await.unwrap();
    writer
        .write(rows(vec![station_var("pressure")]))
        .await
        .unwrap();
    let dataset = writer.finish().await.unwrap();
    assert_eq!(dataset.version().version, version + 1);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 1);
    let names: Vec<String> = variables(&dataset)
        .unwrap()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, vec!["pressure"]);
}

#[tokio::test]
async fn concurrent_schema_change() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let version = table_with_one_row(&uri).await.version().version;
    let mut first = NdWriter::append(&uri, &WriteOptions::default())
        .await
        .unwrap();
    let mut second = NdWriter::append(&uri, &WriteOptions::default())
        .await
        .unwrap();
    first.write(rows(vec![station_var("one")])).await.unwrap();
    second.write(rows(vec![station_var("two")])).await.unwrap();
    first.finish().await.unwrap();
    // Lance 11 rejects the second schema change: two merges conflict.
    assert!(second.finish().await.is_err());
    let dataset = Dataset::open(&uri).await.unwrap();
    assert_eq!(dataset.version().version, version + 1);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 2);
}

#[tokio::test]
async fn concurrent_plain_appends_both_commit() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let version = table_with_one_row(&uri).await.version().version;
    let mut first = NdWriter::append(&uri, &WriteOptions::default())
        .await
        .unwrap();
    let mut second = NdWriter::append(&uri, &WriteOptions::default())
        .await
        .unwrap();
    first.write(rows(vec![sst(2, "one")])).await.unwrap();
    second.write(rows(vec![sst(3, "two")])).await.unwrap();
    first.finish().await.unwrap();
    let dataset = second.finish().await.unwrap();
    assert_eq!(dataset.version().version, version + 2);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 3);
}

#[tokio::test]
async fn create_fails_when_another_writer_created_the_table() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let mut first = NdWriter::create(&uri, &WriteOptions::default())
        .await
        .unwrap();
    let mut second = NdWriter::create(&uri, &WriteOptions::default())
        .await
        .unwrap();
    first
        .write(rows(vec![sst(2, "a"), sst(2, "b")]))
        .await
        .unwrap();
    second.write(rows(vec![sst(2, "c")])).await.unwrap();
    second.finish().await.unwrap();
    assert!(matches!(
        first.finish().await,
        Err(Error::InvalidArgument(_))
    ));
    let dataset = Dataset::open(&uri).await.unwrap();
    assert_eq!(dataset.version().version, 1);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 1);
}
