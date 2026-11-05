//! Tables on S3. These tests need the `aws` feature and an S3 server, for
//! example moto: set `LANCE_ND_S3_ENDPOINT` to its URL. Without it, each test
//! returns at once.
//!
//! `cargo test --features aws --test object_store`
#![cfg(feature = "aws")]

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use arrow::array::{Float32Array, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use futures::stream;
use lance::io::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry, StorageOptionsAccessor};
use lance_nd::datafusion::functions::core::expr_fn::get_field;
use lance_nd::datafusion::logical_expr::{col, lit};
use lance_nd::{
    CompactOptions, Error, NdWriter, StorageOptions, WriteOptions, append, cleanup_old_versions,
    compact, delete, open_with, scan, widen,
};
use nd_arrow_array::{Attributes, Dimensions, NdArrowArray, NdRecordBatch};
use object_store::{PutMode, PutPayload};

/// The S3 server, or `None` to skip the test.
fn endpoint() -> Option<String> {
    let endpoint = std::env::var("LANCE_ND_S3_ENDPOINT").ok();
    if endpoint.is_none() {
        eprintln!("LANCE_ND_S3_ENDPOINT is not set, the test does nothing");
    }
    endpoint
}

/// Test credentials for a local S3 server.
fn storage_options(endpoint: &str) -> StorageOptions {
    [
        ("aws_endpoint", endpoint),
        ("aws_access_key_id", "test"),
        ("aws_secret_access_key", "test"),
        ("aws_region", "us-east-1"),
        ("allow_http", "true"),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value.to_string()))
    .collect()
}

/// A new table uri for each test run.
fn table_uri(test: &str) -> String {
    let bucket = std::env::var("LANCE_ND_S3_BUCKET").unwrap_or("lance-nd-test".to_string());
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("s3://{bucket}/{test}-{nanos}/table.lance")
}

fn options(endpoint: &str) -> WriteOptions {
    WriteOptions {
        stats: true,
        storage_options: storage_options(endpoint),
        ..Default::default()
    }
}

/// One row with `sst` of `times` Float64 values and the global attribute `title`.
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
    .unwrap();
    NdRecordBatch::try_new_inferred(schema, vec![sst])
        .unwrap()
        .with_attributes(Attributes::new().with("title", title))
}

/// One row with `t` as Float32.
fn t32(value: f32) -> NdRecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("t", DataType::Float32, true)]));
    let t = NdArrowArray::try_new(
        Arc::new(Float32Array::from(vec![value])),
        Dimensions::try_new([("station", 1)]).unwrap(),
    )
    .unwrap();
    NdRecordBatch::try_new_inferred(schema, vec![t]).unwrap()
}

fn rows(
    batches: Vec<NdRecordBatch>,
) -> impl futures::Stream<Item = lance_nd::Result<NdRecordBatch>> {
    stream::iter(batches.into_iter().map(Ok))
}

/// Lance commits on S3 with a conditional put. The server must reject a
/// second put of the same key, or two commits can overwrite each other.
#[tokio::test]
async fn the_server_rejects_a_second_conditional_put() {
    let Some(endpoint) = endpoint() else { return };
    let params = ObjectStoreParams {
        storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
            storage_options(&endpoint),
        ))),
        ..Default::default()
    };
    let uri = table_uri("conditional-put").replace("table.lance", "probe");
    let registry = Arc::new(ObjectStoreRegistry::default());
    let (store, path) = ObjectStore::from_uri_and_params(registry, &uri, &params)
        .await
        .unwrap();
    let create = || PutMode::Create.into();
    store
        .inner
        .put_opts(&path, PutPayload::from_static(b"first"), create())
        .await
        .unwrap();
    let second = store
        .inner
        .put_opts(&path, PutPayload::from_static(b"second"), create())
        .await;
    assert!(
        matches!(second, Err(object_store::Error::AlreadyExists { .. })),
        "{second:?}"
    );
}

#[tokio::test]
async fn a_table_works_on_s3() {
    let Some(endpoint) = endpoint() else { return };
    let uri = table_uri("round-trip");
    let options = options(&endpoint);

    let mut writer = NdWriter::create(&uri, &options).await.unwrap();
    writer
        .write(rows(vec![sst(3, "a"), sst(4, "b"), t32(1.5)]))
        .await
        .unwrap();
    writer.finish().await.unwrap();

    // `open_with` needs the options. `append` then takes them from the table.
    let mut dataset = open_with(&uri, &options.storage_options).await.unwrap();
    append(
        &mut dataset,
        rows(vec![sst(2, "c")]),
        &WriteOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(scan(&dataset).try_collect().await.unwrap().len(), 4);

    // Only row "b" has an sst value above 2.5. The stats skip the others.
    let filter = get_field(col("sst"), "values").gt(lit(2.5));
    let kept = scan(&dataset)
        .filter_expr(filter)
        .try_collect()
        .await
        .unwrap();
    assert_eq!(kept.len(), 1);

    widen(&mut dataset, "t", &DataType::Float64).await.unwrap();
    assert_eq!(
        delete(&mut dataset, "__global_attrs__.title = 'b'")
            .await
            .unwrap(),
        1
    );
    compact(&mut dataset, &CompactOptions::default())
        .await
        .unwrap();
    cleanup_old_versions(&dataset, Duration::ZERO)
        .await
        .unwrap();

    let dataset = open_with(&uri, &options.storage_options).await.unwrap();
    let rows = scan(&dataset).try_collect().await.unwrap();
    assert_eq!(rows.len(), 3);
}

#[tokio::test]
async fn concurrent_writers_on_s3() {
    let Some(endpoint) = endpoint() else { return };
    let uri = table_uri("concurrent");
    let options = options(&endpoint);
    let mut writer = NdWriter::create(&uri, &options).await.unwrap();
    writer
        .write(rows(vec![sst(2, "a"), t32(1.0)]))
        .await
        .unwrap();
    let mut dataset = writer.finish().await.unwrap();
    let version = dataset.version().version;

    // Two plain appends from the same version both commit.
    let mut first = NdWriter::append(&uri, &options).await.unwrap();
    let mut second = NdWriter::append(&uri, &options).await.unwrap();
    first.write(rows(vec![sst(2, "b")])).await.unwrap();
    second.write(rows(vec![sst(2, "c")])).await.unwrap();
    let (first, second) = tokio::join!(first.finish(), second.finish());
    first.unwrap();
    let after = second.unwrap();
    assert_eq!(after.version().version, version + 2);

    // A writer from before a widen fails and adds no rows.
    let mut stale = NdWriter::append(&uri, &options).await.unwrap();
    stale.write(rows(vec![t32(2.0)])).await.unwrap();
    dataset.checkout_latest().await.unwrap();
    widen(&mut dataset, "t", &DataType::Float64).await.unwrap();
    let result = stale.finish().await;
    assert!(
        matches!(result, Err(Error::SchemaMismatch(_))),
        "{:?}",
        result.map(|_| ())
    );
    let dataset = open_with(&uri, &options.storage_options).await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 4);
}
