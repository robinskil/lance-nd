use std::sync::Arc;

use arrow::array::{Float32Array, Float64Array, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use futures::TryStreamExt;
use futures::stream;
use lance::Dataset;
use lance_nd::encode_rows;
use lance_nd::has_stats;
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
async fn type_change_rejects_only_its_rows() {
    // New table: the type change is rejected, and the other rows commit.
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let mut writer = NdWriter::create(&uri, &WriteOptions::default())
        .await
        .unwrap();
    writer.write(rows(vec![sst(2, "a")])).await.unwrap();
    let error = writer.write(rows(vec![sst_f32()])).await.unwrap_err();
    assert!(matches!(error, Error::SchemaMismatch(_)), "{error}");
    writer.write(rows(vec![sst(3, "b")])).await.unwrap();
    let dataset = writer.finish().await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 2);

    // Existing table: the type change is rejected, and the other rows commit.
    let version = dataset.version().version;
    let mut writer = NdWriter::append(&uri, &WriteOptions::default())
        .await
        .unwrap();
    writer
        .write(rows(vec![station_var("extra")]))
        .await
        .unwrap();
    assert!(writer.write(rows(vec![sst_f32()])).await.is_err());
    let dataset = writer.finish().await.unwrap();
    // One commit adds the new variable, one appends the rows.
    assert_eq!(dataset.version().version, version + 2);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 3);
}

/// `sst(2, title)` with the attribute `resolution` on `sst`.
fn with_resolution(title: &str, resolution: Attributes) -> NdRecordBatch {
    let row = sst(2, title);
    let column = row.column_by_name("sst").unwrap().clone();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "sst",
        DataType::Float64,
        true,
    )]));
    NdRecordBatch::try_new_inferred(schema, vec![column.with_attributes(resolution)])
        .unwrap()
        .with_attributes(Attributes::new().with("title", title))
}

#[tokio::test]
async fn attribute_type_conflict_rejects_only_its_row() {
    let dir = tempfile::tempdir().unwrap();
    let mut writer = NdWriter::create(&uri(&dir), &WriteOptions::default())
        .await
        .unwrap();
    let scalar = |title| with_resolution(title, Attributes::new().with("resolution", 0.5f64));
    let list = with_resolution("b", Attributes::new().with("resolution", vec![1.0f64, 2.0]));
    writer
        .write_batch(encode_rows(&[scalar("a")]).unwrap())
        .await
        .unwrap();
    let error = writer
        .write_batch(encode_rows(&[list]).unwrap())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("resolution"), "{error}");
    writer
        .write_batch(encode_rows(&[scalar("c")]).unwrap())
        .await
        .unwrap();
    let dataset = writer.finish().await.unwrap();
    let rows = scan(&dataset).try_collect().await.unwrap();
    let titles: Vec<String> = rows
        .iter()
        .map(|row| format!("{:?}", row.attributes()))
        .collect();
    assert_eq!(rows.len(), 2);
    assert!(
        titles[0].contains("\"a\"") && titles[1].contains("\"c\""),
        "{titles:?}"
    );
}

#[tokio::test]
async fn create_existing_fails() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    table_with_one_row(&uri).await;
    assert!(matches!(
        NdWriter::create(&uri, &WriteOptions::default()).await,
        Err(Error::TableExists(_))
    ));
}

#[tokio::test]
async fn append_missing_fails() {
    let dir = tempfile::tempdir().unwrap();
    assert!(matches!(
        NdWriter::append(&uri(&dir), &WriteOptions::default()).await,
        Err(Error::TableNotFound(_))
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
    // The first writer adds its variable, then appends its rows.
    assert_eq!(dataset.version().version, version + 2);
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
    assert!(matches!(first.finish().await, Err(Error::TableExists(_))));
    let dataset = Dataset::open(&uri).await.unwrap();
    assert_eq!(dataset.version().version, 1);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 1);
}

/// Write each row in its own `write` call, then commit.
async fn write_one_by_one(
    uri: &str,
    batches: Vec<NdRecordBatch>,
    options: &WriteOptions,
) -> Dataset {
    let mut writer = NdWriter::create(uri, options).await.unwrap();
    for batch in batches {
        writer.write(rows(vec![batch])).await.unwrap();
    }
    writer.finish().await.unwrap()
}

async fn read_all(dataset: &Dataset) -> Vec<NdRecordBatch> {
    scan(dataset)
        .try_into_stream()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap()
}

#[tokio::test]
async fn small_writes_share_one_fragment() {
    let dir = tempfile::tempdir().unwrap();
    let batches = (1..=20).map(|i| sst(i, "a")).collect();
    let dataset = write_one_by_one(&uri(&dir), batches, &WriteOptions::default()).await;
    assert_eq!(dataset.get_fragments().len(), 1);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 20);
}

#[tokio::test]
async fn target_bytes_splits_fragments() {
    let dir = tempfile::tempdir().unwrap();
    let options = WriteOptions {
        target_fragment_bytes: 1,
        ..Default::default()
    };
    let batches = (1..=20).map(|i| sst(i, "a")).collect();
    let dataset = write_one_by_one(&uri(&dir), batches, &options).await;
    assert_eq!(dataset.get_fragments().len(), 20);
}

#[tokio::test]
async fn parallel_writes_keep_row_order() {
    let dir = tempfile::tempdir().unwrap();
    let options = WriteOptions {
        target_fragment_bytes: 1,
        write_parallelism: 4,
        ..Default::default()
    };
    let batches = (1..=40).map(|i| sst(i, "a")).collect();
    let dataset = write_one_by_one(&uri(&dir), batches, &options).await;
    let lengths: Vec<usize> = read_all(&dataset)
        .await
        .iter()
        .map(|row| row.column_by_name("sst").unwrap().len())
        .collect();
    assert_eq!(lengths, (1..=40).collect::<Vec<_>>());
}

#[tokio::test]
async fn buffered_rows_with_other_variables() {
    let dir = tempfile::tempdir().unwrap();
    let batches = vec![sst(2, "a"), station_var("pressure"), sst(3, "b")];
    let dataset = write_one_by_one(&uri(&dir), batches, &WriteOptions::default()).await;
    assert_eq!(dataset.get_fragments().len(), 1);
    let names: Vec<Vec<String>> = read_all(&dataset)
        .await
        .iter()
        .map(|row| {
            row.schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect()
        })
        .collect();
    assert_eq!(names, [vec!["sst"], vec!["pressure"], vec!["sst"]]);
}

#[tokio::test]
async fn write_batch_takes_encoded_rows() {
    let dir = tempfile::tempdir().unwrap();
    let options = WriteOptions {
        stats: true,
        ..Default::default()
    };
    let mut writer = NdWriter::create(&uri(&dir), &options).await.unwrap();
    let batch = encode_rows(&[sst(2, "a"), station_var("pressure")]).unwrap();
    writer.write_batch(batch).await.unwrap();
    writer.write(rows(vec![sst(4, "b")])).await.unwrap();
    let dataset = writer.finish().await.unwrap();
    assert!(has_stats(&Schema::from(dataset.schema())));
    assert!(dataset.schema().field("__stats__:sst").is_some());
    assert!(dataset.schema().field("__stats__:pressure").is_some());
    let read = read_all(&dataset).await;
    assert_eq!(read.len(), 3);
    assert_eq!(read[0].column_by_name("sst").unwrap().len(), 2);
    assert!(read[1].column_by_name("sst").is_none());
    assert_eq!(read[2].column_by_name("sst").unwrap().len(), 4);
}

#[tokio::test]
async fn write_batch_rejects_bad_columns() {
    let dir = tempfile::tempdir().unwrap();
    let mut writer = NdWriter::create(&uri(&dir), &WriteOptions::default())
        .await
        .unwrap();
    let plain = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)])),
        vec![Arc::new(Int32Array::from(vec![1]))],
    )
    .unwrap();
    assert!(writer.write_batch(plain).await.is_err());

    let mut writer = NdWriter::create(&uri(&dir), &WriteOptions::default())
        .await
        .unwrap();
    let encoded = encode_rows(&[sst(2, "a")]).unwrap();
    let mut fields: Vec<Field> = encoded
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    let mut columns = encoded.columns().to_vec();
    fields.push(fields[0].clone().with_name("__stats__"));
    columns.push(columns[0].clone());
    let reserved =
        RecordBatch::try_new(Arc::new(Schema::new(fields.clone())), columns.clone()).unwrap();
    assert!(writer.write_batch(reserved).await.is_err());

    let global = Schema::new(vec![fields[0].clone().with_name("__global_attrs__")]);
    let global = RecordBatch::try_new(Arc::new(global), vec![columns[0].clone()]).unwrap();
    let error = writer.write_batch(global).await.unwrap_err();
    assert!(error.to_string().contains("__global_attrs__"), "{error}");
}

/// A `(prof, level)` Float32 variable `psal` with nulls in its values.
fn psal(levels: usize) -> NdRecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "psal",
        DataType::Float32,
        true,
    )]));
    let values = (0..20 * levels).map(|i| (i % 7 != 0).then_some(i as f32));
    let array = NdArrowArray::try_new(
        Arc::new(Float32Array::from_iter(values)),
        Dimensions::try_new([("prof", 20), ("level", levels)]).unwrap(),
    )
    .unwrap();
    NdRecordBatch::try_new_inferred(schema, vec![array]).unwrap()
}

#[tokio::test]
async fn absent_large_variable_in_buffered_batches() {
    let dir = tempfile::tempdir().unwrap();
    let batches = (0..40)
        .map(|i| {
            if i == 0 {
                station_var("pressure")
            } else {
                psal(5000 + i)
            }
        })
        .collect();
    let dataset = write_one_by_one(&uri(&dir), batches, &WriteOptions::default()).await;
    assert_eq!(dataset.get_fragments().len(), 1);
    let read = read_all(&dataset).await;
    assert_eq!(read.len(), 40);
    assert!(read[0].column_by_name("psal").is_none());
    assert_eq!(read[1].column_by_name("psal").unwrap().len(), 20 * 5001);
}

#[tokio::test]
async fn absent_variable_in_a_fragment_of_many_rows() {
    let dir = tempfile::tempdir().unwrap();
    let batches = (0..2000)
        .map(|i| {
            if i % 500 == 0 {
                station_var("pressure")
            } else {
                psal(100 + i % 7)
            }
        })
        .collect();
    let dataset = write_one_by_one(&uri(&dir), batches, &WriteOptions::default()).await;
    assert_eq!(dataset.get_fragments().len(), 1);
    assert_eq!(read_all(&dataset).await.len(), 2000);
}

/// Row `i`: `v<i % 40>` as Float64, `w<i % 7>` as Int32 with the attribute
/// `k<i % 5> = i`, and the global attribute `id = i`.
fn varied_row(i: usize) -> NdRecordBatch {
    let v = format!("v{}", i % 40);
    let w = format!("w{}", i % 7);
    let schema = Arc::new(Schema::new(vec![
        Field::new(&v, DataType::Float64, true),
        Field::new(&w, DataType::Int32, true),
    ]));
    let v_array = NdArrowArray::try_new(
        Arc::new(Float64Array::from(vec![i as f64, i as f64 + 0.5])),
        Dimensions::try_new([("n", 2)]).unwrap(),
    )
    .unwrap();
    let w_array = NdArrowArray::try_new(
        Arc::new(Int32Array::from(vec![i as i32])),
        Dimensions::try_new([("m", 1)]).unwrap(),
    )
    .unwrap()
    .with_attributes(Attributes::new().with(format!("k{}", i % 5).as_str(), i as i64));
    NdRecordBatch::try_new_inferred(schema, vec![v_array, w_array])
        .unwrap()
        .with_attributes(Attributes::new().with("id", i as i64))
}

#[tokio::test]
async fn many_varying_variables_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let options = WriteOptions {
        stats: true,
        target_fragment_bytes: 64 << 10,
        ..Default::default()
    };
    let rows_in: Vec<NdRecordBatch> = (0..300).map(varied_row).collect();
    let dataset = write_one_by_one(&uri(&dir), rows_in.clone(), &options).await;
    assert!(dataset.get_fragments().len() > 2);
    let rows_out = read_all(&dataset).await;
    assert_eq!(rows_out.len(), rows_in.len());
    for (i, (out, expected)) in rows_out.iter().zip(&rows_in).enumerate() {
        assert_eq!(out.attributes(), expected.attributes(), "row {i}");
        assert_eq!(out.num_columns(), 2, "row {i}");
        for field in expected.schema().fields() {
            assert_eq!(
                out.column_by_name(field.name()),
                expected.column_by_name(field.name()),
                "row {i} {}",
                field.name()
            );
        }
    }
    // The stats of a variable that only some fragments hold.
    let filter = lance_nd::datafusion::functions::core::expr_fn::get_field(
        lance_nd::datafusion::logical_expr::col("v3"),
        "values",
    )
    .gt(lance_nd::datafusion::logical_expr::lit(200.0));
    let pruned = scan(&dataset)
        .filter_expr(filter)
        .try_collect()
        .await
        .unwrap();
    let ids: Vec<String> = pruned
        .iter()
        .map(|row| format!("{:?}", row.attributes()))
        .collect();
    assert_eq!(pruned.len(), 3, "{ids:?}");
}

#[tokio::test]
async fn append_with_new_fields_commits_new_rows_only() {
    use lance::dataset::transaction::Operation;
    let dir = tempfile::tempdir().unwrap();
    let options = WriteOptions {
        stats: true,
        target_fragment_bytes: 64 << 10,
        ..Default::default()
    };
    let first: Vec<NdRecordBatch> = (0..10).map(varied_row).collect();
    let dataset = write_one_by_one(&uri(&dir), first.clone(), &options).await;
    let version = dataset.version().version;

    // New variables, attribute keys and stats children, in several fragments.
    let later: Vec<NdRecordBatch> = (10..300).map(varied_row).collect();
    let mut writer = NdWriter::append(&uri(&dir), &options).await.unwrap();
    for row in &later {
        writer.write(rows(vec![row.clone()])).await.unwrap();
    }
    let dataset = writer.finish().await.unwrap();
    let transaction = dataset.read_transaction().await.unwrap().unwrap();
    // An Update with new fragments only, which conflicts with a widen.
    assert!(
        matches!(
            &transaction.operation,
            Operation::Update { updated_fragments, removed_fragment_ids, new_fragments, .. }
                if updated_fragments.is_empty()
                    && removed_fragment_ids.is_empty()
                    && !new_fragments.is_empty()
        ),
        "{:?}",
        transaction.operation
    );
    assert!(dataset.version().version > version + 1);

    // A stats filter on a variable that only later fragments have.
    let filter = lance_nd::datafusion::functions::core::expr_fn::get_field(
        lance_nd::datafusion::logical_expr::col("v33"),
        "values",
    )
    .gt(lance_nd::datafusion::logical_expr::lit(100.0));
    let pruned = scan(&dataset)
        .filter_expr(filter)
        .try_collect()
        .await
        .unwrap();
    assert_eq!(pruned.len(), 5);

    let rows_in: Vec<NdRecordBatch> = first.into_iter().chain(later).collect();
    let rows_out = read_all(&dataset).await;
    assert_eq!(rows_out.len(), rows_in.len());
    for (i, (out, expected)) in rows_out.iter().zip(&rows_in).enumerate() {
        assert_eq!(out.attributes(), expected.attributes(), "row {i}");
        for field in expected.schema().fields() {
            assert_eq!(
                out.column_by_name(field.name()),
                expected.column_by_name(field.name()),
                "row {i} {}",
                field.name()
            );
        }
    }
}

/// The ids of the leaf fields of `field`, as a data file lists them.
fn field_ids(field: &lance_core::datatypes::Field) -> Vec<i32> {
    if field.children.is_empty() {
        return vec![field.id];
    }
    field.children.iter().flat_map(field_ids).collect()
}

#[tokio::test]
async fn fragments_hold_only_their_stats_columns() {
    let dir = tempfile::tempdir().unwrap();
    let options = WriteOptions {
        stats: true,
        target_fragment_bytes: 1,
        ..Default::default()
    };
    let rows_in = vec![sst(2, "a"), station_var("pressure")];
    let dataset = write_one_by_one(&uri(&dir), rows_in, &options).await;
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 2);
    let pressure = field_ids(dataset.schema().field("__stats__:pressure").unwrap());
    let sst_stats = field_ids(dataset.schema().field("__stats__:sst").unwrap());
    let ids = |i: usize| -> Vec<i32> {
        fragments[i]
            .metadata()
            .files
            .iter()
            .flat_map(|file| file.fields.iter().copied())
            .collect()
    };
    assert!(sst_stats.iter().all(|id| ids(0).contains(id)));
    assert!(pressure.iter().all(|id| !ids(0).contains(id)));
    assert!(pressure.iter().all(|id| ids(1).contains(id)));
    assert!(sst_stats.iter().all(|id| !ids(1).contains(id)));
}

#[tokio::test]
async fn variable_name_with_a_colon() {
    let dir = tempfile::tempdir().unwrap();
    let options = WriteOptions {
        stats: true,
        ..Default::default()
    };
    let dataset = write_one_by_one(&uri(&dir), vec![station_var("TEMP:qc")], &options).await;
    assert!(dataset.schema().field("__stats__:TEMP:qc").is_some());
    assert_eq!(
        lance_nd::stats_variable("__stats__:TEMP:qc"),
        Some("TEMP:qc")
    );
    let values = lance_nd::datafusion::functions::core::expr_fn::get_field(
        lance_nd::datafusion::logical_expr::col(
            lance_nd::datafusion::common::Column::new_unqualified("TEMP:qc"),
        ),
        "values",
    );
    let low = values
        .clone()
        .gt(lance_nd::datafusion::logical_expr::lit(0.0));
    let high = values.gt(lance_nd::datafusion::logical_expr::lit(1000.0));
    assert_eq!(
        scan(&dataset)
            .filter_expr(low)
            .try_collect()
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(
        scan(&dataset)
            .filter_expr(high)
            .try_collect()
            .await
            .unwrap()
            .is_empty()
    );
}

/// Row `i` with 20,000 zeros: 160 KB in memory, a few bytes on disk.
fn zeros_row(i: usize) -> NdRecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
    let values = NdArrowArray::try_new(
        Arc::new(Float64Array::from(vec![0.0; 20_000])),
        Dimensions::try_new([("n", 20_000)]).unwrap(),
    )
    .unwrap();
    NdRecordBatch::try_new_inferred(schema, vec![values])
        .unwrap()
        .with_attributes(Attributes::new().with("id", i as i64))
}

#[tokio::test]
async fn fragment_target_learns_the_size_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let options = WriteOptions {
        target_fragment_bytes: 256 << 10,
        write_parallelism: 1,
        ..Default::default()
    };
    let rows_in: Vec<NdRecordBatch> = (0..200).map(zeros_row).collect();
    let dataset = write_one_by_one(&uri(&dir), rows_in, &options).await;
    // By memory size, a fragment would end every two rows.
    let fragments = dataset.get_fragments().len();
    assert!((2..=30).contains(&fragments), "{fragments} fragments");
    assert_eq!(dataset.count_rows(None).await.unwrap(), 200);
}

#[tokio::test]
async fn a_dropped_writer_stops_its_writes_and_commits_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let before = table_with_one_row(&uri).await;
    // Each row is a fragment write, and all of them can run at once.
    let options = WriteOptions {
        target_fragment_bytes: 1,
        write_parallelism: 64,
        ..Default::default()
    };
    let mut writer = NdWriter::append(&uri, &options).await.unwrap();
    let many: Vec<NdRecordBatch> = (0..20).map(|i| sst(i + 2, "x")).collect();
    writer.write(rows(many)).await.unwrap();
    drop(writer);
    let dataset = Dataset::open(&uri).await.unwrap();
    assert_eq!(dataset.version().version, before.version().version);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 1);
}
