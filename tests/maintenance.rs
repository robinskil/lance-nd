//! Delete rows, compact the table and clean up old versions.

use std::sync::Arc;
use std::time::Duration;

use arrow::array::{ArrayRef, AsArray, Float64Array, Int64Array};
use arrow::datatypes::{Field, Int64Type, Schema};
use futures::stream;
use lance::Dataset;
use lance_nd::datafusion::functions::core::expr_fn::get_field;
use lance_nd::datafusion::logical_expr::{col, lit};
use lance_nd::{
    CompactOptions, NdWriter, WriteOptions, cleanup_old_versions, compact, delete, delete_expr,
    scan,
};
use nd_arrow_array::{Attributes, Dimensions, NdArrowArray, NdRecordBatch};

fn variable(name: &str, array: ArrayRef) -> (Field, NdArrowArray) {
    let dims = Dimensions::try_new([(format!("{name}_dim"), array.len())]).unwrap();
    let field = Field::new(name, array.data_type().clone(), true);
    (field, NdArrowArray::try_new(array, dims).unwrap())
}

/// Row `id` with the global attribute `id = r<id>`. Even rows have a large
/// `temp` with units `degF` in rows divisible by 4, else `degC`.
fn row(id: i64) -> NdRecordBatch {
    let mut variables = vec![variable("id", Arc::new(Int64Array::from(vec![id])))];
    if id % 2 == 0 {
        let values = Float64Array::from_iter_values((0..5000).map(|i| (id * 10_000 + i) as f64));
        let (field, temp) = variable("temp", Arc::new(values));
        let units = if id % 4 == 0 { "degF" } else { "degC" };
        variables.push((
            field,
            temp.with_attributes(Attributes::new().with("units", units)),
        ));
    }
    let (fields, arrays): (Vec<_>, Vec<_>) = variables.into_iter().unzip();
    NdRecordBatch::try_new_inferred(Arc::new(Schema::new(fields)), arrays)
        .unwrap()
        .with_attributes(Attributes::new().with("id", format!("r{id}")))
}

/// A table of `rows` rows, one fragment per row.
async fn table(dir: &tempfile::TempDir, rows: i64, stats: bool) -> Dataset {
    let uri = dir.path().join("table.lance").to_str().unwrap().to_string();
    let options = WriteOptions {
        stats,
        max_rows_per_group: 1,
        target_fragment_bytes: 1,
        ..Default::default()
    };
    let mut writer = NdWriter::create(&uri, &options).await.unwrap();
    writer
        .write(stream::iter((0..rows).map(|id| Ok(row(id)))))
        .await
        .unwrap();
    writer.finish().await.unwrap()
}

async fn ids(dataset: &Dataset) -> Vec<i64> {
    let rows = scan(dataset).try_collect().await.unwrap();
    rows.iter()
        .map(|row| {
            let id = row.column_by_name("id").expect("an id");
            id.values().as_primitive::<Int64Type>().value(0)
        })
        .collect()
}

#[tokio::test]
async fn delete_by_global_attribute() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(&dir, 4, false).await;
    let version = dataset.version().version;
    let deleted = delete(&mut dataset, "__global_attrs__.id = 'r1'")
        .await
        .unwrap();
    assert_eq!(deleted, 1);
    assert_eq!(ids(&dataset).await, [0, 2, 3]);
    assert!(dataset.version().version > version);
    let reopened = Dataset::open(dataset.uri()).await.unwrap();
    assert_eq!(ids(&reopened).await, [0, 2, 3]);
}

#[tokio::test]
async fn delete_expr_by_variable_attribute() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(&dir, 6, false).await;
    let units = get_field(get_field(col("temp"), "attrs"), "units");
    let deleted = delete_expr(&mut dataset, units.eq(lit("degF")))
        .await
        .unwrap();
    assert_eq!(deleted, 2);
    assert_eq!(ids(&dataset).await, [1, 2, 3, 5]);
}

#[tokio::test]
async fn delete_without_match() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(&dir, 3, false).await;
    let deleted = delete(&mut dataset, "__global_attrs__.id = 'nope'")
        .await
        .unwrap();
    assert_eq!(deleted, 0);
    assert_eq!(ids(&dataset).await, [0, 1, 2]);
}

#[tokio::test]
async fn bad_predicate_fails() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(&dir, 3, false).await;
    assert!(delete(&mut dataset, "nope = 1").await.is_err());
    assert_eq!(ids(&dataset).await, [0, 1, 2]);
}

#[tokio::test]
async fn compact_merges_fragments_and_drops_deleted_rows() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(&dir, 12, true).await;
    assert_eq!(dataset.get_fragments().len(), 12);
    delete(&mut dataset, "__global_attrs__.id IN ('r3', 'r4', 'r7')")
        .await
        .unwrap();
    // Lance drops a fragment when each of its rows is deleted.
    assert_eq!(dataset.get_fragments().len(), 9);
    let expected: Vec<NdRecordBatch> = scan(&dataset).try_collect().await.unwrap();

    let metrics = compact(&mut dataset, &CompactOptions::default())
        .await
        .unwrap();
    assert_eq!(metrics.fragments_removed, 9);
    assert_eq!(dataset.get_fragments().len(), 1);
    assert_eq!(dataset.count_deleted_rows().await.unwrap(), 0);
    let rows: Vec<NdRecordBatch> = scan(&dataset).try_collect().await.unwrap();
    assert_eq!(rows.len(), expected.len());
    for (row, want) in rows.iter().zip(&expected) {
        assert_eq!(row.schema(), want.schema());
        assert_eq!(row.columns(), want.columns());
        assert_eq!(row.attributes(), want.attributes());
    }

    // The stats still prune after the compaction.
    let filter = get_field(col("temp"), "values").gt(lit(100_000.0));
    let pruned = scan(&dataset)
        .filter_expr(filter)
        .try_collect()
        .await
        .unwrap();
    let pruned: Vec<i64> = pruned
        .iter()
        .map(|row| {
            row.column_by_name("id")
                .unwrap()
                .values()
                .as_primitive::<Int64Type>()
                .value(0)
        })
        .collect();
    assert_eq!(pruned, [10]);
}

#[tokio::test]
async fn compact_splits_a_large_fragment_by_size() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("table.lance").to_str().unwrap().to_string();
    let mut writer = NdWriter::create(&uri, &WriteOptions::default())
        .await
        .unwrap();
    writer
        .write(stream::iter((0..12).map(|id| Ok(row(id)))))
        .await
        .unwrap();
    let mut dataset = writer.finish().await.unwrap();
    assert_eq!(dataset.get_fragments().len(), 1);
    let expected: Vec<NdRecordBatch> = scan(&dataset).try_collect().await.unwrap();

    // Far below the size of one row: each row becomes a fragment.
    let options = CompactOptions {
        target_fragment_bytes: 16,
        ..Default::default()
    };
    let metrics = compact(&mut dataset, &options).await.unwrap();
    assert_eq!(metrics.fragments_removed, 1);
    assert_eq!(dataset.get_fragments().len(), 12);
    let rows: Vec<NdRecordBatch> = scan(&dataset).try_collect().await.unwrap();
    assert_eq!(rows.len(), expected.len());
    for (row, want) in rows.iter().zip(&expected) {
        assert_eq!(row.columns(), want.columns());
        assert_eq!(row.attributes(), want.attributes());
    }
}

#[tokio::test]
async fn compact_keeps_fragments_near_the_target() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(&dir, 12, false).await;
    let options = CompactOptions {
        target_fragment_bytes: 1,
        ..Default::default()
    };
    let metrics = compact(&mut dataset, &options).await.unwrap();
    assert_eq!(metrics.fragments_removed, 0);
    assert_eq!(dataset.get_fragments().len(), 12);
}

#[tokio::test]
async fn cleanup_removes_old_versions() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(&dir, 6, false).await;
    delete(&mut dataset, "__global_attrs__.id = 'r0'")
        .await
        .unwrap();
    compact(&mut dataset, &CompactOptions::default())
        .await
        .unwrap();
    let stats = cleanup_old_versions(&dataset, Duration::ZERO)
        .await
        .unwrap();
    assert!(stats.old_versions > 0, "{stats:?}");
    assert!(stats.bytes_removed > 0, "{stats:?}");
    assert_eq!(ids(&dataset).await, [1, 2, 3, 4, 5]);
    let reopened = Dataset::open(dataset.uri()).await.unwrap();
    assert_eq!(ids(&reopened).await, [1, 2, 3, 4, 5]);
}

#[tokio::test]
async fn compact_rewrites_a_fragment_with_deleted_rows() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("table.lance").to_str().unwrap().to_string();
    let mut writer = NdWriter::create(&uri, &WriteOptions::default())
        .await
        .unwrap();
    writer
        .write(stream::iter((0..8).map(|id| Ok(row(id)))))
        .await
        .unwrap();
    let mut dataset = writer.finish().await.unwrap();
    assert_eq!(dataset.get_fragments().len(), 1);
    delete(&mut dataset, "__global_attrs__.id IN ('r2', 'r5')")
        .await
        .unwrap();
    assert_eq!(dataset.count_deleted_rows().await.unwrap(), 2);

    let metrics = compact(&mut dataset, &CompactOptions::default())
        .await
        .unwrap();
    assert_eq!(metrics.fragments_removed, 1);
    assert_eq!(dataset.get_fragments().len(), 1);
    assert_eq!(dataset.count_deleted_rows().await.unwrap(), 0);
    assert_eq!(ids(&dataset).await, [0, 1, 3, 4, 6, 7]);
}

#[tokio::test]
async fn delete_by_value() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(&dir, 4, false).await;
    let deleted = delete(&mut dataset, "array_has(temp.values, 20001.0)")
        .await
        .unwrap();
    assert_eq!(deleted, 1);
    assert_eq!(ids(&dataset).await, [0, 1, 3]);
}
