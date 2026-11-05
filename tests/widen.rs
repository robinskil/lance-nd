//! `widen`: change the value type of a variable in the table.

use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, Float32Array, Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Float64Type, Schema};
use futures::stream;
use lance::Dataset;
use lance_nd::datafusion::functions::core::expr_fn::get_field;
use lance_nd::datafusion::logical_expr::{col, lit};
use lance_nd::{Error, NdWriter, WriteOptions, scan, widen};
use nd_arrow_array::{Dimensions, NdArrowArray, NdRecordBatch};

/// One row with the 1-D variable `name`.
fn row(name: &str, values: ArrayRef) -> NdRecordBatch {
    let dims = Dimensions::try_new([(format!("{name}_dim"), values.len())]).unwrap();
    let field = Field::new(name, values.data_type().clone(), true);
    let array = NdArrowArray::try_new(values, dims).unwrap();
    NdRecordBatch::try_new_inferred(Arc::new(Schema::new(vec![field])), vec![array]).unwrap()
}

/// Rows: `v` as Float32 [1.5, 2.5], `w` only, `v` as Float32 [7.0].
async fn table(dir: &tempfile::TempDir, stats: bool) -> Dataset {
    let uri = dir.path().join("table.lance").to_str().unwrap().to_string();
    let options = WriteOptions {
        stats,
        ..Default::default()
    };
    let rows = vec![
        row("v", Arc::new(Float32Array::from(vec![1.5f32, 2.5]))),
        row("w", Arc::new(Int32Array::from(vec![3]))),
        row("v", Arc::new(Float32Array::from(vec![7.0f32]))),
    ];
    let mut writer = NdWriter::create(&uri, &options).await.unwrap();
    writer
        .write(stream::iter(rows.into_iter().map(Ok)))
        .await
        .unwrap();
    writer.finish().await.unwrap()
}

fn value_type(dataset: &Dataset, variable: &str) -> DataType {
    let schema = Schema::from(dataset.schema());
    let DataType::Struct(fields) = schema.field_with_name(variable).unwrap().data_type() else {
        panic!("an nd column is a struct");
    };
    let DataType::List(item) = fields.find("values").unwrap().1.data_type() else {
        panic!("values are a list");
    };
    item.data_type().clone()
}

fn stats_type(dataset: &Dataset, variable: &str) -> DataType {
    let schema = Schema::from(dataset.schema());
    let column = format!("__stats__:{variable}");
    let DataType::Struct(bounds) = schema.field_with_name(&column).unwrap().data_type() else {
        panic!("a stats column is a struct");
    };
    bounds.find("min").unwrap().1.data_type().clone()
}

/// The `v` values of each row, `None` for a row without `v`.
async fn v_values(dataset: &Dataset) -> Vec<Option<Vec<f64>>> {
    let rows = scan(dataset).try_collect().await.unwrap();
    rows.iter()
        .map(|row| {
            row.column_by_name("v")
                .map(|v| v.values().as_primitive::<Float64Type>().values().to_vec())
        })
        .collect()
}

#[tokio::test]
async fn widen_keeps_values_and_stats() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(&dir, true).await;
    widen(&mut dataset, "v", &DataType::Float64).await.unwrap();
    assert_eq!(value_type(&dataset, "v"), DataType::Float64);
    assert_eq!(stats_type(&dataset, "v"), DataType::Float64);
    assert_eq!(stats_type(&dataset, "w"), DataType::Int32);
    assert_eq!(
        v_values(&dataset).await,
        [Some(vec![1.5, 2.5]), None, Some(vec![7.0])]
    );
    let filter = get_field(col("v"), "values").gt(lit(5.0));
    let rows = scan(&dataset)
        .filter_expr(filter)
        .try_collect()
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);

    // The table now takes Float64 rows without a cast.
    let mut writer = NdWriter::append(dataset.uri(), &WriteOptions::default())
        .await
        .unwrap();
    writer
        .write(stream::iter([Ok(row(
            "v",
            Arc::new(Float64Array::from(vec![0.1])),
        ))]))
        .await
        .unwrap();
    let dataset = writer.finish().await.unwrap();
    assert_eq!(v_values(&dataset).await[3], Some(vec![0.1]));
}

#[tokio::test]
async fn widen_without_stats() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(&dir, false).await;
    widen(&mut dataset, "v", &DataType::Float64).await.unwrap();
    assert_eq!(value_type(&dataset, "v"), DataType::Float64);
    assert_eq!(v_values(&dataset).await[2], Some(vec![7.0]));
}

#[tokio::test]
async fn widen_to_the_same_type_does_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(&dir, true).await;
    let version = dataset.version().version;
    widen(&mut dataset, "v", &DataType::Float32).await.unwrap();
    assert_eq!(dataset.version().version, version);
}

#[tokio::test]
async fn widen_rejects_other_casts_and_names() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(&dir, true).await;
    let version = dataset.version().version;
    for (variable, to) in [
        ("v", DataType::Float16),
        ("v", DataType::Int64),
        ("w", DataType::Utf8),
        ("nope", DataType::Float64),
        ("__global_attrs__", DataType::Float64),
        ("__stats__", DataType::Float64),
    ] {
        let error = widen(&mut dataset, variable, &to).await;
        assert!(
            matches!(error, Err(Error::InvalidArgument(_))),
            "{variable} {to}: {error:?}"
        );
    }
    assert_eq!(dataset.version().version, version);
}

/// The writer opened the table before the widen and commits after it.
#[tokio::test]
async fn writer_from_before_a_widen_loses_no_values() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(&dir, true).await;
    let mut writer = NdWriter::append(dataset.uri(), &WriteOptions::default())
        .await
        .unwrap();
    let rows = vec![row("v", Arc::new(Float32Array::from(vec![9.5f32])))];
    writer
        .write(stream::iter(rows.into_iter().map(Ok)))
        .await
        .unwrap();
    widen(&mut dataset, "v", &DataType::Float64).await.unwrap();
    let result = writer.finish().await;
    let dataset = Dataset::open(dataset.uri()).await.unwrap();
    let values = v_values(&dataset).await;
    // The rows hold the old field ids of `v`, so the commit fails and adds no rows.
    assert!(
        matches!(&result, Err(Error::SchemaMismatch(message)) if message.contains("'v")),
        "{:?}",
        result.map(|_| ())
    );
    assert_eq!(values.len(), 3);
    assert_eq!(values[0], Some(vec![1.5, 2.5]));
}

/// An append from a table handle older than the widen.
#[tokio::test]
async fn append_from_before_a_widen_loses_no_values() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(&dir, true).await;
    let mut stale = dataset.clone();
    widen(&mut dataset, "v", &DataType::Float64).await.unwrap();
    let rows = vec![row("v", Arc::new(Float32Array::from(vec![9.5f32])))];
    let result = lance_nd::append(
        &mut stale,
        stream::iter(rows.into_iter().map(Ok)),
        &WriteOptions::default(),
    )
    .await;
    let dataset = Dataset::open(dataset.uri()).await.unwrap();
    let values = v_values(&dataset).await;
    // The append starts from the widened table, which rejects Float32 values.
    assert!(
        matches!(result, Err(Error::SchemaMismatch(_))),
        "{result:?}"
    );
    assert_eq!(values.len(), 3);
}

mod promote {
    use super::*;
    use lance_nd::{Casting, WriteMode, write};

    fn options(casting: Casting) -> WriteOptions {
        WriteOptions {
            stats: true,
            casting,
            ..Default::default()
        }
    }

    fn one(
        name: &str,
        values: ArrayRef,
    ) -> futures::stream::Iter<std::vec::IntoIter<lance_nd::Result<NdRecordBatch>>> {
        stream::iter(vec![Ok(row(name, values))])
    }

    #[tokio::test]
    async fn write_fn_widens_the_table() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("table.lance").to_str().unwrap().to_string();
        let narrow = one("v", Arc::new(Float32Array::from(vec![1.5f32])));
        write(&uri, narrow, &options(Casting::No)).await.unwrap();
        let wide = one("v", Arc::new(Float64Array::from(vec![0.1])));
        let append = WriteOptions {
            mode: WriteMode::Append,
            ..options(Casting::Promote)
        };
        let dataset = write(&uri, wide, &append).await.unwrap();
        assert_eq!(value_type(&dataset, "v"), DataType::Float64);
        assert_eq!(stats_type(&dataset, "v"), DataType::Float64);
        assert_eq!(v_values(&dataset).await, [Some(vec![1.5]), Some(vec![0.1])]);
    }

    #[tokio::test]
    async fn append_writer_widens_before_its_first_rows() {
        let dir = tempfile::tempdir().unwrap();
        let dataset = table(&dir, true).await;
        let mut writer = NdWriter::append(dataset.uri(), &options(Casting::Promote))
            .await
            .unwrap();
        writer
            .write(one("v", Arc::new(Float64Array::from(vec![0.1]))))
            .await
            .unwrap();
        let dataset = writer.finish().await.unwrap();
        assert_eq!(value_type(&dataset, "v"), DataType::Float64);
        assert_eq!(
            v_values(&dataset).await,
            [Some(vec![1.5, 2.5]), None, Some(vec![7.0]), Some(vec![0.1])]
        );
    }

    #[tokio::test]
    async fn writer_commits_its_rows_then_widens() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("table.lance").to_str().unwrap().to_string();
        let mut writer = NdWriter::create(&uri, &options(Casting::Promote))
            .await
            .unwrap();
        writer
            .write(one("v", Arc::new(Float32Array::from(vec![1.5f32]))))
            .await
            .unwrap();
        writer
            .write(one("w", Arc::new(Int32Array::from(vec![3]))))
            .await
            .unwrap();
        writer
            .write(one("v", Arc::new(Float64Array::from(vec![0.1]))))
            .await
            .unwrap();
        let dataset = writer.finish().await.unwrap();
        assert_eq!(value_type(&dataset, "v"), DataType::Float64);
        assert_eq!(stats_type(&dataset, "v"), DataType::Float64);
        assert_eq!(
            v_values(&dataset).await,
            [Some(vec![1.5]), None, Some(vec![0.1])]
        );
        let filter = get_field(col("v"), "values").gt(lit(1.0));
        let rows = scan(&dataset)
            .filter_expr(filter)
            .try_collect()
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[tokio::test]
    async fn promotes_to_a_common_type() {
        // As numpy: int32 and float32 promote to float64.
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("table.lance").to_str().unwrap().to_string();
        let mut writer = NdWriter::create(&uri, &options(Casting::Promote))
            .await
            .unwrap();
        writer
            .write(one("v", Arc::new(Int32Array::from(vec![16_777_217]))))
            .await
            .unwrap();
        writer
            .write(one("v", Arc::new(Float32Array::from(vec![0.5f32]))))
            .await
            .unwrap();
        let dataset = writer.finish().await.unwrap();
        assert_eq!(value_type(&dataset, "v"), DataType::Float64);
        assert_eq!(
            v_values(&dataset).await,
            [Some(vec![16_777_217.0]), Some(vec![0.5])]
        );
    }

    #[tokio::test]
    async fn narrower_rows_do_not_widen() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("table.lance").to_str().unwrap().to_string();
        let mut writer = NdWriter::create(&uri, &options(Casting::Promote))
            .await
            .unwrap();
        writer
            .write(one("v", Arc::new(Float64Array::from(vec![0.1]))))
            .await
            .unwrap();
        writer
            .write(one("v", Arc::new(Float32Array::from(vec![1.5f32]))))
            .await
            .unwrap();
        assert!(
            writer
                .write(one(
                    "v",
                    Arc::new(arrow::array::StringArray::from(vec!["x"]))
                ))
                .await
                .is_err()
        );
        let dataset = writer.finish().await.unwrap();
        assert_eq!(dataset.version().version, 1);
        assert_eq!(value_type(&dataset, "v"), DataType::Float64);
    }
}
