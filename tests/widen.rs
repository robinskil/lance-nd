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
async fn widen_keeps_only_the_table_columns() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(&dir, true).await;
    let names = |dataset: &Dataset| -> Vec<String> {
        dataset
            .schema()
            .fields
            .iter()
            .map(|f| f.name.clone())
            .collect()
    };
    let before = names(&dataset);
    widen(&mut dataset, "v", &DataType::Float64).await.unwrap();
    assert_eq!(names(&dataset), before);
    // The old data file of v and its stats held no other field, so it is gone.
    let latest = Dataset::open(dataset.uri()).await.unwrap();
    let ids: std::collections::HashSet<i32> = latest.schema().field_ids().into_iter().collect();
    for fragment in latest.get_fragments() {
        for file in &fragment.metadata().files {
            assert!(file.fields.iter().any(|id| ids.contains(id)));
        }
    }
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

    /// One row with the 1-D variables `v` and `w`.
    fn pair(v: ArrayRef, w: ArrayRef) -> NdRecordBatch {
        let fields = vec![
            Field::new("v", v.data_type().clone(), true),
            Field::new("w", w.data_type().clone(), true),
        ];
        let arrays = vec![
            NdArrowArray::try_new(
                v.clone(),
                Dimensions::try_new([("v_dim", v.len())]).unwrap(),
            )
            .unwrap(),
            NdArrowArray::try_new(
                w.clone(),
                Dimensions::try_new([("w_dim", w.len())]).unwrap(),
            )
            .unwrap(),
        ];
        NdRecordBatch::try_new_inferred(Arc::new(Schema::new(fields)), arrays).unwrap()
    }

    #[tokio::test]
    async fn widens_all_variables_of_a_row_in_one_commit() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("table.lance").to_str().unwrap().to_string();
        let mut writer = NdWriter::create(&uri, &options(Casting::Promote))
            .await
            .unwrap();
        let narrow = pair(
            Arc::new(Float32Array::from(vec![1.5f32])),
            Arc::new(Float32Array::from(vec![2.5f32])),
        );
        let wide = pair(
            Arc::new(Float64Array::from(vec![0.1])),
            Arc::new(Float64Array::from(vec![0.2])),
        );
        writer.write(stream::iter([Ok(narrow)])).await.unwrap();
        writer.write(stream::iter([Ok(wide)])).await.unwrap();
        let dataset = writer.finish().await.unwrap();
        // Version 1 has the first row, versions 2 to 4 widen v and w, and
        // version 5 has the second row.
        assert_eq!(dataset.version().version, 5);
        for variable in ["v", "w"] {
            assert_eq!(value_type(&dataset, variable), DataType::Float64);
            assert_eq!(stats_type(&dataset, variable), DataType::Float64);
        }
        assert_eq!(v_values(&dataset).await, [Some(vec![1.5]), Some(vec![0.1])]);
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

    mod deferred {
        use super::*;

        fn deferred() -> WriteOptions {
            WriteOptions {
                defer_promotion: true,
                ..options(Casting::Promote)
            }
        }

        /// An append writer that wrote a Float32 row and a Float64 row of `v`.
        async fn promoted(dataset: &Dataset) -> NdWriter {
            let mut writer = NdWriter::append(dataset.uri(), &deferred()).await.unwrap();
            writer
                .write(one("v", Arc::new(Float32Array::from(vec![4.5f32]))))
                .await
                .unwrap();
            writer
                .write(one("v", Arc::new(Float64Array::from(vec![0.1]))))
                .await
                .unwrap();
            writer
        }

        #[tokio::test]
        async fn finish_makes_one_commit() {
            let dir = tempfile::tempdir().unwrap();
            let dataset = table(&dir, true).await;
            let mut writer = promoted(&dataset).await;
            writer
                .write(one("x", Arc::new(Int32Array::from(vec![5]))))
                .await
                .unwrap();

            // Readers do not see the widen or the rows before finish.
            let before = Dataset::open(dataset.uri()).await.unwrap();
            assert_eq!(before.version().version, 1);
            assert_eq!(value_type(&before, "v"), DataType::Float32);

            let dataset = writer.finish().await.unwrap();
            assert_eq!(dataset.version().version, 2);
            let latest = Dataset::open(dataset.uri()).await.unwrap();
            assert_eq!(latest.version().version, 2);
            assert_eq!(value_type(&latest, "v"), DataType::Float64);
            assert_eq!(stats_type(&latest, "v"), DataType::Float64);
            assert_eq!(value_type(&latest, "x"), DataType::Int32);
            assert_eq!(
                v_values(&latest).await,
                [
                    Some(vec![1.5, 2.5]),
                    None,
                    Some(vec![7.0]),
                    Some(vec![4.5]),
                    Some(vec![0.1]),
                    None
                ]
            );
        }

        #[tokio::test]
        async fn abort_leaves_the_table_unchanged() {
            let dir = tempfile::tempdir().unwrap();
            let dataset = table(&dir, true).await;
            promoted(&dataset).await.abort();
            let latest = Dataset::open(dataset.uri()).await.unwrap();
            assert_eq!(latest.version().version, 1);
            assert_eq!(value_type(&latest, "v"), DataType::Float32);
            assert_eq!(scan(&latest).try_collect().await.unwrap().len(), 3);
        }

        #[tokio::test]
        async fn another_commit_makes_finish_fail() {
            let dir = tempfile::tempdir().unwrap();
            let dataset = table(&dir, true).await;
            let writer = promoted(&dataset).await;
            let mut other = NdWriter::append(dataset.uri(), &WriteOptions::default())
                .await
                .unwrap();
            other
                .write(one("v", Arc::new(Float32Array::from(vec![9.0f32]))))
                .await
                .unwrap();
            other.finish().await.unwrap();

            let result = writer.finish().await;
            assert!(
                matches!(
                    result,
                    Err(Error::CommitConflict {
                        retryable: false,
                        ..
                    })
                ),
                "{result:?}"
            );
            let latest = Dataset::open(dataset.uri()).await.unwrap();
            assert_eq!(latest.version().version, 2);
            assert_eq!(value_type(&latest, "v"), DataType::Float32);
            assert_eq!(scan(&latest).try_collect().await.unwrap().len(), 4);
        }

        #[tokio::test]
        async fn overwrite_makes_one_commit() {
            let dir = tempfile::tempdir().unwrap();
            let dataset = table(&dir, true).await;
            let overwrite = WriteOptions {
                mode: WriteMode::Overwrite,
                ..deferred()
            };
            let mut writer = NdWriter::create(dataset.uri(), &overwrite).await.unwrap();
            writer
                .write(one("v", Arc::new(Float32Array::from(vec![1.5f32]))))
                .await
                .unwrap();
            writer
                .write(one("v", Arc::new(Float64Array::from(vec![0.1]))))
                .await
                .unwrap();
            let dataset = writer.finish().await.unwrap();
            let latest = Dataset::open(dataset.uri()).await.unwrap();
            assert_eq!(latest.version().version, 2);
            assert_eq!(value_type(&latest, "v"), DataType::Float64);
            assert_eq!(v_values(&latest).await, [Some(vec![1.5]), Some(vec![0.1])]);
        }

        #[tokio::test]
        async fn a_new_table_widens_as_without_the_flag() {
            let dir = tempfile::tempdir().unwrap();
            let uri = dir.path().join("table.lance").to_str().unwrap().to_string();
            let mut writer = NdWriter::create(&uri, &deferred()).await.unwrap();
            writer
                .write(one("v", Arc::new(Float32Array::from(vec![1.5f32]))))
                .await
                .unwrap();
            writer
                .write(one("v", Arc::new(Float64Array::from(vec![0.1]))))
                .await
                .unwrap();
            let dataset = writer.finish().await.unwrap();
            assert_eq!(dataset.version().version, 5);
            assert_eq!(v_values(&dataset).await, [Some(vec![1.5]), Some(vec![0.1])]);
        }
    }
}

/// The values of `v` as f64, for a Float32 or a Float64 table.
async fn v_any(dataset: &Dataset) -> Vec<Option<Vec<f64>>> {
    let rows = scan(dataset).try_collect().await.unwrap();
    rows.iter()
        .map(|row| {
            row.column_by_name("v").map(|v| {
                let values = arrow::compute::cast(v.values(), &DataType::Float64).unwrap();
                values.as_primitive::<Float64Type>().values().to_vec()
            })
        })
        .collect()
}

/// A widen relies on this Lance rule: a Project without
/// `preserves_nullability` conflicts with an append that it did not see.
#[tokio::test]
async fn project_conflicts_with_an_unseen_append() {
    use lance::dataset::transaction::{Operation, Transaction};
    use lance::dataset::{CommitBuilder, NewColumnTransform};

    for preserves_nullability in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let mut dataset = table(&dir, false).await;
        let extra = Schema::new(vec![Field::new("extra", DataType::Int32, true)]);
        dataset
            .add_columns(NewColumnTransform::AllNulls(Arc::new(extra)), None, None)
            .await
            .unwrap();
        let stale = dataset.clone();
        let rows = vec![row("v", Arc::new(Float32Array::from(vec![9.5f32])))];
        lance_nd::append(
            &mut dataset,
            stream::iter(rows.into_iter().map(Ok)),
            &WriteOptions::default(),
        )
        .await
        .unwrap();
        let mut schema = stale.schema().clone();
        schema.fields.retain(|field| field.name != "extra");
        let operation = Operation::Project {
            schema,
            preserves_nullability,
        };
        let version = stale.version().version;
        let result = CommitBuilder::new(Arc::new(stale))
            .execute(Transaction::new(version, operation, None))
            .await;
        if preserves_nullability {
            assert!(result.is_ok(), "{:?}", result.map(|_| ()));
        } else {
            assert!(
                matches!(result, Err(lance::Error::RetryableCommitConflict { .. })),
                "{:?}",
                result.map(|_| ())
            );
        }
    }
}

/// Appends run while a widen runs. Each committed value reads back once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn widen_with_concurrent_appends_loses_no_values() {
    use std::sync::atomic::{AtomicBool, Ordering};

    for round in 0..10 {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("table.lance").to_str().unwrap().to_string();
        let rows: Vec<_> = (0..4)
            .map(|i| Ok(row("v", Arc::new(Float32Array::from(vec![i as f32])))))
            .collect();
        let mut dataset = lance_nd::write(&uri, stream::iter(rows), &WriteOptions::default())
            .await
            .unwrap();
        let done = Arc::new(AtomicBool::new(false));
        let appenders: Vec<_> = (1..=3)
            .map(|id| {
                let (uri, done) = (uri.clone(), done.clone());
                tokio::spawn(async move {
                    let mut committed = Vec::new();
                    let mut i = 0;
                    // A few appends after the widen too. The cap stops the
                    // appends from preempting each widen attempt.
                    let mut after = 0;
                    while after < 3 && i < 40 {
                        if done.load(Ordering::SeqCst) {
                            after += 1;
                        }
                        i += 1;
                        let value = (id * 100_000 + i) as f32;
                        let mut writer = NdWriter::append(&uri, &WriteOptions::default())
                            .await
                            .unwrap();
                        let rows = [Ok(row("v", Arc::new(Float32Array::from(vec![value]))))];
                        match writer.write(stream::iter(rows)).await {
                            Ok(()) => {}
                            Err(Error::SchemaMismatch(_)) => continue,
                            Err(error) => panic!("write: {error:?}"),
                        }
                        match writer.finish().await {
                            Ok(_) => committed.push(f64::from(value)),
                            Err(Error::SchemaMismatch(_) | Error::CommitConflict { .. }) => {}
                            Err(error) => panic!("finish: {error:?}"),
                        }
                    }
                    committed
                })
            })
            .collect();
        let widened = widen(&mut dataset, "v", &DataType::Float64).await;
        done.store(true, Ordering::SeqCst);
        let mut committed: Vec<f64> = (0..4).map(f64::from).collect();
        for appender in appenders {
            committed.extend(appender.await.unwrap());
        }
        let latest = Dataset::open(&uri).await.unwrap();
        match widened {
            Ok(()) => assert_eq!(value_type(&latest, "v"), DataType::Float64),
            Err(Error::CommitConflict { .. }) => {}
            Err(error) => panic!("round {round}: widen: {error:?}"),
        }
        let mut read: Vec<f64> = v_any(&latest)
            .await
            .into_iter()
            .map(|values| values.expect("each row has v")[0])
            .collect();
        read.sort_by(f64::total_cmp);
        committed.sort_by(f64::total_cmp);
        assert_eq!(read, committed, "round {round}");
        assert!(
            latest
                .schema()
                .fields
                .iter()
                .all(|field| !field.name.starts_with("__stats__widen__")),
            "round {round}"
        );
    }
}

/// A widen that stopped between its commits leaves a temporary column.
#[tokio::test]
async fn widen_drops_a_leftover_temporary_column() {
    use lance::dataset::NewColumnTransform;

    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(&dir, true).await;
    let leftover = Schema::new(vec![Field::new("__stats__widen__v", DataType::Int32, true)]);
    dataset
        .add_columns(NewColumnTransform::AllNulls(Arc::new(leftover)), None, None)
        .await
        .unwrap();
    assert_eq!(v_any(&dataset).await.len(), 3);
    widen(&mut dataset, "v", &DataType::Float64).await.unwrap();
    assert!(dataset.schema().field("__stats__widen__v").is_none());
    assert_eq!(
        v_values(&dataset).await,
        [Some(vec![1.5, 2.5]), None, Some(vec![7.0])]
    );
}

/// A table whose rows are all deleted has no fragments.
#[tokio::test]
async fn widen_a_table_without_fragments() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(&dir, true).await;
    lance_nd::delete(&mut dataset, "true").await.unwrap();
    assert!(dataset.get_fragments().is_empty());
    widen(&mut dataset, "v", &DataType::Float64).await.unwrap();
    assert_eq!(value_type(&dataset, "v"), DataType::Float64);
    assert_eq!(stats_type(&dataset, "v"), DataType::Float64);
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
    assert_eq!(v_values(&dataset).await, [Some(vec![0.1])]);
}

/// The min and max of `v` per row, from its stats column, as f64.
async fn v_bounds(dataset: &Dataset) -> Vec<Option<(f64, f64)>> {
    use arrow::array::Array;

    let batch = dataset
        .scan()
        .project(&["__stats__:v"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let stats = batch.column(0).as_struct();
    let bound = |name: &str| {
        let column = stats.column_by_name(name).unwrap();
        assert_eq!(column.data_type(), &DataType::Float64);
        column.as_primitive::<Float64Type>().clone()
    };
    let (min, max) = (bound("min"), bound("max"));
    (0..batch.num_rows())
        .map(|i| min.is_valid(i).then(|| (min.value(i), max.value(i))))
        .collect()
}

/// The rows of `dataset` that a stats filter `v > limit` keeps.
async fn rows_above(dataset: &Dataset, limit: f64) -> Vec<Option<Vec<f64>>> {
    let filter = get_field(col("v"), "values").gt(lit(limit));
    let rows = scan(dataset)
        .filter_expr(filter)
        .try_collect()
        .await
        .unwrap();
    rows.iter()
        .map(|row| {
            row.column_by_name("v")
                .map(|v| v.values().as_primitive::<Float64Type>().values().to_vec())
        })
        .collect()
}

#[tokio::test]
async fn stats_keep_their_values_after_a_widen() {
    let dir = tempfile::tempdir().unwrap();
    let mut dataset = table(&dir, true).await;
    widen(&mut dataset, "v", &DataType::Float64).await.unwrap();
    assert_eq!(
        v_bounds(&dataset).await,
        [Some((1.5, 2.5)), None, Some((7.0, 7.0))]
    );
    assert_eq!(rows_above(&dataset, 5.0).await, [Some(vec![7.0])]);
    assert_eq!(rows_above(&dataset, 2.0).await.len(), 2);

    // Rows after the widen get Float64 stats.
    let mut writer = NdWriter::append(dataset.uri(), &WriteOptions::default())
        .await
        .unwrap();
    writer
        .write(stream::iter([Ok(row(
            "v",
            Arc::new(Float64Array::from(vec![0.1, 12.25])),
        ))]))
        .await
        .unwrap();
    let dataset = writer.finish().await.unwrap();
    assert_eq!(v_bounds(&dataset).await[3], Some((0.1, 12.25)));
    assert_eq!(rows_above(&dataset, 10.0).await, [Some(vec![0.1, 12.25])]);
}

#[tokio::test]
async fn stats_keep_their_values_after_a_deferred_widen() {
    use lance_nd::Casting;

    let dir = tempfile::tempdir().unwrap();
    let dataset = table(&dir, true).await;
    let options = WriteOptions {
        casting: Casting::Promote,
        defer_promotion: true,
        ..Default::default()
    };
    let mut writer = NdWriter::append(dataset.uri(), &options).await.unwrap();
    for values in [
        Arc::new(Float32Array::from(vec![4.5f32])) as ArrayRef,
        Arc::new(Float64Array::from(vec![0.1, 20.5])),
    ] {
        writer
            .write(stream::iter([Ok(row("v", values))]))
            .await
            .unwrap();
    }
    let dataset = writer.finish().await.unwrap();
    let latest = Dataset::open(dataset.uri()).await.unwrap();
    assert_eq!(latest.version().version, 2);
    assert_eq!(
        v_bounds(&latest).await,
        [
            Some((1.5, 2.5)),
            None,
            Some((7.0, 7.0)),
            Some((4.5, 4.5)),
            Some((0.1, 20.5))
        ]
    );
    assert_eq!(
        rows_above(&latest, 5.0).await,
        [Some(vec![7.0]), Some(vec![0.1, 20.5])]
    );
}
