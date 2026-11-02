use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Float64Array, Int32Array, RecordBatch, StringArray,
    StructArray, TimestampNanosecondArray,
};
use arrow::datatypes::{DataType, Field, Float64Type, Int32Type, Schema, TimeUnit};
use lance_nd::encode::encode_rows;
use lance_nd::stats::{STATS_FIELD, add_stats, has_stats};
use nd_arrow_array::{Dimensions, NdArrowArray, NdRecordBatch};

/// One row with the given 1-D variables, each on its own dimension.
fn row(variables: Vec<(&str, ArrayRef)>) -> NdRecordBatch {
    let fields: Vec<Field> = variables
        .iter()
        .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
        .collect();
    let arrays = variables
        .into_iter()
        .map(|(name, array)| {
            let dims = Dimensions::try_new([(format!("{name}_dim"), array.len())]).unwrap();
            NdArrowArray::try_new(array, dims).unwrap()
        })
        .collect();
    NdRecordBatch::try_new_inferred(Arc::new(Schema::new(fields)), arrays).unwrap()
}

fn stats_of(rows: &[NdRecordBatch]) -> RecordBatch {
    add_stats(&encode_rows(rows).unwrap()).unwrap()
}

fn child<'a>(batch: &'a RecordBatch, variable: &str) -> &'a StructArray {
    batch
        .column_by_name(STATS_FIELD)
        .expect("a __stats__ column")
        .as_struct()
        .column_by_name(variable)
        .expect("a stats child")
        .as_struct()
}

fn min_max(child: &StructArray) -> (&ArrayRef, &ArrayRef) {
    (
        child.column_by_name("min").unwrap(),
        child.column_by_name("max").unwrap(),
    )
}

#[test]
fn min_max_per_row() {
    let batch = stats_of(&[
        row(vec![
            (
                "lat",
                Arc::new(Float64Array::from(vec![-90.0, -89.0, -88.0])),
            ),
            (
                "sst",
                Arc::new(Float64Array::from(vec![Some(5.0), None, Some(1.0)])),
            ),
        ]),
        row(vec![
            ("lat", Arc::new(Float64Array::from(vec![11.0, 10.0]))),
            ("sst", Arc::new(Float64Array::from(vec![None, Some(7.0)]))),
        ]),
    ]);
    let (min, max) = min_max(child(&batch, "lat"));
    assert_eq!(min.as_primitive::<Float64Type>().values(), &[-90.0, 10.0]);
    assert_eq!(max.as_primitive::<Float64Type>().values(), &[-88.0, 11.0]);
    let (min, max) = min_max(child(&batch, "sst"));
    assert_eq!(min.as_primitive::<Float64Type>().values(), &[1.0, 7.0]);
    assert_eq!(max.as_primitive::<Float64Type>().values(), &[5.0, 7.0]);
    assert!(has_stats(batch.schema().as_ref()));
}

#[test]
fn nan_and_all_null() {
    let batch = stats_of(&[
        row(vec![(
            "v",
            Arc::new(Float64Array::from(vec![f64::NAN, 2.0, f64::NAN])),
        )]),
        row(vec![(
            "v",
            Arc::new(Float64Array::from(vec![Some(f64::NAN), None])),
        )]),
    ]);
    let stats = child(&batch, "v");
    assert!(stats.is_valid(0) && stats.is_valid(1));
    let (min, max) = min_max(stats);
    assert_eq!(min.as_primitive::<Float64Type>().value(0), 2.0);
    assert_eq!(max.as_primitive::<Float64Type>().value(0), 2.0);
    assert!(min.is_null(1) && max.is_null(1));
}

#[test]
fn absent_variable_is_null_child() {
    let batch = stats_of(&[
        row(vec![(
            "pressure",
            Arc::new(Int32Array::from(vec![3, 1, 2])),
        )]),
        row(vec![("other", Arc::new(Int32Array::from(vec![0])))]),
    ]);
    let stats = child(&batch, "pressure");
    assert!(stats.is_valid(0) && stats.is_null(1));
    let (min, max) = min_max(stats);
    assert_eq!(min.as_primitive::<Int32Type>().value(0), 1);
    assert_eq!(max.as_primitive::<Int32Type>().value(0), 3);
}

#[test]
fn strings_and_timestamps() {
    let times = TimestampNanosecondArray::from(vec![30, 10, 20]).with_timezone("UTC");
    let batch = stats_of(&[row(vec![
        ("name", Arc::new(StringArray::from(vec!["b", "a", "c"]))),
        ("time", Arc::new(times)),
    ])]);
    let (min, max) = min_max(child(&batch, "name"));
    assert_eq!(min.as_string::<i32>().value(0), "a");
    assert_eq!(max.as_string::<i32>().value(0), "c");
    let (min, max) = min_max(child(&batch, "time"));
    assert_eq!(
        min.data_type(),
        &DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into()))
    );
    assert_eq!(
        min.as_primitive::<arrow::datatypes::TimestampNanosecondType>()
            .value(0),
        10
    );
    assert_eq!(
        max.as_primitive::<arrow::datatypes::TimestampNanosecondType>()
            .value(0),
        30
    );
}

#[test]
fn other_types_get_no_child() {
    let batch = stats_of(&[row(vec![
        ("flag", Arc::new(BooleanArray::from(vec![true, false]))),
        ("v", Arc::new(Int32Array::from(vec![1]))),
    ])]);
    let stats = batch.column_by_name(STATS_FIELD).unwrap().as_struct();
    assert!(stats.column_by_name("flag").is_none());
    assert!(stats.column_by_name("v").is_some());

    let only_flags = stats_of(&[row(vec![(
        "flag",
        Arc::new(BooleanArray::from(vec![true])),
    )])]);
    assert!(only_flags.column_by_name(STATS_FIELD).is_none());
    assert!(has_stats(only_flags.schema().as_ref()));
}

mod tables {
    use super::*;
    use futures::{TryStreamExt, stream};
    use lance::Dataset;
    use lance_nd::{Error, NdWriter, WriteMode, WriteOptions, append, scan, variables, write};

    fn uri(dir: &tempfile::TempDir) -> String {
        dir.path().join("table.lance").to_str().unwrap().to_string()
    }

    fn lat_row(values: Vec<f64>) -> NdRecordBatch {
        row(vec![("lat", Arc::new(Float64Array::from(values)))])
    }

    fn rows(
        batches: Vec<NdRecordBatch>,
    ) -> impl futures::Stream<Item = lance_nd::Result<NdRecordBatch>> {
        stream::iter(batches.into_iter().map(Ok))
    }

    fn with_stats() -> WriteOptions {
        WriteOptions {
            stats: true,
            ..Default::default()
        }
    }

    async fn stats_column(dataset: &Dataset) -> RecordBatch {
        dataset
            .scan()
            .project(&[STATS_FIELD])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
    }

    fn table_has_stats(dataset: &Dataset) -> bool {
        has_stats(&Schema::from(dataset.schema()))
    }

    #[tokio::test]
    async fn writer_with_stats_stores_them() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = NdWriter::create(&uri(&dir), &with_stats()).await.unwrap();
        writer
            .write(rows(vec![lat_row(vec![3.0, 1.0]), lat_row(vec![-5.0])]))
            .await
            .unwrap();
        let dataset = writer.finish().await.unwrap();
        assert!(table_has_stats(&dataset));
        let batch = stats_column(&dataset).await;
        let (min, max) = min_max(child(&batch, "lat"));
        assert_eq!(min.as_primitive::<Float64Type>().values(), &[1.0, -5.0]);
        assert_eq!(max.as_primitive::<Float64Type>().values(), &[3.0, -5.0]);
    }

    #[tokio::test]
    async fn stats_off_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = NdWriter::create(&uri(&dir), &WriteOptions::default())
            .await
            .unwrap();
        writer.write(rows(vec![lat_row(vec![1.0])])).await.unwrap();
        let dataset = writer.finish().await.unwrap();
        assert!(!table_has_stats(&dataset));
        assert!(dataset.schema().field(STATS_FIELD).is_none());
    }

    #[tokio::test]
    async fn append_follows_the_table() {
        let dir = tempfile::tempdir().unwrap();
        let uri = uri(&dir);
        let mut writer = NdWriter::create(&uri, &with_stats()).await.unwrap();
        writer.write(rows(vec![lat_row(vec![1.0])])).await.unwrap();
        writer.finish().await.unwrap();
        let mut writer = NdWriter::append(&uri, &WriteOptions::default())
            .await
            .unwrap();
        writer
            .write(rows(vec![lat_row(vec![7.0, 8.0])]))
            .await
            .unwrap();
        let dataset = writer.finish().await.unwrap();
        let batch = stats_column(&dataset).await;
        let (min, _) = min_max(child(&batch, "lat"));
        assert_eq!(min.as_primitive::<Float64Type>().values(), &[1.0, 7.0]);

        let plain = dir.path().join("plain.lance").to_str().unwrap().to_string();
        let mut writer = NdWriter::create(&plain, &WriteOptions::default())
            .await
            .unwrap();
        writer.write(rows(vec![lat_row(vec![1.0])])).await.unwrap();
        writer.finish().await.unwrap();
        let mut writer = NdWriter::append(&plain, &with_stats()).await.unwrap();
        writer.write(rows(vec![lat_row(vec![2.0])])).await.unwrap();
        let dataset = writer.finish().await.unwrap();
        assert!(!table_has_stats(&dataset));
        assert!(dataset.schema().field(STATS_FIELD).is_none());
    }

    #[tokio::test]
    async fn write_fn_with_stats() {
        let dir = tempfile::tempdir().unwrap();
        let uri = uri(&dir);
        let options = WriteOptions {
            max_rows_per_commit: 1,
            ..with_stats()
        };
        let mut dataset = write(
            &uri,
            rows(vec![lat_row(vec![1.0]), lat_row(vec![2.0])]),
            &options,
        )
        .await
        .unwrap();
        append(
            &mut dataset,
            rows(vec![lat_row(vec![3.0])]),
            &WriteOptions::default(),
        )
        .await
        .unwrap();
        assert!(table_has_stats(&dataset));
        let batch = stats_column(&dataset).await;
        let (min, _) = min_max(child(&batch, "lat"));
        assert_eq!(min.as_primitive::<Float64Type>().values(), &[1.0, 2.0, 3.0]);

        let overwrite = WriteOptions {
            mode: WriteMode::Overwrite,
            ..Default::default()
        };
        let dataset = write(&uri, rows(vec![lat_row(vec![1.0])]), &overwrite)
            .await
            .unwrap();
        assert!(!table_has_stats(&dataset));
    }

    #[tokio::test]
    async fn new_variable_adds_a_child() {
        let dir = tempfile::tempdir().unwrap();
        let uri = uri(&dir);
        let mut writer = NdWriter::create(&uri, &with_stats()).await.unwrap();
        writer.write(rows(vec![lat_row(vec![1.0])])).await.unwrap();
        writer.finish().await.unwrap();
        let mut writer = NdWriter::append(&uri, &WriteOptions::default())
            .await
            .unwrap();
        let pressure = row(vec![("pressure", Arc::new(Int32Array::from(vec![4, 2])))]);
        writer.write(rows(vec![pressure])).await.unwrap();
        let dataset = writer.finish().await.unwrap();
        let batch = stats_column(&dataset).await;
        let stats = child(&batch, "pressure");
        assert!(stats.is_null(0) && stats.is_valid(1));
        assert_eq!(min_max(stats).0.as_primitive::<Int32Type>().value(1), 2);
        assert!(child(&batch, "lat").is_null(1));
    }

    #[tokio::test]
    async fn scan_hides_stats() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = NdWriter::create(&uri(&dir), &with_stats()).await.unwrap();
        writer
            .write(rows(vec![lat_row(vec![1.0, 2.0])]))
            .await
            .unwrap();
        let dataset = writer.finish().await.unwrap();
        let read: Vec<NdRecordBatch> = scan(&dataset)
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(read.len(), 1);
        let names: Vec<_> = read[0]
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, ["lat"]);
        let names: Vec<_> = variables(&dataset)
            .unwrap()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, ["lat"]);
        let error = scan(&dataset)
            .project(&[STATS_FIELD])
            .try_into_stream()
            .await
            .err();
        assert!(matches!(error, Some(Error::InvalidArgument(_))));
        let read: Vec<NdRecordBatch> = scan(&dataset)
            .project(&["lat"])
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(read[0].schema().fields().len(), 1);
    }

    #[test]
    fn stats_name_is_reserved() {
        let bad = row(vec![(STATS_FIELD, Arc::new(Int32Array::from(vec![1])))]);
        assert!(matches!(encode_rows(&[bad]), Err(Error::SchemaMismatch(_))));
    }
}

mod pruning {
    use super::*;
    use arrow::array::{Int64Array, TimestampNanosecondArray as Times};
    use futures::{TryStreamExt, stream};
    use lance::Dataset;
    use lance_nd::datafusion::common::ScalarValue;
    use lance_nd::datafusion::logical_expr::{Expr, col, lit};
    use lance_nd::{Error, NdScanner, NdWriter, WriteOptions, scan};

    async fn table(dir: &tempfile::TempDir, rows: Vec<NdRecordBatch>, stats: bool) -> Dataset {
        let uri = dir.path().join("table.lance").to_str().unwrap().to_string();
        let options = WriteOptions {
            stats,
            ..Default::default()
        };
        let mut writer = NdWriter::create(&uri, &options).await.unwrap();
        writer
            .write(stream::iter(rows.into_iter().map(Ok)))
            .await
            .unwrap();
        writer.finish().await.unwrap()
    }

    fn time(ns: i64) -> Expr {
        lit(ScalarValue::TimestampNanosecond(Some(ns), None))
    }

    /// Rows 0 to 4: `lat` ranges [-90,-80], [0,10], [17,30], absent, all NaN.
    fn ocean() -> Vec<NdRecordBatch> {
        let lats: [Option<Vec<f64>>; 5] = [
            Some(vec![-90.0, -80.0]),
            Some(vec![0.0, 10.0]),
            Some(vec![17.0, 30.0]),
            None,
            Some(vec![f64::NAN, f64::NAN]),
        ];
        let names = [vec!["a"], vec!["b", "c"], vec!["b"], vec!["z"], vec!["q"]];
        lats.into_iter()
            .zip(names)
            .enumerate()
            .map(|(id, (lat, name))| {
                let mut variables: Vec<(&str, ArrayRef)> = vec![
                    ("id", Arc::new(Int64Array::from(vec![id as i64]))),
                    ("name", Arc::new(StringArray::from(name))),
                    ("time", Arc::new(Times::from(vec![100 * (id as i64 + 1)]))),
                ];
                if let Some(lat) = lat {
                    variables.push(("lat", Arc::new(Float64Array::from(lat))));
                }
                row(variables)
            })
            .collect()
    }

    async fn ids(scanner: NdScanner<'_>) -> Vec<i64> {
        let rows: Vec<NdRecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        rows.iter()
            .map(|row| {
                let id = row.column_by_name("id").expect("an id");
                id.values()
                    .as_primitive::<arrow::datatypes::Int64Type>()
                    .value(0)
            })
            .collect()
    }

    #[tokio::test]
    async fn between_prunes() {
        let dir = tempfile::tempdir().unwrap();
        let dataset = table(&dir, ocean(), true).await;
        let filter = col("lat").between(lit(17.5), lit(25.4));
        assert_eq!(ids(scan(&dataset).filter_expr(filter)).await, [2, 4]);
    }

    #[tokio::test]
    async fn and_or_across_variables() {
        let dir = tempfile::tempdir().unwrap();
        let dataset = table(&dir, ocean(), true).await;
        let filter = col("lat")
            .gt(lit(5.0))
            .and(col("name").eq(lit("b")))
            .or(col("time").lt(time(150)));
        assert_eq!(ids(scan(&dataset).filter_expr(filter)).await, [0, 1, 2]);
    }

    #[tokio::test]
    async fn absent_variable_is_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let dataset = table(&dir, ocean(), true).await;
        let filter = col("lat").gt_eq(lit(-100.0));
        assert_eq!(ids(scan(&dataset).filter_expr(filter)).await, [0, 1, 2, 4]);
    }

    #[tokio::test]
    async fn all_nan_row_is_kept() {
        let dir = tempfile::tempdir().unwrap();
        let dataset = table(&dir, ocean(), true).await;
        let filter = col("lat").gt(lit(1000.0));
        assert_eq!(ids(scan(&dataset).filter_expr(filter)).await, [4]);
    }

    #[tokio::test]
    async fn int_literal_on_float() {
        let dir = tempfile::tempdir().unwrap();
        let dataset = table(&dir, ocean(), true).await;
        let filter = col("lat").gt(lit(17i32));
        assert_eq!(ids(scan(&dataset).filter_expr(filter)).await, [2, 4]);
        let filter = col("id").lt(lit(1.5f64));
        assert_eq!(ids(scan(&dataset).filter_expr(filter)).await, [0, 1]);
    }

    #[tokio::test]
    async fn table_without_stats_keeps_all() {
        let dir = tempfile::tempdir().unwrap();
        let dataset = table(&dir, ocean(), false).await;
        let filter = col("lat").gt(lit(1000.0));
        assert_eq!(
            ids(scan(&dataset).filter_expr(filter)).await,
            [0, 1, 2, 3, 4]
        );
    }

    #[tokio::test]
    async fn filter_and_filter_expr_conflict() {
        let dir = tempfile::tempdir().unwrap();
        let dataset = table(&dir, ocean(), true).await;
        let scanner = scan(&dataset)
            .filter("id > 1")
            .filter_expr(col("lat").gt(lit(1.0)));
        assert!(matches!(
            scanner.try_into_stream().await.err(),
            Some(Error::InvalidArgument(_))
        ));
    }

    #[tokio::test]
    async fn project_with_filter_expr() {
        let dir = tempfile::tempdir().unwrap();
        let dataset = table(&dir, ocean(), true).await;
        let rows: Vec<NdRecordBatch> = scan(&dataset)
            .project(&["id"])
            .filter_expr(col("lat").between(lit(17.5), lit(25.4)))
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(rows.len(), 2);
        for row in rows {
            let names: Vec<_> = row
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
            assert_eq!(names, ["id"]);
        }
    }

    /// A small xorshift generator with a fixed seed.
    struct Random(u64);

    impl Random {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// The values of one generated row. `None` is an absent variable.
    struct Data {
        x: Option<Vec<Option<f64>>>,
        k: Option<Vec<Option<i32>>>,
        s: Option<Vec<Option<String>>>,
    }

    fn values<T: Clone>(
        random: &mut Random,
        value: impl Fn(&mut Random) -> T,
    ) -> Option<Vec<Option<T>>> {
        match random.below(6) {
            0 => None,
            1 => Some(vec![None; 1 + random.below(3) as usize]),
            _ => {
                let len = 1 + random.below(5);
                Some(
                    (0..len)
                        .map(|_| (random.below(5) != 0).then(|| value(random)))
                        .collect(),
                )
            }
        }
    }

    fn letter(random: &mut Random, n: u64) -> String {
        char::from(b'a' + random.below(n) as u8).to_string()
    }

    fn generate(random: &mut Random) -> Data {
        Data {
            x: values(random, |r| {
                if r.below(10) == 0 {
                    f64::NAN
                } else {
                    r.below(100) as f64 - 50.0
                }
            }),
            k: values(random, |r| r.below(40) as i32 - 20),
            s: values(random, |r| letter(r, 6)),
        }
    }

    fn encode(id: usize, data: &Data) -> NdRecordBatch {
        let mut variables: Vec<(&str, ArrayRef)> =
            vec![("id", Arc::new(Int64Array::from(vec![id as i64])))];
        if let Some(x) = &data.x {
            variables.push(("x", Arc::new(Float64Array::from(x.clone()))));
        }
        if let Some(k) = &data.k {
            variables.push(("k", Arc::new(Int32Array::from(k.clone()))));
        }
        if let Some(s) = &data.s {
            variables.push(("s", Arc::new(StringArray::from(s.clone()))));
        }
        row(variables)
    }

    /// One generated filter: the `Expr` and its exact result for one row.
    struct Filter {
        expr: Expr,
        matches: Box<dyn Fn(&Data) -> bool>,
    }

    /// A filter on one variable. A row matches when one of its values matches.
    fn leaf<T: Clone + PartialOrd + 'static>(
        random: &mut Random,
        name: &'static str,
        get: fn(&Data) -> Option<&Vec<Option<T>>>,
        literal: fn(&T) -> Expr,
        pick: fn(&mut Random) -> T,
    ) -> Filter {
        let a = pick(random);
        let b = pick(random);
        let column = col(name);
        let (expr, test): (Expr, Box<dyn Fn(&T) -> bool>) = match random.below(7) {
            0 => (column.eq(literal(&a)), Box::new(move |v| *v == a)),
            1 => (column.lt(literal(&a)), Box::new(move |v| *v < a)),
            2 => (column.lt_eq(literal(&a)), Box::new(move |v| *v <= a)),
            3 => (column.gt(literal(&a)), Box::new(move |v| *v > a)),
            4 => (column.gt_eq(literal(&a)), Box::new(move |v| *v >= a)),
            5 => (
                column.between(literal(&a), literal(&b)),
                Box::new(move |v| *v >= a && *v <= b),
            ),
            _ => (
                column.in_list(vec![literal(&a), literal(&b)], false),
                Box::new(move |v| *v == a || *v == b),
            ),
        };
        let matches = Box::new(move |data: &Data| {
            get(data).is_some_and(|values| values.iter().flatten().any(|v| test(v)))
        });
        Filter { expr, matches }
    }

    fn filter(random: &mut Random, depth: u32) -> Filter {
        if depth == 0 && random.below(5) < 2 {
            let (l, r) = (filter(random, 1), filter(random, 1));
            let and = random.below(2) == 0;
            let expr = if and {
                l.expr.and(r.expr)
            } else {
                l.expr.or(r.expr)
            };
            let (left, right) = (l.matches, r.matches);
            let matches: Box<dyn Fn(&Data) -> bool> = if and {
                Box::new(move |d| left(d) && right(d))
            } else {
                Box::new(move |d| left(d) || right(d))
            };
            return Filter { expr, matches };
        }
        match random.below(3) {
            0 => leaf(
                random,
                "x",
                |d| d.x.as_ref(),
                |v| lit(*v),
                |r| r.below(110) as f64 - 55.0,
            ),
            1 => leaf(
                random,
                "k",
                |d| d.k.as_ref(),
                |v| lit(*v),
                |r| r.below(44) as i32 - 22,
            ),
            _ => leaf(
                random,
                "s",
                |d| d.s.as_ref(),
                |v| lit(v.clone()),
                |r| letter(r, 7),
            ),
        }
    }

    #[tokio::test]
    async fn never_skips_a_match() {
        let mut random = Random(0x9E37_79B9_7F4A_7C15);
        let data: Vec<Data> = (0..200).map(|_| generate(&mut random)).collect();
        let rows = data
            .iter()
            .enumerate()
            .map(|(id, d)| encode(id, d))
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let dataset = table(&dir, rows, true).await;
        let mut pruned = 0;
        for _ in 0..300 {
            let filter = filter(&mut random, 0);
            let kept = ids(scan(&dataset).filter_expr(filter.expr.clone())).await;
            for (id, d) in data.iter().enumerate() {
                if (filter.matches)(d) {
                    assert!(
                        kept.contains(&(id as i64)),
                        "{} skips row {id}",
                        filter.expr
                    );
                }
            }
            pruned += data.len() - kept.len();
        }
        assert!(pruned > 0, "no filter pruned a row");
    }
}
