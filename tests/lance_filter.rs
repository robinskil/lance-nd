//! Filters that mix attributes and values: `filter_expr` and `lance_filter`.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, AsArray, Float32Array, Int64Array, TimestampNanosecondArray as Times,
};
use arrow::datatypes::{Field, Int64Type, Schema};
use futures::stream;
use lance::Dataset;
use lance_nd::datafusion::common::ScalarValue;
use lance_nd::datafusion::functions::core::expr_fn::{get_field, get_field_path};
use lance_nd::datafusion::logical_expr::{Expr, col, lit};
use lance_nd::prune::prune_filter;
use lance_nd::{Error, NdScanner, NdWriter, WriteOptions, scan};
use nd_arrow_array::{Attributes, Dimensions, NdArrowArray, NdRecordBatch};

fn variable(name: &str, array: ArrayRef) -> (Field, NdArrowArray) {
    let dims = Dimensions::try_new([(format!("{name}_dim"), array.len())]).unwrap();
    let field = Field::new(name, array.data_type().clone(), true);
    (field, NdArrowArray::try_new(array, dims).unwrap())
}

/// Rows 0 to 4 with the global attribute `station = s<id>`, `time` and
/// `temp` in [10 id, 10 id + 5]. Row 3 has no `temp`. The `temp` units are
/// `degF` in row 1 and `degC` in the other rows.
fn stations() -> Vec<NdRecordBatch> {
    (0..5i64)
        .map(|id| {
            let mut variables = vec![
                variable("id", Arc::new(Int64Array::from(vec![id]))),
                variable("time", Arc::new(Times::from(vec![100 * (id + 1)]))),
            ];
            if id != 3 {
                let low = 10.0 * id as f32;
                let (field, temp) =
                    variable("temp", Arc::new(Float32Array::from(vec![low, low + 5.0])));
                let units = if id == 1 { "degF" } else { "degC" };
                variables.push((
                    field,
                    temp.with_attributes(Attributes::new().with("units", units)),
                ));
            }
            let (fields, arrays): (Vec<_>, Vec<_>) = variables.into_iter().unzip();
            NdRecordBatch::try_new_inferred(Arc::new(Schema::new(fields)), arrays)
                .unwrap()
                .with_attributes(Attributes::new().with("station", format!("s{id}")))
        })
        .collect()
}

async fn table(dir: &tempfile::TempDir, stats: bool) -> Dataset {
    let uri = dir.path().join("table.lance").to_str().unwrap().to_string();
    let options = WriteOptions {
        stats,
        ..Default::default()
    };
    let mut writer = NdWriter::create(&uri, &options).await.unwrap();
    writer
        .write(stream::iter(stations().into_iter().map(Ok)))
        .await
        .unwrap();
    writer.finish().await.unwrap()
}

async fn ids(scanner: NdScanner<'_>) -> Vec<i64> {
    let rows: Vec<NdRecordBatch> = scanner.try_collect().await.unwrap();
    rows.iter()
        .map(|row| {
            let id = row.column_by_name("id").expect("an id");
            id.values().as_primitive::<Int64Type>().value(0)
        })
        .collect()
}

/// The values of `variable`: `<variable>.values`.
fn values(variable: &str) -> Expr {
    get_field(col(variable), "values")
}

fn station() -> Expr {
    get_field(col("__global_attrs__"), "station")
}

fn time(ns: i64) -> Expr {
    lit(ScalarValue::TimestampNanosecond(Some(ns), None))
}

#[tokio::test]
async fn lance_filter_passes_through() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = table(&dir, true).await;
    let filter = station().eq(lit("s3"));
    assert_eq!(ids(scan(&dataset).lance_filter(filter)).await, [3]);
}

#[tokio::test]
async fn mixed_or_in_filter_expr() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = table(&dir, true).await;
    // A Float64 literal on Float32 values.
    let filter = station().eq(lit("s3")).or(values("temp").gt(lit(35.0f64)));
    assert_eq!(ids(scan(&dataset).filter_expr(filter)).await, [3, 4]);
}

#[tokio::test]
async fn variable_attribute_in_filter_expr() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = table(&dir, true).await;
    let nested = get_field(get_field(col("temp"), "attrs"), "units");
    let path = get_field_path(col("temp"), vec![lit("attrs"), lit("units")]);
    for units in [nested, path] {
        let filter = units.eq(lit("degF"));
        assert_eq!(ids(scan(&dataset).filter_expr(filter)).await, [1]);
    }
}

#[tokio::test]
async fn stats_filter_with_time_literal_in_lance_filter() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = table(&dir, true).await;
    let table = Schema::from(dataset.schema());
    let values = values("time").lt(time(250));
    let stats = prune_filter(&values, &table)
        .unwrap()
        .expect("a stats filter");
    let filter = station().eq(lit("s3")).or(stats);
    assert_eq!(ids(scan(&dataset).lance_filter(filter)).await, [0, 1, 3]);
}

#[tokio::test]
async fn filter_expr_and_lance_filter_combine() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = table(&dir, true).await;
    let scanner = scan(&dataset)
        .filter_expr(values("temp").gt(lit(15.0)))
        .lance_filter(station().not_eq(lit("s2")));
    assert_eq!(ids(scanner).await, [4]);
}

#[tokio::test]
async fn not_over_attribute_is_kept() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = table(&dir, true).await;
    let filter = (!station().eq(lit("s0"))).and(values("temp").lt(lit(12.0)));
    assert_eq!(ids(scan(&dataset).filter_expr(filter)).await, [1]);
}

#[tokio::test]
async fn table_without_stats_keeps_attribute_part() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = table(&dir, false).await;
    let filter = station().eq(lit("s3")).and(values("temp").gt(lit(1000.0)));
    assert_eq!(ids(scan(&dataset).filter_expr(filter)).await, [3]);
}

#[tokio::test]
async fn filter_with_lance_filter_conflicts() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = table(&dir, true).await;
    let scanner = scan(&dataset)
        .filter("id > 1")
        .lance_filter(station().eq(lit("s3")));
    assert!(matches!(
        scanner.try_into_stream().await.err(),
        Some(Error::InvalidArgument(_))
    ));
}

#[tokio::test]
async fn wrong_name_fails_in_the_scan() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = table(&dir, true).await;
    let filter = get_field(col("nope"), "attrs").is_null();
    assert!(
        scan(&dataset)
            .filter_expr(filter)
            .try_collect()
            .await
            .is_err()
    );
}

/// The columns of the first Lance read, the read that applies the filter.
fn filter_read(plan: &str) -> String {
    let line = plan
        .lines()
        .find(|line| line.contains("full_filter="))
        .unwrap_or_else(|| panic!("no filter read in {plan}"));
    let start = line.find("projection=[").expect("a projection") + "projection=[".len();
    line[start..start + line[start..].find(']').unwrap()].to_string()
}

#[tokio::test]
async fn filter_reads_variables_late() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = table(&dir, true).await;
    let scans = [
        scan(&dataset)
            .project(&["temp"])
            .filter("__global_attrs__.station = 's3'"),
        scan(&dataset)
            .project(&["temp"])
            .filter_expr(values("temp").gt(lit(35.0))),
        scan(&dataset)
            .project(&["temp"])
            .lance_filter(station().eq(lit("s3"))),
    ];
    for scanner in scans {
        let plan = scanner.explain_plan(true).await.unwrap();
        let columns = filter_read(&plan);
        assert!(!columns.contains("temp"), "{columns}\n{plan}");
    }
    let rows = scan(&dataset)
        .project(&["temp"])
        .filter("__global_attrs__.station = 'nope'")
        .try_collect()
        .await
        .unwrap();
    assert!(rows.is_empty());
}

#[tokio::test]
async fn scan_without_filter_reads_once() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = table(&dir, true).await;
    let plan = scan(&dataset)
        .project(&["temp"])
        .explain_plan(true)
        .await
        .unwrap();
    assert_eq!(plan.matches("LanceRead").count(), 1, "{plan}");
    assert!(plan.contains("temp"), "{plan}");
}
