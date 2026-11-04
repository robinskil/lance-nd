//! `WriteOptions::casting`: new rows with another type than the table.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, AsArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray, UInt16Array,
};
use arrow::datatypes::{DataType, Field, Float32Type, Float64Type, Int32Type, Schema};
use futures::stream;
use lance::Dataset;
use lance_nd::{Casting, Error, NdWriter, WriteMode, WriteOptions, scan, write};
use nd_arrow_array::{Attributes, Dimensions, NdArrowArray, NdRecordBatch};

/// One row with the 1-D variable `v` and the attributes `attrs` on it.
fn row(values: ArrayRef, attrs: Attributes) -> NdRecordBatch {
    let dims = Dimensions::try_new([("n", values.len())]).unwrap();
    let field = Field::new("v", values.data_type().clone(), true);
    let array = NdArrowArray::try_new(values, dims)
        .unwrap()
        .with_attributes(attrs);
    NdRecordBatch::try_new_inferred(Arc::new(Schema::new(vec![field])), vec![array]).unwrap()
}

fn plain(values: ArrayRef) -> NdRecordBatch {
    row(values, Attributes::new())
}

fn options(casting: Casting) -> WriteOptions {
    WriteOptions {
        stats: true,
        casting,
        ..Default::default()
    }
}

/// A writer for a new table with the first row, then the other rows, each
/// in its own write. Returns the result of each later write and the table.
async fn write_rows(
    dir: &tempfile::TempDir,
    casting: Casting,
    first: NdRecordBatch,
    later: Vec<NdRecordBatch>,
) -> (Vec<lance_nd::Result<()>>, Dataset) {
    let uri = dir.path().join("table.lance").to_str().unwrap().to_string();
    let mut writer = NdWriter::create(&uri, &options(casting)).await.unwrap();
    writer.write(stream::iter([Ok(first)])).await.unwrap();
    let mut results = Vec::new();
    for row in later {
        results.push(writer.write(stream::iter([Ok(row)])).await);
    }
    (results, writer.finish().await.unwrap())
}

async fn values(dataset: &Dataset) -> Vec<ArrayRef> {
    let rows = scan(dataset).try_collect().await.unwrap();
    rows.iter()
        .map(|row| row.column_by_name("v").unwrap().values().clone())
        .collect()
}

fn stats_type(dataset: &Dataset) -> DataType {
    let schema = Schema::from(dataset.schema());
    let DataType::Struct(children) = schema.field_with_name("__stats__").unwrap().data_type()
    else {
        panic!("stats are a struct");
    };
    let (_, child) = children.find("v").unwrap();
    let DataType::Struct(bounds) = child.data_type() else {
        panic!("a stats child is a struct");
    };
    bounds.find("min").unwrap().1.data_type().clone()
}

#[tokio::test]
async fn no_casting_rejects_another_type() {
    let dir = tempfile::tempdir().unwrap();
    let first = plain(Arc::new(Float64Array::from(vec![1.5])));
    let later = vec![plain(Arc::new(Float32Array::from(vec![2.5f32])))];
    let (results, dataset) = write_rows(&dir, Casting::No, first, later).await;
    assert!(matches!(results[0], Err(Error::SchemaMismatch(_))));
    assert_eq!(dataset.count_rows(None).await.unwrap(), 1);
}

#[tokio::test]
async fn safe_widens_new_values() {
    let dir = tempfile::tempdir().unwrap();
    let first = plain(Arc::new(Float64Array::from(vec![1.5])));
    let later = vec![plain(Arc::new(Float32Array::from(vec![2.5f32, 3.25])))];
    let (results, dataset) = write_rows(&dir, Casting::Safe, first, later).await;
    results[0].as_ref().unwrap();
    let values = values(&dataset).await;
    let second = values[1].as_primitive::<Float64Type>();
    assert_eq!(second.values().to_vec(), [2.5, 3.25]);
    assert_eq!(stats_type(&dataset), DataType::Float64);
}

#[tokio::test]
async fn safe_rejects_a_narrowing() {
    let dir = tempfile::tempdir().unwrap();
    let first = plain(Arc::new(Float32Array::from(vec![1.5f32])));
    let later = vec![
        plain(Arc::new(Float64Array::from(vec![2.5]))),
        plain(Arc::new(Int64Array::from(vec![2]))),
    ];
    let (results, dataset) = write_rows(&dir, Casting::Safe, first, later).await;
    assert!(results.iter().all(|r| r.is_err()), "{results:?}");
    assert_eq!(dataset.count_rows(None).await.unwrap(), 1);
}

#[tokio::test]
async fn safe_follows_numpy() {
    // uint16 fits int32, and int32 fits float64.
    let dir = tempfile::tempdir().unwrap();
    let first = plain(Arc::new(Int32Array::from(vec![1])));
    let later = vec![plain(Arc::new(UInt16Array::from(vec![65_535u16])))];
    let (results, dataset) = write_rows(&dir, Casting::Safe, first, later).await;
    results[0].as_ref().unwrap();
    assert_eq!(
        values(&dataset).await[1]
            .as_primitive::<Int32Type>()
            .value(0),
        65_535
    );

    let dir = tempfile::tempdir().unwrap();
    let first = plain(Arc::new(Float64Array::from(vec![0.5])));
    let later = vec![plain(Arc::new(Int32Array::from(vec![7])))];
    let (results, _) = write_rows(&dir, Casting::Safe, first, later).await;
    results[0].as_ref().unwrap();
}

#[tokio::test]
async fn same_kind_narrows_new_values() {
    let dir = tempfile::tempdir().unwrap();
    let first = plain(Arc::new(Float32Array::from(vec![1.5f32])));
    let later = vec![plain(Arc::new(Float64Array::from(vec![0.1])))];
    let (results, dataset) = write_rows(&dir, Casting::SameKind, first, later).await;
    results[0].as_ref().unwrap();
    let values = values(&dataset).await;
    assert_eq!(values[1].as_primitive::<Float32Type>().value(0), 0.1f32);
    assert_eq!(stats_type(&dataset), DataType::Float32);
}

#[tokio::test]
async fn same_kind_rejects_overflow_and_other_kinds() {
    let dir = tempfile::tempdir().unwrap();
    let first = plain(Arc::new(Int32Array::from(vec![1])));
    let later = vec![
        plain(Arc::new(Int64Array::from(vec![5_000_000_000]))),
        plain(Arc::new(Int64Array::from(vec![42]))),
        plain(Arc::new(Float64Array::from(vec![1.0]))),
        plain(Arc::new(StringArray::from(vec!["7"]))),
    ];
    let (results, dataset) = write_rows(&dir, Casting::SameKind, first, later).await;
    assert!(results[0].is_err(), "overflow");
    results[1].as_ref().unwrap();
    assert!(results[2].is_err(), "float into int");
    assert!(results[3].is_err(), "text into int");
    let values = values(&dataset).await;
    assert_eq!(values.len(), 2);
    assert_eq!(values[1].as_primitive::<Int32Type>().value(0), 42);
}

#[tokio::test]
async fn attributes_follow_the_casting() {
    let dir = tempfile::tempdir().unwrap();
    let with = |fill: Attributes| row(Arc::new(Float64Array::from(vec![1.0])), fill);
    let first = with(Attributes::new().with("fill", -1.0f64));
    let later = vec![
        with(Attributes::new().with("fill", -2.0f32)),
        with(Attributes::new().with("fill", vec![1.0f64, 2.0])),
    ];
    let (results, dataset) = write_rows(&dir, Casting::Safe, first, later).await;
    results[0].as_ref().unwrap();
    assert!(results[1].is_err(), "a list is not a number");
    assert_eq!(dataset.count_rows(None).await.unwrap(), 2);
}

#[tokio::test]
async fn write_fn_casts_to_the_table() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("table.lance").to_str().unwrap().to_string();
    let rows = |rows: Vec<NdRecordBatch>| stream::iter(rows.into_iter().map(Ok));
    let first = plain(Arc::new(Float64Array::from(vec![1.5])));
    write(&uri, rows(vec![first]), &options(Casting::No))
        .await
        .unwrap();
    let narrow = plain(Arc::new(Float32Array::from(vec![2.5f32])));
    let append = WriteOptions {
        mode: WriteMode::Append,
        ..options(Casting::Safe)
    };
    let dataset = write(&uri, rows(vec![narrow]), &append).await.unwrap();
    let values = values(&dataset).await;
    assert_eq!(values[1].as_primitive::<Float64Type>().value(0), 2.5);
}
