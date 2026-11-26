//! Cross-version check: create a table with one lance-nd build, read and
//! change it with another. Set `LANCE_ND_COMPAT_DIR` to a shared folder.

use std::sync::Arc;

use arrow::array::AsArray;
use arrow::array::{ArrayRef, Float32Array, Int32Array};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Float64Type, Schema};
use futures::stream;
use lance_nd::datafusion::functions::core::expr_fn::get_field;
use lance_nd::datafusion::prelude::{col, lit};
use lance_nd::{
    Casting, CompactOptions, NdWriter, WriteOptions, compact, delete, open, scan, widen,
};
use nd_arrow_array::{Attributes, Dimensions, NdArrowArray, NdRecordBatch};

fn uri() -> String {
    let dir = std::env::var("LANCE_ND_COMPAT_DIR").expect("set LANCE_ND_COMPAT_DIR");
    format!("{dir}/t.lance")
}

fn row(index: usize, with_c: bool) -> NdRecordBatch {
    let dims = || Dimensions::try_new([("x", 3)]).unwrap();
    let base = index as f32 * 10.0;
    let a: ArrayRef = Arc::new(Float32Array::from(vec![base, base + 1.0, base + 2.0]));
    let b: ArrayRef = Arc::new(Int32Array::from(vec![index as i32; 3]));
    let mut fields = vec![
        Field::new("a", DataType::Float32, true),
        Field::new("b", DataType::Int32, true),
    ];
    let mut arrays = vec![
        NdArrowArray::try_new(a, dims()).unwrap(),
        NdArrowArray::try_new(b, dims()).unwrap(),
    ];
    if with_c {
        fields.push(Field::new("c", DataType::Float32, true));
        let c: ArrayRef = Arc::new(Float32Array::from(vec![0.5; 3]));
        arrays.push(NdArrowArray::try_new(c, dims()).unwrap());
    }
    NdRecordBatch::try_new_inferred(Arc::new(Schema::new(fields)), arrays)
        .unwrap()
        .with_attributes(Attributes::new().with("id", format!("row{index}")))
}

async fn write_rows(rows: Vec<NdRecordBatch>, create: bool) {
    let options = WriteOptions {
        stats: true,
        casting: Casting::Safe,
        ..Default::default()
    };
    let mut writer = if create {
        NdWriter::create(&uri(), &options).await.unwrap()
    } else {
        NdWriter::append(&uri(), &options).await.unwrap()
    };
    writer
        .write(stream::iter(rows.into_iter().map(Ok)))
        .await
        .unwrap();
    writer.finish().await.unwrap();
}

/// Print the rows, the sum of each variable and a pruned scan.
async fn report() {
    let dataset = open(&uri()).await.unwrap();
    let rows: Vec<NdRecordBatch> = scan(&dataset).try_collect().await.unwrap();
    println!(
        "version {}, {} rows, {} fragments",
        dataset.version().version,
        rows.len(),
        dataset.get_fragments().len()
    );
    for name in ["a", "b", "c"] {
        let mut sum = 0.0;
        let mut kinds = Vec::new();
        for row in &rows {
            if let Some(column) = row.column_by_name(name) {
                let values = column.values();
                kinds.push(values.data_type().to_string());
                let values = cast(values, &DataType::Float64).unwrap();
                sum += values
                    .as_primitive::<Float64Type>()
                    .values()
                    .iter()
                    .sum::<f64>();
            }
        }
        kinds.dedup();
        println!("  {name}: sum {sum}, types {kinds:?}");
    }
    let pruned: Vec<NdRecordBatch> = scan(&dataset)
        .filter_expr(get_field(col("a"), "values").gt(lit(25.0)))
        .try_collect()
        .await
        .unwrap();
    println!("  rows with a > 25: {}", pruned.len());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "cross-version check"]
async fn compat_create() {
    let _ = std::fs::remove_dir_all(uri());
    write_rows((0..3).map(|i| row(i, false)).collect(), true).await;
    write_rows(vec![row(3, false)], false).await;
    let mut dataset = open(&uri()).await.unwrap();
    widen(&mut dataset, "a", &DataType::Float64).await.unwrap();
    report().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "cross-version check"]
async fn compat_read() {
    report().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "cross-version check"]
async fn compat_change() {
    write_rows(vec![row(4, true), row(5, true)], false).await;
    let mut dataset = open(&uri()).await.unwrap();
    widen(&mut dataset, "b", &DataType::Int64).await.unwrap();
    delete(&mut dataset, "__global_attrs__.id = 'row0'")
        .await
        .unwrap();
    compact(&mut dataset, &CompactOptions::default())
        .await
        .unwrap();
    report().await;
}
