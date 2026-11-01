//! Scale test: many variables across many datasets, then a projected read.
//!
//! Run in release mode:
//! `cargo test --release --test scale -- --ignored --nocapture`
//!
//! Set `LANCE_ND_DATASETS`, `LANCE_ND_VARIABLES`, `LANCE_ND_PER_DATASET`,
//! `LANCE_ND_PROJECTED`, `LANCE_ND_ROWS_PER_COMMIT`, `LANCE_ND_BATCH_SIZES`
//! and `LANCE_ND_DECODE_PARALLELISM` to change the sizes.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::Float64Array;
use arrow::datatypes::{DataType, Field, Schema};
use futures::{StreamExt, stream};
use nd_arrow_array::{Attributes, Dimensions, NdArrowArray, NdRecordBatch};

use lance_nd::{WriteOptions, scan, write};

/// Values per array, on an `(x, y)` grid.
const X: usize = 8;
const Y: usize = 8;

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct Sizes {
    datasets: usize,
    variables: usize,
    per_dataset: usize,
}

impl Sizes {
    /// The variables of dataset `index`. Consecutive datasets use other
    /// variables, so the table gets all of them early in the stream.
    fn variables_of(&self, index: usize) -> Vec<usize> {
        let mut ids: Vec<usize> = (0..self.per_dataset)
            .map(|j| (index * self.per_dataset + j * 7919) % self.variables)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn dataset(&self, index: usize) -> NdRecordBatch {
        let ids = self.variables_of(index);
        let fields: Vec<Field> = ids
            .iter()
            .map(|v| Field::new(name(*v), DataType::Float64, true))
            .collect();
        let arrays = ids
            .iter()
            .map(|v| {
                let values = (0..X * Y).map(|i| (index * 31 + v * 7 + i) as f64 * 0.25);
                NdArrowArray::try_new(
                    Arc::new(Float64Array::from_iter_values(values)),
                    Dimensions::try_new([("x", X), ("y", Y)]).unwrap(),
                )
                .unwrap()
                .with_attributes(Attributes::new().with("units", "K"))
            })
            .collect();
        NdRecordBatch::try_new_inferred(Arc::new(Schema::new(fields)), arrays)
            .unwrap()
            .with_attributes(Attributes::new().with("id", format!("ds{index}")))
    }
}

/// The size of the data files of the table, in bytes.
fn data_size(uri: &str) -> u64 {
    std::fs::read_dir(std::path::Path::new(uri).join("data"))
        .unwrap()
        .map(|entry| entry.unwrap().metadata().unwrap().len())
        .sum()
}

/// The scan batch sizes to time, from `LANCE_ND_BATCH_SIZES`, for example `1,64`.
fn batch_sizes() -> Vec<usize> {
    std::env::var("LANCE_ND_BATCH_SIZES")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 64, 256])
}

fn name(variable: usize) -> String {
    format!("var{variable:05}")
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "scale test, run in release mode with --ignored"]
async fn projected_read_of_many_variables() {
    let sizes = Sizes {
        datasets: env("LANCE_ND_DATASETS", 10_000),
        variables: env("LANCE_ND_VARIABLES", 2_048),
        per_dataset: env("LANCE_ND_PER_DATASET", 20),
    };
    let projected = env("LANCE_ND_PROJECTED", 20);
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("scale.lance");
    let uri = uri.to_str().unwrap();

    let start = Instant::now();
    let rows = stream::iter((0..sizes.datasets).map(|i| Ok(sizes.dataset(i))));
    let options = WriteOptions {
        max_rows_per_commit: env("LANCE_ND_ROWS_PER_COMMIT", 1024),
        ..WriteOptions::default()
    };
    let dataset = write(uri, rows, &options).await.unwrap();
    let write_time = start.elapsed();
    let columns = dataset.schema().fields.len();
    assert_eq!(columns, sizes.variables + 1);
    println!(
        "write: {} datasets, {} variables, {} per dataset: {:.2?} ({:.0} datasets/s), {} versions, {} fragments, {:.1} MB",
        sizes.datasets,
        sizes.variables,
        sizes.per_dataset,
        write_time,
        sizes.datasets as f64 / write_time.as_secs_f64(),
        dataset.version().version,
        dataset.get_fragments().len(),
        data_size(uri) as f64 / 1e6,
    );

    // Spread the projected variables over the whole range.
    let ids: Vec<usize> = (0..projected)
        .map(|k| k * sizes.variables / projected)
        .collect();
    let names: Vec<String> = ids.iter().map(|v| name(*v)).collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let expected_arrays: usize = (0..sizes.datasets)
        .map(|i| {
            let have = sizes.variables_of(i);
            ids.iter().filter(|v| have.binary_search(v).is_ok()).count()
        })
        .sum();

    for batch_size in batch_sizes() {
        let start = Instant::now();
        let mut scanner = scan(&dataset).project(&names).batch_size(batch_size);
        if let Ok(parallelism) = std::env::var("LANCE_ND_DECODE_PARALLELISM") {
            scanner = scanner.decode_parallelism(parallelism.parse().unwrap());
        }
        let mut stream = scanner.try_into_stream().await.unwrap();
        let (mut rows, mut arrays, mut values) = (0usize, 0usize, 0usize);
        while let Some(batch) = stream.next().await {
            let batch = batch.unwrap();
            rows += 1;
            arrays += batch.columns().len();
            values += batch.columns().iter().map(|c| c.len()).sum::<usize>();
        }
        let read_time = start.elapsed();
        println!(
            "read: batch size {batch_size}, {projected} of {} variables, {rows} rows, {arrays} arrays, {values} values: {:.2?} ({:.0} rows/s, {:.0} arrays/s)",
            sizes.variables,
            read_time,
            rows as f64 / read_time.as_secs_f64(),
            arrays as f64 / read_time.as_secs_f64(),
        );

        assert_eq!(rows, sizes.datasets);
        assert_eq!(arrays, expected_arrays);
        assert_eq!(values, expected_arrays * X * Y);
    }
}
