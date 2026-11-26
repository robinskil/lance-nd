//! Performance checks: widen and commit time against schema width, and
//! writer memory against the row count.
//!
//! Run in release mode:
//! `cargo test --release --test perf -- --ignored --nocapture --test-threads 1`
//!
//! Set `LANCE_ND_WIDTHS` (for example `250,1000,4000`), `LANCE_ND_NEW_FIELDS`,
//! `LANCE_ND_FILES`, `LANCE_ND_VARIABLES`, `LANCE_ND_PER_FILE`, `LANCE_ND_GRID`
//! and `LANCE_ND_TARGET_MB` to change the sizes.

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{ArrayRef, Float32Array, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use futures::stream;
use lance::Dataset;
use lance_nd::{Casting, NdWriter, WriteOptions, widen};
use nd_arrow_array::{Attributes, Dimensions, NdArrowArray, NdRecordBatch};

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn widths() -> Vec<usize> {
    std::env::var("LANCE_ND_WIDTHS")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![250, 500, 1000, 2000])
}

fn name(variable: usize) -> String {
    format!("var{variable:05}")
}

/// One row with the variables `ids`, each `grid` x `grid` values.
fn row(ids: &[usize], grid: usize, wide: bool, index: usize) -> NdRecordBatch {
    let kind = if wide {
        DataType::Float64
    } else {
        DataType::Float32
    };
    let fields: Vec<Field> = ids
        .iter()
        .map(|v| Field::new(name(*v), kind.clone(), true))
        .collect();
    let arrays = ids
        .iter()
        .map(|v| {
            let values = (0..grid * grid).map(|i| ((index * 31 + v * 7 + i) % 1000) as f32);
            let values: ArrayRef = if wide {
                Arc::new(Float64Array::from_iter_values(values.map(f64::from)))
            } else {
                Arc::new(Float32Array::from_iter_values(values))
            };
            NdArrowArray::try_new(
                values,
                Dimensions::try_new([("x", grid), ("y", grid)]).unwrap(),
            )
            .unwrap()
            .with_attributes(Attributes::new().with("units", "K"))
        })
        .collect();
    NdRecordBatch::try_new_inferred(Arc::new(Schema::new(fields)), arrays)
        .unwrap()
        .with_attributes(Attributes::new().with("id", format!("file{index}")))
}

/// A table with `width` Float32 variables, 20 variables per row.
async fn table(uri: &str, width: usize, options: &WriteOptions) -> Dataset {
    let rows: Vec<_> = (0..width)
        .step_by(20)
        .enumerate()
        .map(|(i, first)| {
            let ids: Vec<usize> = (first..(first + 20).min(width)).collect();
            Ok(row(&ids, 8, false, i))
        })
        .collect();
    let mut writer = NdWriter::create(uri, options).await.unwrap();
    writer.write(stream::iter(rows)).await.unwrap();
    writer.finish().await.unwrap()
}

fn secs(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "performance check, run in release mode with --ignored"]
async fn widen_time_against_width() {
    println!("widen: one Float32 variable to Float64, the data of the variable stays the same");
    println!("{:>8} {:>10} {:>10}", "width", "widen s", "s/width^2");
    for width in widths() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("t.lance");
        let options = WriteOptions {
            stats: true,
            ..Default::default()
        };
        let mut dataset = table(uri.to_str().unwrap(), width, &options).await;
        let start = Instant::now();
        widen(&mut dataset, &name(0), &DataType::Float64)
            .await
            .unwrap();
        let time = start.elapsed();
        println!(
            "{width:>8} {:>10.3} {:>10.3e}",
            secs(time),
            secs(time) / (width * width) as f64
        );
    }
}

/// Append `new` new variables to a table of `width` variables. With
/// `promote`, the first row widens `var00000`.
async fn append_time(width: usize, new: usize, promote: bool, defer: bool) -> (f64, f64) {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("t.lance");
    let uri = uri.to_str().unwrap();
    let base = WriteOptions {
        stats: true,
        ..Default::default()
    };
    table(uri, width, &base).await;
    let options = WriteOptions {
        casting: Casting::Promote,
        defer_promotion: defer,
        ..Default::default()
    };
    let start = Instant::now();
    let mut writer = NdWriter::append(uri, &options).await.unwrap();
    let mut rows = Vec::new();
    if promote {
        rows.push(Ok(row(&[0], 8, true, 0)));
    }
    for first in (width..width + new).step_by(10) {
        let ids: Vec<usize> = (first..(first + 10).min(width + new)).collect();
        rows.push(Ok(row(&ids, 8, false, first)));
    }
    writer.write(stream::iter(rows)).await.unwrap();
    let written = start.elapsed();
    let finish = Instant::now();
    let dataset = writer.finish().await.unwrap();
    let finish = finish.elapsed();
    assert_eq!(
        Schema::from(dataset.schema())
            .fields()
            .iter()
            .filter(|f| f.name().starts_with("var"))
            .count(),
        width + new
    );
    (secs(written), secs(finish))
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "performance check, run in release mode with --ignored"]
async fn append_time_against_width() {
    let new = env("LANCE_ND_NEW_FIELDS", 50);
    println!("append: {new} new variables to a table of `width` variables");
    println!(
        "{:>8} {:>22} {:>22} {:>22}",
        "width", "plain write/finish s", "promote write/finish s", "deferred write/finish s"
    );
    for width in widths() {
        let plain = append_time(width, new, false, false).await;
        let promote = append_time(width, new, true, false).await;
        let deferred = append_time(width, new, true, true).await;
        println!(
            "{width:>8} {:>10.3} {:>11.3} {:>10.3} {:>11.3} {:>10.3} {:>11.3}",
            plain.0, plain.1, promote.0, promote.1, deferred.0, deferred.1
        );
    }
}

/// The resident memory of this process, in MB.
#[cfg(windows)]
fn resident_mb() -> f64 {
    #[repr(C)]
    #[derive(Default)]
    struct Counters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(process: isize, counters: *mut Counters, cb: u32) -> i32;
    }
    let mut counters = Counters {
        cb: std::mem::size_of::<Counters>() as u32,
        ..Default::default()
    };
    // SAFETY: the counters struct matches PROCESS_MEMORY_COUNTERS and lives on the stack.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    counters.working_set_size as f64 / 1e6
}

#[cfg(not(windows))]
fn resident_mb() -> f64 {
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let pages: f64 = statm
        .split_whitespace()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    pages * 4096.0 / 1e6
}

async fn memory_run(target_mb: usize) {
    let files = env("LANCE_ND_FILES", 5000);
    let variables = env("LANCE_ND_VARIABLES", 2000);
    let per_file = env("LANCE_ND_PER_FILE", 20);
    let grid = env("LANCE_ND_GRID", 32);
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("t.lance");
    let uri = uri.to_str().unwrap();
    // A small table first, so the writer appends and adds fields as it goes.
    table(uri, 20, &WriteOptions::default()).await;
    let options = WriteOptions {
        target_fragment_bytes: target_mb << 20,
        ..Default::default()
    };
    println!(
        "memory: {files} files, {variables} variables, {per_file} per file, \
         {grid}x{grid} values, target fragment {target_mb} MB"
    );
    let before = resident_mb();
    let mut writer = NdWriter::append(uri, &options).await.unwrap();
    let start = Instant::now();
    for index in 0..files {
        let ids: Vec<usize> = (0..per_file)
            .map(|j| (index * per_file + j * 7919) % variables)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        // One write per file, as the CLI does.
        writer
            .write(stream::iter([Ok(row(&ids, grid, false, index))]))
            .await
            .unwrap();
        if (index + 1) % 500 == 0 {
            println!(
                "{:>6} files {:>8.0} MB resident {:>8.1} s",
                index + 1,
                resident_mb() - before,
                secs(start.elapsed())
            );
        }
    }
    let finish = Instant::now();
    let dataset = writer.finish().await.unwrap();
    println!(
        "finish {:.2} s, {:.0} MB resident, {} fragments",
        secs(finish.elapsed()),
        resident_mb() - before,
        dataset.get_fragments().len()
    );
    drop(dataset);
    println!("after drop {:.0} MB resident", resident_mb() - before);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "performance check, run in release mode with --ignored"]
async fn writer_memory_against_files() {
    memory_run(env("LANCE_ND_TARGET_MB", 128)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "performance check, run in release mode with --ignored"]
async fn writer_memory_with_small_fragments() {
    memory_run(8).await;
}
