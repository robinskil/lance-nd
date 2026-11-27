//! Performance checks: widen and commit time against schema width, and
//! writer memory against the row count.
//!
//! Run in release mode:
//! `cargo test --release --test perf -- --ignored --nocapture --test-threads 1`
//!
//! Set `LANCE_ND_WIDTHS` (for example `250,1000,4000`), `LANCE_ND_NEW_FIELDS`,
//! `LANCE_ND_FILES`, `LANCE_ND_VARIABLES`, `LANCE_ND_PER_FILE`, `LANCE_ND_GRID`,
//! `LANCE_ND_TARGET_MB` and `LANCE_ND_KINDS` to change the sizes.

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

/// A table with `width` Float32 variables in 4 rows that each hold all
/// variables. Dense rows keep the setup fast at large widths.
async fn table(uri: &str, width: usize, options: &WriteOptions) -> Dataset {
    let ids: Vec<usize> = (0..width).collect();
    let rows: Vec<_> = (0..4).map(|i| Ok(row(&ids, 2, false, i))).collect();
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
    println!(
        "{:>8} {:>10} {:>10} {:>10}",
        "width", "setup s", "widen s", "s/width^2"
    );
    for width in widths() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("t.lance");
        let options = WriteOptions {
            stats: true,
            ..Default::default()
        };
        let setup = Instant::now();
        let mut dataset = table(uri.to_str().unwrap(), width, &options).await;
        let setup = secs(setup.elapsed());
        let start = Instant::now();
        widen(&mut dataset, &name(0), &DataType::Float64)
            .await
            .unwrap();
        let time = start.elapsed();
        println!(
            "{width:>8} {setup:>10.3} {:>10.3} {:>10.3e}",
            secs(time),
            secs(time) / (width * width) as f64
        );
    }
}

/// Rows of 20 variables each, `width` variables in all, no two rows alike.
fn sparse_rows(width: usize) -> Vec<lance_nd::Result<NdRecordBatch>> {
    (0..width)
        .step_by(20)
        .enumerate()
        .map(|(i, first)| {
            let ids: Vec<usize> = (first..(first + 20).min(width)).collect();
            Ok(row(&ids, 8, false, i))
        })
        .collect()
}

/// Compact a sparse table with `width` variables, in order and in cluster
/// mode, and time a read of one row with its own variables after it.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "performance check, run in release mode with --ignored"]
async fn compact_time_against_width() {
    println!("compact: a sparse table, 20 variables per row, 1 row per fragment group");
    println!(
        "{:>8} {:>10} {:>18} {:>18} {:>12}",
        "width", "fragments", "ordered s / frags", "cluster s / frags", "read row s"
    );
    for width in widths() {
        let mut results = Vec::new();
        let mut read = 0.0;
        let mut before = 0;
        for cluster in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let uri = dir.path().join("t.lance");
            let mut writer = NdWriter::create(uri.to_str().unwrap(), &WriteOptions::default())
                .await
                .unwrap();
            writer
                .write(stream::iter(sparse_rows(width)))
                .await
                .unwrap();
            let mut dataset = writer.finish().await.unwrap();
            before = dataset.get_fragments().len();
            let options = lance_nd::CompactOptions {
                cluster,
                ..Default::default()
            };
            let start = Instant::now();
            lance_nd::compact(&mut dataset, &options).await.unwrap();
            results.push((secs(start.elapsed()), dataset.get_fragments().len()));
            if cluster {
                let target = width / 40;
                let own: Vec<String> = (target * 20..target * 20 + 20).map(name).collect();
                let own: Vec<&str> = own.iter().map(String::as_str).collect();
                let start = Instant::now();
                let rows: Vec<NdRecordBatch> = lance_nd::scan(&dataset)
                    .filter(format!("__global_attrs__.id = 'file{target}'"))
                    .project(&own)
                    .try_collect()
                    .await
                    .unwrap();
                assert_eq!(rows.len(), 1);
                read = secs(start.elapsed());
            }
        }
        println!(
            "{width:>8} {before:>10} {:>10.3} {:>7} {:>10.3} {:>7} {read:>12.3}",
            results[0].0, results[0].1, results[1].0, results[1].1
        );
    }
}

/// Time a create of a table with `width` variables, 20 variables per row.
/// Each row holds other variables, so the table is sparse.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "performance check, run in release mode with --ignored"]
async fn sparse_write_time_against_width() {
    println!("write: 20 variables per row, `width` variables in the table");
    println!(
        "{:>8} {:>8} {:>10} {:>10}",
        "width", "rows", "write s", "s/width^2"
    );
    for width in widths() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("t.lance");
        let rows = sparse_rows(width);
        let count = rows.len();
        let start = Instant::now();
        let mut writer = NdWriter::create(uri.to_str().unwrap(), &WriteOptions::default())
            .await
            .unwrap();
        writer.write(stream::iter(rows)).await.unwrap();
        writer.finish().await.unwrap();
        let time = secs(start.elapsed());
        println!(
            "{width:>8} {count:>8} {time:>10.3} {:>10.3e}",
            time / (width * width) as f64
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

/// The variables of file `index`. With `kinds` above 0, each file is one of
/// `kinds` kinds in a mixed order, and a kind has its own `per_file`
/// variables. Else each file has its own mix of the `variables`.
fn file_ids(index: usize, kinds: usize, per_file: usize, variables: usize) -> Vec<usize> {
    let ids: std::collections::BTreeSet<usize> = if kinds > 0 {
        let kind = (index.wrapping_mul(2_654_435_761) >> 7) % kinds;
        (0..per_file)
            .map(|j| (kind * per_file + j) % variables)
            .collect()
    } else {
        (0..per_file)
            .map(|j| (index * per_file + j * 7919) % variables)
            .collect()
    };
    ids.into_iter().collect()
}

/// Write `files` files one by one, then compact in `cluster` mode or not.
/// Gives the write time, the fragments, the compaction time, the fragments
/// after it, and the time of a read of the variables of the first file.
async fn grouped_run(group: bool, cluster: bool, kinds: usize) -> [f64; 5] {
    let files = env("LANCE_ND_FILES", 3000);
    let per_file = env("LANCE_ND_PER_FILE", 20);
    let variables = env("LANCE_ND_VARIABLES", 2000);
    let grid = env("LANCE_ND_GRID", 16);
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("t.lance");
    let options = WriteOptions {
        target_fragment_bytes: env("LANCE_ND_TARGET_MB", 8) << 20,
        group_variables: group,
        ..Default::default()
    };
    let start = Instant::now();
    let mut writer = NdWriter::create(uri.to_str().unwrap(), &options)
        .await
        .unwrap();
    for index in 0..files {
        let ids = file_ids(index, kinds, per_file, variables);
        writer
            .write(stream::iter([Ok(row(&ids, grid, false, index))]))
            .await
            .unwrap();
    }
    let mut dataset = writer.finish().await.unwrap();
    let write = secs(start.elapsed());
    let written = dataset.get_fragments().len();
    let compact_options = lance_nd::CompactOptions {
        target_fragment_bytes: options.target_fragment_bytes as u64,
        cluster,
        ..Default::default()
    };
    let start = Instant::now();
    lance_nd::compact(&mut dataset, &compact_options)
        .await
        .unwrap();
    let compact = secs(start.elapsed());
    let compacted = dataset.get_fragments().len();
    let own: Vec<String> = file_ids(0, kinds, per_file, variables)
        .into_iter()
        .map(name)
        .collect();
    let own: Vec<&str> = own.iter().map(String::as_str).collect();
    let start = Instant::now();
    let rows: Vec<NdRecordBatch> = lance_nd::scan(&dataset)
        .project(&own)
        .try_collect()
        .await
        .unwrap();
    assert!(!rows.is_empty());
    let read = secs(start.elapsed());
    [write, written as f64, compact, compacted as f64, read]
}

/// Compare plain writes with variable groups, then each compaction mode.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "performance check, run in release mode with --ignored"]
async fn grouped_write_and_compact() {
    let kinds = env("LANCE_ND_KINDS", 50);
    for kinds in [kinds, 0] {
        match kinds {
            0 => println!("files with their own mix of variables"),
            _ => println!("files of {kinds} kinds in a mixed order"),
        }
        println!(
            "{:>8} {:>8} {:>9} {:>6} {:>10} {:>6} {:>8}",
            "groups", "compact", "write s", "frags", "compact s", "frags", "read s"
        );
        for group in [false, true] {
            for cluster in [false, true] {
                let [write, written, compact, compacted, read] =
                    grouped_run(group, cluster, kinds).await;
                let mode = if cluster { "cluster" } else { "ordered" };
                println!(
                    "{group:>8} {mode:>8} {write:>9.2} {written:>6} {compact:>10.2} {compacted:>6} {read:>8.3}"
                );
            }
        }
    }
}
