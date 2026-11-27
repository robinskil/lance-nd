<p align="center">
  <img src="assets/logo-wordmark.svg" alt="lance-nd" width="420">
</p>

<h3 align="center">N-dimensional Arrow arrays, stored in Lance</h3>

<p align="center">
  Stream nd record batches into a Lance table and read them back,<br>
  one nd record batch per row, with projection, filters and schema evolution.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-experimental-orange" alt="Status: experimental">
  <a href="https://crates.io/crates/lance-nd"><img src="https://img.shields.io/crates/v/lance-nd" alt="crates.io"></a>
  <a href="https://docs.rs/lance-nd"><img src="https://img.shields.io/docsrs/lance-nd" alt="docs.rs"></a>
  <a href="https://pypi.org/project/lance_nd/"><img src="https://img.shields.io/pypi/v/lance_nd" alt="PyPI"></a>
  <a href="https://github.com/robinskil/lance-nd/actions/workflows/ci.yml"><img src="https://github.com/robinskil/lance-nd/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License: Apache-2.0"></a>
  <img src="https://img.shields.io/badge/rust-1.91%2B-orange?logo=rust" alt="Rust 1.91+">
  <img src="https://img.shields.io/badge/lance-14-1E3A8A" alt="Lance 14">
  <img src="https://img.shields.io/badge/arrow-58-1E3A8A" alt="Arrow 58">
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#rust">Rust</a> ·
  <a href="#python">Python</a> ·
  <a href="https://github.com/robinskil/lance-nd/wiki">Wiki</a> ·
  <a href="FORMAT.md">Specification</a>
</p>

---

lance-nd stores N-dimensional array data in a
[Lance](https://lancedb.github.io/lance/) table. Each table row is one nd
record batch of [nd-arrow-array](https://github.com/robinskil/nd-arrow-array):
a set of named N-dimensional arrays with their dimensions and attributes,
such as one xarray dataset.

- The Lance table schema **is** the nd-arrow encoding, so any Arrow or Lance
  reader can read the data.
- Rows can have different variables. New variables and attributes extend
  the table schema.
- Optional per-row stats let a scan skip rows on value filters.
- A variable can get a wider type, for example `Float32` to `Float64`. Only
  that column is written again, also in a table with thousands of variables.
- Tables live on local disk, S3, Google Cloud Storage or Azure.

## Install

```toml
[dependencies]
lance-nd = "0.3"
nd-arrow-array = "0.5"
```

Turn on the cargo feature `aws`, `gcp` or `azure` for an object store.
Lance needs `protoc` to build.

## Rust

```rust
use futures::{StreamExt, stream};
use lance_nd::{WriteOptions, scan, write};

// `batches` is an iterator of nd_arrow_array::NdRecordBatch.
let rows = stream::iter(batches.map(Ok));
let table = write("data/arrays.lance", rows, &WriteOptions::default()).await?;

let mut rows = scan(&table)
    .project(&["temperature"])
    .filter("__global_attrs__.id = 'a'")
    .try_into_stream()
    .await?;
while let Some(batch) = rows.next().await {
    println!("{:?}", batch?.target());
}
```

## Python

```python
import xarray as xr
import lance_nd

with lance_nd.create("data/arrays.lance", stats=True) as writer:
    writer.write(xr.open_dataset("obs.nc"))

table = lance_nd.open("data/arrays.lance")
for batch in table.scan(project=["sst"], prune="sst.values > 300"):
    ds = batch.to_xarray()
```

See [python/README.md](python/README.md) to install the package.

## Documentation

The [wiki](https://github.com/robinskil/lance-nd/wiki) explains each part in depth:

- [Table layout](https://github.com/robinskil/lance-nd/wiki/Table-Layout): the data model, columns and schema evolution.
- [Writing](https://github.com/robinskil/lance-nd/wiki/Writing): `NdWriter`, errors, fragments, casting and column encoding.
- [Reading and filters](https://github.com/robinskil/lance-nd/wiki/Reading-and-Filters): stats, pruning and filters.
- [Object stores](https://github.com/robinskil/lance-nd/wiki/Object-Stores): S3, Google Cloud Storage and Azure.
- [Delete, compact and versions](https://github.com/robinskil/lance-nd/wiki/Maintenance).
- [Python](https://github.com/robinskil/lance-nd/wiki/Python): the full Python API and the encoding rules.

[FORMAT.md](FORMAT.md) is the specification of the lance-nd table.

## Tests

```bash
cargo test --workspace
```

The scale and performance tests do not run by default. Run them in release
mode:

```bash
cargo test --release --test scale -- --ignored --nocapture
cargo test --release --test perf -- --ignored --nocapture --test-threads 1
```

The `perf` tests time a widen, an append, a sparse write and a compaction
against the number of variables, and measure the memory of a writer. Environment variables such
as `LANCE_ND_WIDTHS` and `LANCE_ND_FILES` set the sizes, see the top of
`tests/perf.rs`.

## Status

lance-nd is experimental. It has no indexes. A table is a `lance::Dataset`,
so a new major version of Lance is a breaking change of the Rust API.

lance-nd 0.3 uses Lance 14. It reads and changes the tables of lance-nd 0.2,
and lance-nd 0.2 reads and changes the tables of 0.3.

From lance-nd 0.4, `target_fragment_bytes` is a size in memory, for writes
and compactions. A fragment is written from one batch of about this size, so
a write holds about twice this size in memory. The fragments are smaller on
disk, as Lance compresses them. Each row stores its size in memory in the
column `__stats__memory`, so a compaction plans without reading the data.
Tables stay the same format, and lance-nd 0.2 and 0.3 ignore the column.

For a table with many variables, where each row has only some of them, use
the cluster mode of compaction (`CompactOptions::cluster`). It merges
fragments with similar variables from anywhere in the table, so fragments
stay narrow and compaction stays fast. It does not keep the row order.
