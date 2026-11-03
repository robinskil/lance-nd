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
  <img src="https://img.shields.io/badge/version-0.1.0-blue" alt="Version 0.1.0">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License: Apache-2.0"></a>
  <img src="https://img.shields.io/badge/rust-1.91%2B-orange?logo=rust" alt="Rust 1.91+">
  <img src="https://img.shields.io/badge/lance-11-1E3A8A" alt="Lance 11">
  <img src="https://img.shields.io/badge/arrow-58-1E3A8A" alt="Arrow 58">
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#data-model">Data model</a> ·
  <a href="#the-lance-nd-table">lance-nd table</a> ·
  <a href="#usage">Usage</a> ·
  <a href="#python">Python</a> ·
  <a href="FORMAT.md">Specification</a>
</p>

---

## Overview

lance-nd is a general way to store N-dimensional array data in a
[Lance](https://lancedb.github.io/lance/) table. It has two layers:

| Layer | What it does |
|---|---|
| [nd-arrow-array](https://github.com/robinskil/nd-arrow-array) | Encodes nd arrow arrays and nd record batches as plain Apache Arrow data. |
| lance-nd | Stores that Arrow data in Lance, one nd record batch per row. |

The Lance table schema **is** the nd-arrow encoding. lance-nd adds no extra
columns and no private layout, so any Arrow or Lance reader can read the data.
The one exception is opt-in: a table with stats has one extra `__stats__`
column, see [Stats](#stats).

## Data model

lance-nd stores the data model of
[nd-arrow-array](https://github.com/robinskil/nd-arrow-array). That repo
defines the encoding. In short:

- An **nd arrow array** is one N-dimensional array. As an Arrow column, it
  is the extension type `nd_arrow.array` over a struct:

  ```text
  Struct {
    values:    List<T>,       // the flat values, in C order
    dim_names: List<Utf8>,    // the name of each dimension
    dim_sizes: List<UInt32>,  // the size of each dimension
    attrs:     Struct { .. }, // optional attributes, such as units
  }
  ```

- An **nd record batch** is a set of named nd arrow arrays on one shared
  grid, plus optional global attributes in the struct column
  `__global_attrs__`.

In Rust, these are `NdArrowArray` and `NdRecordBatch`. See the
[nd-arrow FORMAT.md](https://github.com/robinskil/nd-arrow-array/blob/main/FORMAT.md)
for the full rules.

## The lance-nd table

A lance-nd table is a Lance table. Each table row is one nd record batch:
a complete one, or one chunk of a larger one.

| Column | Type | Nullable |
|---|---|---|
| one column per array name | `nd_arrow.array` | yes |
| `__global_attrs__` | `nd_arrow.global_attributes` | yes |

For example, a table with three nd record batches:

| Row | `x` | `y` | `temperature` | `pressure` | `__global_attrs__` |
|---|---|---|---|---|---|
| 0 | `(x=3)` | `(y=4)` | `(x=3, y=4)` | absent | `{id: "a"}` |
| 1 | absent | absent | absent | `(station=3)` | `{id: "b"}` |
| 2 | `(x=2)` | `(y=4)` | `(x=2, y=4)` | absent | `{id: "c"}` |

### Array columns

- The table has one column for each array name in any row.
- A null in a row means that the nd record batch of that row does not have
  that array. A reader leaves it out of the row.
- One array name has one value type in a table. A write that changes the type
  fails.

### Schema evolution

A write can bring new array names, new attribute keys and new global
attributes. lance-nd adds them to the table schema first, then appends the
rows. Older rows read the new arrays as absent.

Lance can add nested fields only with file version 2.2 or later, so lance-nd
creates tables with file version 2.2.

### Rows and chunks

The format does not separate a complete nd record batch from a chunk, and it
has no reserved columns for names or positions. Put them in the global
attributes, for example `id` or `chunk_offset`, then filter on them:

```sql
__global_attrs__.id = 'a'
```

### Stats

A table created with `WriteOptions { stats: true, .. }` stores the min and the
max of each variable per row, in one extra `__stats__` column. It has one
child per variable, `__stats__.<variable>.min` and `.max`. Null and NaN
values do not count. Numeric, string, timestamp, date, time and duration
variables get stats. An append writes stats when the table has them.

`lance_nd::datafusion` is the DataFusion version of Lance. Use it to build
filters. Older lance-nd releases cannot read a table with stats.

### Pruning

A scan with `filter_expr` uses the stats to skip the rows that surely have no
matching value. lance-nd changes each part of the filter that reads
`<variable>.values` into a test on the min and the max of the row:

| Filter on the values | Filter on the stats |
|---|---|
| `v = x` | `min <= x AND max >= x` |
| `v < x`, `v <= x` | `min < x`, `min <= x` |
| `v > x`, `v >= x` | `max > x`, `max >= x` |
| `v BETWEEN a AND b` | `max >= a AND min <= b` |
| `v IN (x, y)` | The `=` test for each value, joined with `OR` |

Then Lance reads only the rows that pass the stats test. For example, with
the filter `lat.values BETWEEN 17.5 AND 25.4`:

| Row | `lat` min | `lat` max | Result |
|---|---|---|---|
| 0 | -90 | -80 | skipped |
| 1 | 17 | 30 | kept |
| 2 | absent | absent | skipped: the row does not have `lat` |
| 3 | null | null | kept: no value counts, for example all NaN |

- A kept row comes back whole. It can hold values outside the filter.
- A value part without a stats form keeps all rows, for example `NOT`,
  `!=` or `t.values * 2 > 10`.
- A table without stats keeps all rows for a value part.

### Filters

A filter for `filter_expr` uses the Lance table columns:

```rust
use lance_nd::datafusion::functions::core::expr_fn::get_field;
use lance_nd::datafusion::prelude::{col, lit};

let lat = get_field(col("lat"), "values");
let units = get_field(get_field(col("sst"), "attrs"), "units");
let id = get_field(col("__global_attrs__"), "id");
let rows = scan(&table)
    .filter_expr(lat.between(lit(17.5), lit(25.4)).and(units.eq(lit("K")).or(id.eq(lit("a")))))
    .try_into_stream()
    .await?;
```

lance-nd does not check or change the column names. A wrong name gives an
error in the scan. In an `Expr`, write a nested column such as `sst.values`
with `get_field`: Lance does not resolve `col("sst.values")`.

A part that does not read values, for example on `__global_attrs__.id` or
`sst.attrs.units`, filters the rows exactly, so it stays as it is. Parts
combine with these rules:

- `AND`: a side without a stats form goes away.
- `OR`: both sides need a form, or the full `OR` keeps all rows.
- `NOT`: only over parts that do not read values.

Use one filter API for each case:

| Filter | API |
|---|---|
| Lance SQL text | `filter` |
| DataFusion `Expr` on the table columns, with values as stats | `filter_expr` |
| DataFusion `Expr` on the Lance table columns, with no change | `lance_filter` |

`filter_expr` and `lance_filter` can be used together. A row must then
match both. `filter` cannot be used with one of them.

### Column encoding

Each column can have its own Lance encoding: compression, compression level,
byte stream split, run-length encoding and structural encoding. The table
stores default options, and each new column gets them. You can also set
options for named columns.

## Usage

Write a stream of nd record batches, then read a projection back as a stream:

```rust
use futures::{StreamExt, stream};
use lance_nd::{ColumnOptions, Compression, WriteOptions, scan, write};

let options = WriteOptions {
    column_defaults: Some(
        ColumnOptions::default().with_compression(Compression::Zstd { level: Some(3) }),
    ),
    ..WriteOptions::default()
};

// `batches` is an iterator of nd_arrow_array::NdRecordBatch.
let rows = stream::iter(batches.map(Ok));
let table = write("data/arrays.lance", rows, &options).await?;

// Read two arrays of one row. Each item is one NdRecordBatch.
let mut rows = scan(&table)
    .project(&["temperature", "x"])
    .filter("__global_attrs__.id = 'a'")
    .try_into_stream()
    .await?;
while let Some(batch) = rows.next().await {
    let batch = batch?;
    println!("{:?}", batch.target());
}
```

The main functions:

| Function | Purpose |
|---|---|
| `write(uri, rows, &options)` | Create, append to or overwrite a table from a stream of nd record batches. |
| `append(&mut table, rows, &options)` | Append a stream of nd record batches to an open table. |
| `NdWriter::create` and `NdWriter::append` | Write nd record batches in several calls, then commit all of them in one version with `finish`. `write_batch` takes rows that are already encoded. |
| `scan(&table)` | Read nd record batches as a stream, with `project`, `filter`, `filter_expr`, `lance_filter` and `limit`. |
| `variables(&table)` | List the array columns of the table. |
| `column_defaults` and `set_column_defaults` | Read and change the encoding defaults for new columns. |
| `delete` and `delete_expr` | Delete the rows that match an exact predicate. |
| `compact` and `cleanup_old_versions` | Merge small fragments, remove deleted rows and free disk space. |

`project` takes whole variables, such as `temperature`, not nested fields
such as `temperature.attrs.units`. Each row keeps all of its global
attributes. An empty projection reads only the global attributes, one row
without arrays for each table row. A filter can use nested fields, see
[Filters](#filters).

`project` takes whole variables, such as `temperature`, not nested fields
such as `temperature.attrs.units`. Each row keeps all of its global
attributes. An empty projection reads only the global attributes, one row
without arrays for each table row. A filter can use nested fields, see
[Filters](#filters).

Rows can be large, so a scan reads Lance batches of at most 256 rows or about
32 MiB, and it decodes them in parallel. `batch_size`, `batch_size_bytes`,
`batch_readahead` and `decode_parallelism` change these limits.

`NdWriter` buffers rows and writes them as Lance fragments of about 128 MiB
in memory, with up to 4 fragment writes at the same time. A larger row gets
a fragment of its own. `WriteOptions::target_fragment_bytes` and
`WriteOptions::write_parallelism` change these limits. The writer must run
on a Tokio runtime.

Scans and writes allocate many large buffers. On Windows, the system
allocator makes them much slower. Set a faster global allocator in your
application, for example [mimalloc](https://crates.io/crates/mimalloc):

```rust
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

### Delete and compact

Delete rows with an exact predicate on the table columns, then compact the
table and remove the files of old versions:

```rust
use std::time::Duration;
use lance_nd::{CompactOptions, cleanup_old_versions, compact, delete};

let deleted = delete(&mut table, "__global_attrs__.id = 'a'").await?;
compact(&mut table, &CompactOptions::default()).await?;
cleanup_old_versions(&table, Duration::from_secs(7 * 24 * 60 * 60)).await?;
```

- `delete` takes Lance SQL, and `delete_expr` takes a DataFusion `Expr`.
  Lance gets the predicate with no change. A delete does not use the stats,
  because a stats filter keeps rows that do not match. A filter on values
  must be exact, for example `array_has(sst.values, 5.0)`.
- A delete marks rows in deletion files. It does not rewrite data files.
- `compact` merges fragments smaller than about 128 MiB on disk, and writes
  again each fragment with at least 10% deleted rows. `CompactOptions`
  changes these limits. lance-nd holds one group of fragments in memory at
  a time. The row order stays.
- Each delete and compaction makes a new table version. The files of old
  versions stay on disk until `cleanup_old_versions` removes them. A reader
  that still uses a removed version fails, so keep a margin.

lance-nd writes the compacted fragments itself. Lance reads the arrays of
an absent variable as null, but it rejects these nulls in a write, so its
own compaction fails on a lance-nd table.

## Python

The `lance_nd` Python package encodes xarray datasets and writes them to a
lance-nd table. It commits all datasets of one writer in one version. It
reads the table back as a stream of nd-arrow record batches:

```python
import xarray as xr
import lance_nd

with lance_nd.create("data/arrays.lance", stats=True) as writer:
    writer.write(xr.open_dataset("obs.nc"))

table = lance_nd.open("data/arrays.lance")
for batch in table.scan(project=["sst"], prune="sst.values > 300"):
    ...
```

See [python/README.md](python/README.md) for installation and the encoding
rules.

## Status

lance-nd is experimental. Version 1 stores and reads rows. It has no indexes.

- [FORMAT.md](FORMAT.md) is the specification of the lance-nd table.
- The [nd-arrow FORMAT.md](https://github.com/robinskil/nd-arrow-array/blob/main/FORMAT.md)
  is the specification of the nd-arrow encoding.
