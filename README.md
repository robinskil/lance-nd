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
  <a href="#the-nd-arrow-encoding">nd-arrow encoding</a> ·
  <a href="#the-lance-nd-table">lance-nd table</a> ·
  <a href="#usage">Usage</a> ·
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

## The nd-arrow encoding

### nd arrow arrays

An nd arrow array is one N-dimensional array. As an Arrow column, it is the
extension type `nd_arrow.array` over a struct, and each row holds one complete
nd arrow array:

```text
Struct {
  values:    List<T>,       // the flat values, in C order
  dim_names: List<Utf8>,    // the name of each dimension
  dim_sizes: List<UInt32>,  // the size of each dimension
  attrs:     Struct { .. }, // optional attributes, such as units
}
```

For example, an `(x=2, y=3)` array of `Float64` with `units = "K"`:

```text
values:    [0.0, 0.1, 0.2, 1.0, 1.1, 1.2]
dim_names: ["x", "y"]
dim_sizes: [2, 3]
attrs:     { units: "K" }
```

The element at `(x=1, y=2)` is at flat position `1 * 3 + 2 = 5`, so its value
is `1.2`.

Rules:

- The rows of one column can have different shapes and dimensions.
- An nd arrow array with no dimensions is a scalar with one value.
- Nulls in the `values` items mark missing elements. The format has no fill
  value and no separate mask.
- `attrs` has one child field per attribute key. A null child means that the
  array does not have that key. One key has one type in a column.

### nd record batches

An nd record batch is a set of named nd arrow arrays on one shared grid, plus
optional global attributes. As Arrow data, it is a record batch of nd columns:

- Each nd column holds one named nd arrow array, for example `x`, `y` or
  `temperature`.
- The arrays broadcast onto one grid. An array can use a subset of the grid
  dimensions: a coordinate array uses one, and a scalar uses none.
- The format does not store the grid. A reader infers it from the arrays.
- Global attributes describe the whole nd record batch. They are in one extra
  struct column, `__global_attrs__`.

In Rust, these are `NdArrowArray` and `NdRecordBatch` from
[nd-arrow-array](https://github.com/robinskil/nd-arrow-array).

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
| `scan(&table)` | Read nd record batches as a stream, with `project`, `filter` and `limit`. |
| `variables(&table)` | List the array columns of the table. |
| `column_defaults` and `set_column_defaults` | Read and change the encoding defaults for new columns. |

Rows can be large, so a scan reads Lance batches of at most 256 rows or about
32 MiB, and it decodes them in parallel. `batch_size`, `batch_size_bytes`,
`batch_readahead` and `decode_parallelism` change these limits.

## Status

lance-nd is experimental. Version 1 stores and reads rows. It has no indexes.

- [FORMAT.md](FORMAT.md) is the specification of the lance-nd table.
- The [nd-arrow FORMAT.md](https://github.com/robinskil/nd-arrow-array/blob/main/FORMAT.md)
  is the specification of the nd-arrow encoding.
