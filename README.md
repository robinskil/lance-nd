# lance-nd

N-dimensional datasets in [Lance](https://lancedb.github.io/lance/).

Each row of a lance-nd table is one
[`NdRecordBatch`](https://github.com/robinskil/nd-arrow-array): a set of
variables on one grid, like an xarray `Dataset` or a chunk of one.
The Lance schema is the nd-arrow encoding: one `nd_arrow.array` column per
variable and an optional `__global_attrs__` column.

## API

- `write(uri, rows, &options)`: write a stream of nd record batches.
- `append(&mut dataset, rows, &options)`: append a stream to an open table.
- `scan(&dataset)`: read a stream of nd record batches. Set `project`,
  `filter` and `limit` to select data. Set `batch_size` (default 256 rows),
  `batch_size_bytes` (default 32 MiB), `batch_readahead` and
  `decode_parallelism` to tune the read.
- `variables(&dataset)`: list the variables of the table.
- `column_defaults(&dataset)` and `set_column_defaults(&mut dataset, &options)`:
  read and change the encoding defaults for new columns.

Rows can have different variables. A row without a variable stores a null.
New variables and attribute keys extend the table schema.

## Column encoding

`ColumnOptions` sets the Lance encoding of a column: compression,
compression level, byte stream split, run-length encoding threshold and
structural encoding. `WriteOptions::column_defaults` sets the defaults of a
new table. The table stores them, and each column that a later write adds
gets them. `WriteOptions::column_options` sets options for named columns.

```rust
use lance_nd::{ColumnOptions, Compression, StructuralEncoding, WriteOptions};

let options = WriteOptions {
    column_defaults: Some(
        ColumnOptions::default().with_compression(Compression::Zstd { level: Some(3) }),
    ),
    column_options: [(
        "sst".to_string(),
        ColumnOptions::default().with_structural_encoding(StructuralEncoding::FullZip),
    )]
    .into(),
    ..WriteOptions::default()
};
```

## Example

```rust
use futures::{StreamExt, stream};
use lance_nd::{WriteOptions, scan, write};

// `datasets` is an iterator of nd_arrow_array::NdRecordBatch.
let rows = stream::iter(datasets.map(Ok));
let table = write("data/ocean.lance", rows, &WriteOptions::default()).await?;

// Read two variables, one row at a time.
let mut rows = scan(&table)
    .project(&["sst", "lat"])
    .filter("__global_attrs__.id = 'sst.nc'")
    .try_into_stream()
    .await?;
while let Some(batch) = rows.next().await {
    let batch = batch?;
    println!("{:?}", batch.target());
}
```

## Scale test

`tests/scale.rs` writes 10K datasets with 20 of 2,048 variables each, then
reads 20 variables from all rows. Run it in release mode:

```sh
cargo test --release --test scale -- --ignored --nocapture
```

## Status

Version 1 stores and reads rows. It has no indexes.

See [FORMAT.md](FORMAT.md) for the table layout.
