# lance_nd for Python

Store xarray datasets as N-dimensional Arrow arrays in a
[lance-nd](../README.md) table.

The package does four things:

1. It encodes an `xarray.Dataset` as a PyArrow record batch in the
   [nd-arrow format](https://github.com/robinskil/nd-arrow-array/blob/main/FORMAT.md).
2. It passes the record batch to Rust.
3. Rust writes the datasets to a lance-nd table, and commits all of them in
   one version when you call `finish()`.
4. It reads the table back as a stream of nd-arrow record batches.

## Install

Build the package from this directory with [maturin](https://www.maturin.rs/):

```bash
python -m venv .venv
.venv/Scripts/python -m pip install maturin pytest numpy pyarrow xarray
.venv/Scripts/python -m maturin develop --release
```

Without `--release`, maturin builds the Rust code without optimizations,
and writes are several times slower.

On Linux and macOS, use `.venv/bin/python`.

## Usage

```python
import xarray as xr
import lance_nd

# Create a table. Readers see nothing before finish().
writer = lance_nd.create("data.lance")
writer.write(ds_a)                       # an xarray.Dataset
writer.write([ds_b, ds_c])               # a list of datasets
writer.write(lance_nd.encode(ds_d))      # an nd-arrow pyarrow.RecordBatch
table = writer.finish()                  # one commit

# Append to the table.
with lance_nd.append("data.lance") as writer:
    writer.write(ds_e)                   # finish at the end of the block

table = lance_nd.open("data.lance")
table.version, table.count_rows(), table.variables()
```

| Function | Purpose |
|---|---|
| `encode(datasets, chunks=None)` | Encode one dataset or a list of datasets. Each dataset is one row, or each dask chunk with `chunks`. |
| `encode_chunks(ds, chunks)` | Encode one dask chunk at a time, one record batch per chunk. |
| `create(uri, mode="create")` | A writer for a new table. `mode="overwrite"` replaces the table at `finish()`. |
| `append(uri)` | A writer for an existing table. |
| `open(uri)` | The table at its current version. |
| `Writer.write(data, chunks=None)` | Write datasets as data files, without a commit. With `chunks`, one chunk at a time. |
| `Writer.finish()` | Commit all written datasets in one version. |
| `Writer.abort()` | Drop all written datasets. |
| `Table.scan(project=None, filter=None, prune=None, limit=None)` | Read the datasets as a stream of nd-arrow record batches. |
| `Table.delete(predicate)`, `Table.compact()` and `Table.cleanup_old_versions()` | Delete datasets and free disk space. |

Rules:

- A writer that you abort, or that you do not finish, commits nothing.
- As a context manager, the writer finishes at the end of the block. When
  the block raises, the writer aborts. If you call `finish()` in the block,
  the block does not finish again. `writer.table` holds the committed table.
- `finish()` on a new table without datasets raises an error. On an append
  without datasets, it makes no commit.
- A `write()` that fails a check, for example a variable or an attribute
  with another type than in the table, raises `LanceNdError` and writes none
  of its datasets. `create()` and `append()` take `casting` to cast such
  types instead, see [Casting](#casting). The writer stays usable, and `finish()` commits the other
  datasets. A failed data file write makes the writer fail, and `finish()`
  then commits nothing.
- If another writer adds variables to the table before your `finish()`, and
  your writer also adds variables, `finish()` raises an error.
- Errors from Rust raise `lance_nd.LanceNdError`.

### Chunked datasets

A dataset that you open with dask chunks can give one row per chunk. Each
row is a complete dataset, so a reader can load one chunk at a time.

```python
ds = xr.open_dataset("ocean.nc", chunks={"time": 24, "depth": 100})

with lance_nd.create("ocean.lance") as writer:
    writer.write(ds, chunks=["time", "depth"], on_drop="warn")

for batch in lance_nd.encode_chunks(ds, chunks=["time", "depth"]):
    ...                                  # one record batch per chunk
```

- `chunks` names the chunk dimensions. Each row is one dask chunk over
  these dimensions, in C order with the first name as the outermost.
- A variable is kept when all its dimensions are chunk dimensions, for
  example `(time)`, `(time, depth)` or a scalar. A variable is also kept
  when it has all chunk dimensions, for example `(time, depth, position)`.
  Its other dimensions keep their full size in each row.
- All other variables are dropped, also coordinates such as
  `position (position)`. `on_drop` sets what happens: `"warn"` (default)
  gives a `UserWarning`, `"raise"` gives a `ValueError`, and `"ignore"`
  drops them silently.
- Kept variables keep their attributes. Each row keeps the global
  attributes.
- `writer.write(..., chunks=...)` and `encode_chunks` compute one chunk at
  a time. `encode(ds, chunks=...)` returns all chunks in one record batch,
  so it loads all data.

### Encoding options

`create()` takes `column_defaults` and `column_options`, and `append()`
takes `column_options`. Each option set is a dict:

| Key | Values |
|---|---|
| `compression` | `"none"`, `"lz4"`, `"zstd"` |
| `compression_level` | an integer, with `"zstd"` only |
| `byte_stream_split` | `"off"`, `"on"`, `"auto"` |
| `rle_threshold` | a float |
| `structural_encoding` | `"miniblock"`, `"fullzip"` |

```python
lance_nd.create(
    "data.lance",
    column_defaults={"compression": "zstd", "compression_level": 3},
    column_options={"sst": {"byte_stream_split": "on"}},
)
```

### Stats

`create(uri, stats=True)` stores the min and the max of each variable per
row, in one column `__stats__:<variable>` for each variable. A scan with
`prune` uses them to skip rows. An append follows the table.
`Table.has_stats` tells if a table stores stats. A table from an earlier
lance-nd release reads as a table without stats; write it again to get
stats. Each variable name that starts with `__stats__` is reserved.

### Read

`Table.scan()` reads the datasets as a stream. Each item is a
`pyarrow.RecordBatch` of one row in the nd-arrow format: one nd column per
variable of the dataset, and the `__global_attrs__` column. The scan reads
only a few Lance batches ahead, so memory stays small for a large table.

```python
table = lance_nd.open("data/obs.lance")
for batch in table.scan(
    project=["sst"],
    filter="__global_attrs__.title = 'run 1'",
    prune="sst.values > 300",
):
    values = batch.column("sst")[0]["values"]
```

- `project` takes whole variable names. Each batch keeps the global
  attributes, so `project=[]` reads only the global attributes. A nested
  field, such as `sst.attrs.units`, raises an error.
- `filter` is an exact Lance SQL filter, for example on
  `__global_attrs__.title` or `sst.attrs.units`.
- `prune` is a filter on values, for example `sst.values > 300`. It uses the
  stats to skip datasets that surely do not match. A kept dataset comes back
  whole, so it can hold values outside the filter. Without stats, it keeps
  all datasets. `filter` and `prune` can be used together.
- `limit`, `batch_size` and `batch_size_bytes` limit the rows and the Lance
  batches that the scan reads.

### Casting

`create()` and `append()` take `casting`, as the numpy `casting`
argument. It decides what happens when a variable or an attribute has
another type than the table:

- `"no"` (default): the write raises `LanceNdError`.
- `"safe"`: the writer casts the new values when no value can change, for
  example float32 to float64, or int16 to float64.
- `"same_kind"`: the writer also casts within a kind, for example float64
  to float32. An integer that does not fit raises `LanceNdError`.
- `"promote"`: as numpy type promotion. When the new values do not fit the
  table type, the writer widens the variable in the table to the smallest
  type that holds both, for example int32 and float32 to float64. A writer
  first commits the datasets that it wrote before such a widening.

With `"no"`, `"safe"` and `"same_kind"`, the table type always wins, so the
first dataset with a variable sets its type. `Table.widen("sst",
pa.float64())` widens a variable of an existing table. It writes that
variable and its stats column again, not the whole table.

```python
with lance_nd.create("data.lance", casting="safe") as writer:
    writer.write(ds_float64)
    writer.write(ds_float32)   # sst becomes float64
```

### Fragments

A writer buffers datasets and writes them as Lance fragments of about
128 MiB on disk, with up to 4 fragment writes at the same time. The writer
estimates the size on disk from the fragments it wrote, so the sizes are
not exact. `Table.compact()` gives exact sizes.
`create()` and `append()` take `target_fragment_bytes` and
`write_parallelism` to change these limits. `Table.num_fragments` gives
the number of fragments.

### Slow NetCDF-4 files on Windows

xarray asks `netCDF4` for the compression filters of each variable of a
NetCDF-4 file. `netCDF4` then searches the HDF5 filter plugins on disk, and
on Windows this can take about 30 ms per variable. A file with 100 variables
then needs several seconds to open. When your files use only the filters
that HDF5 has built in, such as zlib, set `HDF5_PLUGIN_PATH` to an empty
folder before you import xarray:

```python
import os

os.environ["HDF5_PLUGIN_PATH"] = r"C:\empty-folder"

import xarray as xr
```

A file that needs a plugin filter, such as zstd or blosc, can then fail
to read. Classic NetCDF-3 files are not affected.

### Delete and compact

```python
from datetime import timedelta

table = lance_nd.open("data/obs.lance")
table.delete("__global_attrs__.title = 'old run'")
table.compact()
table.cleanup_old_versions(timedelta(days=7))
```

- `delete()` takes an exact Lance SQL filter and returns the number of
  deleted datasets.
- `compact()` merges small fragments, splits large fragments and removes
  deleted datasets. It takes `target_fragment_bytes` (default 128 MiB on
  disk) and `deleted_fraction` (default 0.1). The new fragments are about
  the same size, near the target.
- `cleanup_old_versions()` removes the files of versions older than
  `older_than` (default 14 days). Until then, deletes and compactions free
  no disk space.

### Type hints

The package has type hints and a `py.typed` marker. The stub
`_lance_nd.pyi` types the native classes. Use these types in your own
annotations:

| Type | Values |
|---|---|
| `lance_nd.ColumnOptions` | the option set of one column, a `TypedDict` |
| `lance_nd.Mode` | `"create"`, `"overwrite"` |
| `lance_nd.OnDrop` | `"warn"`, `"raise"`, `"ignore"` |

Run `mypy` in `python/` to check the package in strict mode.

## Encoding rules

| xarray | nd-arrow |
|---|---|
| One `Dataset` | One row |
| Each data variable and coordinate | One `nd_arrow.array` column |
| `Variable.values` | `values`, flat in C order |
| `Variable.dims` and `Variable.shape` | `dim_names` and `dim_sizes` |
| `Variable.attrs` | `attrs` |
| `Dataset.attrs` | the `__global_attrs__` column |
| `Variable.encoding`, `Dataset.encoding` | not written |

Value types:

| NumPy dtype | Arrow type |
|---|---|
| `bool` | `Boolean` |
| integers | the same integer type |
| `float16`, `float32`, `float64` | the same float type |
| `datetime64[unit]` | `Timestamp(unit)` |
| `timedelta64[unit]` | `Duration(unit)` |
| `str`, `bytes`, `object` with strings | `Utf8` |
| `object` with cftime values, with `convert_calendar=True` | `Timestamp(ns)` |
| other dtypes | `TypeError` |

Missing values: NaN, NaT and `None` become nulls. Big-endian data and
arrays in Fortran order are converted. Dask arrays are loaded into memory.

Calendars: xarray decodes a calendar that `datetime64` cannot hold, such as
a Julian one, into cftime objects. `encode()`, `encode_chunks()` and
`Writer.write()` take `convert_calendar=True` to store them as
`datetime64[ns]` values of the same instant. The calendar label changes: a
Julian 1973-02-25 becomes the Gregorian 1973-03-10. A calendar without real
instants, such as `360_day` or `noleap`, and a date outside 1677 to 2262
raise a `ValueError`. Without the option, cftime values raise a
`TypeError`. To keep the raw numbers with their `units` and `calendar`
attributes instead, open the files with `decode_times=False`.

Attribute types: `str`, `bool`, `int` (`Int64`), `float` (`Float64`),
NumPy scalars, and 1-D arrays, lists or tuples of one scalar type. Other
values raise a `TypeError`.

Coordinates: each data variable gets a CF `coordinates` attribute with the
names of its other coordinates, for example `"lat2d lon2d"`. A coordinate
with the name of its only dimension, such as `lat (lat)`, is not listed.
Coordinates that no data variable lists go into a global `coordinates`
attribute. A dataset or variable that already has a `coordinates`
attribute raises a `ValueError`.

In a list of datasets, one variable must have one value type, and one
attribute key must have one type. An attribute that is an empty array, such
as `[]`, holds no value, so the encoder leaves it out. One `write()` call holds at most
2,147,483,647 values per variable over all its datasets. For more, write
fewer datasets per call.

## Tests

```bash
.venv/Scripts/python -m pytest tests
```
