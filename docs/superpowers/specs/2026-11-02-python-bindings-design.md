# Python bindings for lance-nd: design

Date: 2026-11-02
Status: draft, waiting for approval

## 1. Intent

Users have their data in xarray. They want to store it in a lance-nd table
from Python, with no manual Arrow work.

The Python package does three things:

1. It encodes an `xarray.Dataset` as a PyArrow record batch in the nd-arrow
   format.
2. It passes that record batch to Rust through PyO3.
3. Rust writes the datasets to a lance-nd table, and commits them all at once
   when the user calls `finish()`.

## 2. Goals

- Create a new lance-nd table from Python.
- Open an existing lance-nd table from Python and append datasets to it.
- Write one or more datasets per call. A dataset is an `xarray.Dataset` or an
  nd-arrow `pyarrow.RecordBatch`.
- Commit all datasets of one writer in one Lance commit at `finish()`.
  Readers see no dataset of the writer before `finish()`, and all of them
  after it.
- Include the encoder from `xarray.Dataset` to nd-arrow PyArrow in the
  package.

## 3. Non-goals

- Reading data back from Python into xarray or PyArrow. (A later version.)
- Chunks of datasets. Each dataset is one row.
- Lazy or dask-based streaming. Dask arrays are loaded into memory before the
  encode.
- An async Python API.
- Publishing wheels to PyPI.

## 4. Anti-goals

- Any data visible to readers before `finish()`. A failed or abandoned writer
  must leave the table as it was.
- An encoder that writes data that the Rust nd-arrow reader rejects. The
  Rust crate `nd-arrow-array` is the reference implementation of the format.
- Silent data changes, for example a type cast that loses precision, or an
  attribute that is dropped with no error.

## 5. Python API

```python
import xarray as xr
import lance_nd

# Encode only.
batch = lance_nd.encode(ds)              # pyarrow.RecordBatch with 1 row
batch = lance_nd.encode([ds_a, ds_b])    # pyarrow.RecordBatch with 2 rows

# Create a new table.
writer = lance_nd.create("data.lance")   # fails if the table exists
writer.write(ds_a)                       # xarray.Dataset
writer.write([ds_b, ds_c])               # list of xarray.Dataset
writer.write(batch)                      # nd-arrow pyarrow.RecordBatch
table = writer.finish()                  # one commit

# Append to an existing table.
writer = lance_nd.append("data.lance")   # fails if the table does not exist
writer.write(ds_d)
table = writer.finish()

# Context manager: finish on success, abort on an exception.
with lance_nd.create("data.lance") as writer:
    writer.write(ds_a)

# Minimal table information.
table = lance_nd.open("data.lance")
table.uri          # str
table.version      # int, the Lance version
table.count_rows() # int
table.variables()  # list[str]
```

Rules:

- `create(uri, mode="create")` accepts `mode="overwrite"`. Overwrite replaces
  the table at `finish()`, in the same single commit.
- `create()` accepts the encoding options that lance-nd already has: column
  defaults and options per column (compression, compression level, byte
  stream split, run-length encoding, structural encoding). `append()` accepts
  options per column for new columns.
- `writer.abort()` drops all uncommitted work. After `finish()` or `abort()`,
  each call on the writer raises an error.
- A writer that is garbage collected without `finish()` commits nothing.
- `finish()` with no written datasets: `create()` raises an error, because a
  table needs at least one row (the same as the existing `write()`).
  `mode="overwrite"` also raises an error. `append()` makes no commit and
  returns the table at its current version. A `with` block that writes
  nothing follows the same rules. (Builder's call.)
- `uri` is a local path, or a URI that Lance opens with no extra options.
  Storage options for object stores are not in this version. They can be
  added later as a keyword argument. (Builder's call.)
- The calls block. Rust releases the GIL while it encodes, writes and
  commits.
- Errors raise Python exceptions. Encoder errors raise `ValueError` or
  `TypeError` with the variable or attribute name. Rust errors raise
  `lance_nd.LanceNdError`.

## 6. Encoder: xarray to nd-arrow

The encoder follows the nd-arrow format specification, version 1:
`FORMAT.md` in https://github.com/robinskil/nd-arrow-array at the commit in
`Cargo.lock`. The rules below map xarray onto that format.

### 6.1 Structure

| xarray | nd-arrow |
|---|---|
| One `Dataset` | One row of the record batch |
| Each data variable and each coordinate | One `nd_arrow.array` column, with the variable name as the column name |
| `Variable.values` | `values`, flat in C order |
| `Variable.dims` | `dim_names` |
| `Variable.shape` | `dim_sizes` (`UInt32`) |
| `Variable.attrs` | `attrs` struct, left out when no row has attributes |
| `Dataset.attrs` | `__global_attrs__` column, left out when no dataset has global attributes |

- Each column field has `ARROW:extension:name = nd_arrow.array` and
  `ARROW:extension:metadata = {"version":1}`.
- The global attributes field has
  `ARROW:extension:name = nd_arrow.global_attributes` and the same metadata.
- For a list of datasets, a variable that a dataset does not have is a null
  row in that column.
- Coordinates without data (an index with no coordinate variable) are not
  written.
- The encoder records which variables are coordinates with the CF
  `coordinates` attribute, as xarray does in `to_netcdf`:
  - A coordinate with the same name as its single dimension, such as
    `lat (lat)`, is a coordinate by the nd-arrow rule. It is not listed.
  - Each data variable gets a `coordinates` attribute with the names of the
    other coordinates whose dimensions are all in the dimensions of that
    variable, separated by spaces, for example `"lat2d lon2d"`.
  - Other coordinates that no data variable lists, such as a scalar `time`
    in some cases, go into a global attribute `coordinates` in
    `__global_attrs__`.
  - If a variable or the dataset already has a `coordinates` attribute in
    `attrs`, the encoder raises a `ValueError`, as xarray does.
  - A later reader uses these attributes to rebuild the same coordinates.
- A variable named `__global_attrs__` is an error.

### 6.2 Value types

| NumPy dtype | Arrow type |
|---|---|
| `bool` | `Boolean` |
| `int8` to `int64`, `uint8` to `uint64` | the same integer type |
| `float16`, `float32`, `float64` | the same float type |
| `datetime64[unit]` | `Timestamp(unit)`, no time zone |
| `timedelta64[unit]` | `Duration(unit)` |
| `str` (`U`), `bytes` (`S`), `object` with only `str` | `Utf8` |
| any other dtype | `TypeError` |

Missing values:

- NaN in a float array becomes a null item. xarray uses NaN for missing
  values after it decodes `_FillValue`.
- NaT in a `datetime64` or `timedelta64` array becomes a null item.
- `None` in an `object` string array becomes a null item.

For a list of datasets, one variable must have one value type in all rows.
Two types for one variable are a `TypeError`.

### 6.3 Attribute types

| Python or NumPy value | Arrow type |
|---|---|
| `str` | `Utf8` |
| `bool`, `numpy.bool_` | `Boolean` |
| `int` | `Int64` |
| `float` | `Float64` |
| NumPy scalar | its Arrow type from section 6.2 |
| 1-D NumPy array, `list` or `tuple` of one scalar type | `List` of that type |
| any other value | `TypeError` with the attribute name |

One attribute key has one type in a column (format section 5). In a list of
datasets, two types for one key are a `TypeError`.

### 6.4 `encoding`

The encoder ignores `Variable.encoding` and `Dataset.encoding`. They describe
how the source file stores the data, for example `_FillValue`,
`scale_factor` or the `units` of CF time. The encoder writes the decoded
values, so these keys no longer apply. Only `attrs` are written. (Builder's
call.)

### 6.5 Memory

The encoder copies each variable once into its Arrow buffer. It must not keep
other copies of the data.

## 7. Rust side

### 7.1 Crate

- A new crate in this repository at `python/`, built with maturin and PyO3.
  It depends on lance-nd with a path dependency.
- The Python package name and import name are `lance_nd`.
- The encoder (section 6) is pure Python, with NumPy, PyArrow and xarray, in
  the same package.
- The record batch goes from PyArrow to Rust through the Arrow C data
  interface. Rust does not copy the data to receive it.
- Rust turns the record batch into rows with `RowDecoder` from
  `lance_nd::encode`. A null row in a column means that the dataset does not
  have that variable, which matches the table format of lance-nd. Do not use
  `NdRecordBatch::try_from_record_batch` here: it decodes a null row as a
  rank-0 null scalar, which would store a variable that is absent.
  (Builder's call.)

### 7.2 Staged writer in lance-nd

The single commit at `finish()` needs a new writer in the lance-nd crate, so
that Rust users get it too:

```rust
let mut writer = NdWriter::create(uri, &options).await?;   // or NdWriter::append
writer.write(rows).await?;      // writes data files, no commit
writer.write(more_rows).await?;
let dataset = writer.finish().await?;  // one commit
```

- `write()` writes Lance data files at once and keeps only their metadata in
  memory. Lance 11 has `InsertBuilder::execute_uncommitted` and
  `CommitBuilder::execute` for this.
- `finish()` builds one transaction with all staged fragments and commits it.
- `abort()`, or a drop without `finish()`, commits nothing. The data files
  stay as orphans until Lance cleanup removes them.
- The existing `write()` and `append()` functions stay as they are.

**Risk: schema changes across writes.** A later `write()` can bring new
variables or new attribute keys. The table schema must then grow before the
commit, and the fragments of earlier writes do not have the new fields. The
current `append()` uses `add_columns` for this, but `add_columns` commits on
its own. The first step of the plan is a spike that checks how one Lance 11
commit can hold fragments with different field sets under one merged schema.
If no way exists, the builder stops and brings the options back before more
work.

### 7.3 Runtime

- The crate keeps one Tokio runtime for all calls.
- Each call releases the GIL with `Python::allow_threads` while it runs.

### 7.4 Concurrent writers

If another writer commits to the table between `append()` and `finish()`,
Lance resolves the conflict for plain appends. If this writer also grows the
schema, the commit can conflict. Then `finish()` raises `LanceNdError` and
commits nothing. The user writes again. `finish()` does not retry or rebase
staged fragments onto a new schema. (Builder's call.) The spike in section
7.2 also checks which of these cases Lance 11 resolves.

## 8. Tests

- Encoder tests in Python (pytest): each row of the type tables in section 6
  gives the expected Arrow type, and the Rust side accepts the batch.
- Round trip in Rust tests: an nd-arrow batch from the encoder, written and
  read back with `lance_nd::scan`, gives the same values, dimensions and
  attributes.
- Writer tests in Python:
  - before `finish()`, `open()` fails for a new table, or shows the old
    version for an appended table;
  - after `finish()`, the table has all rows in one new version;
  - after `abort()` or an exception in a `with` block, the table is
    unchanged;
  - a second dataset with a new variable works in one writer.
- The tests run on Windows and Linux.

## 9. Left to the builder

- The PyO3 and maturin versions, and the bridge for the Arrow C data
  interface (the `pyarrow` feature of the `arrow` crate or the `pyo3-arrow`
  crate). The PyO3 version must match the bridge.
- The minimum Python version and whether the wheel uses the stable ABI.
- How Python passes encoding options: keyword arguments, dicts or small
  classes.
- The module layout inside the Python package.
- How many rows the staged writer puts in one data file.
