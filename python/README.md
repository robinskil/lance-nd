# lance_nd for Python

[![PyPI](https://img.shields.io/pypi/v/lance_nd)](https://pypi.org/project/lance_nd/)
[![CI](https://github.com/robinskil/lance-nd/actions/workflows/ci.yml/badge.svg)](https://github.com/robinskil/lance-nd/actions/workflows/ci.yml)
[![Status: experimental](https://img.shields.io/badge/status-experimental-orange)](https://github.com/robinskil/lance-nd)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/robinskil/lance-nd/blob/main/LICENSE)

Store xarray datasets as N-dimensional Arrow arrays in a
[lance-nd](https://github.com/robinskil/lance-nd) table.

The package does three things:

1. It converts each `xarray.Dataset` to an `NdRecordBatch` of the
   [nd-arrow-array](https://pypi.org/project/nd-arrow-array/) package.
2. Rust writes the datasets to a lance-nd table, and commits all of them in
   one version when you call `finish()`.
3. It reads the table back as a stream of `NdRecordBatch` objects. Each
   object gives its dataset with `to_xarray()`.

## Install

Install the package from [PyPI](https://pypi.org/project/lance_nd/):

```bash
pip install lance_nd
```

The package needs Python 3.10 or later.

## Build from source

Build the package from this directory with [maturin](https://www.maturin.rs/):

```bash
python -m venv .venv
.venv/bin/python -m pip install maturin
.venv/bin/python -m maturin develop --release --extras dev
```

On Windows, use `.venv/Scripts/python`. The `dev` extra installs the test
and type check tools.

Without `--release`, maturin builds the Rust code without optimizations,
and writes are several times slower.

## Usage

```python
import xarray as xr
import lance_nd

# Create a table. Readers see nothing before finish().
with lance_nd.create("data.lance", stats=True) as writer:
    writer.write(ds_a)                   # an xarray.Dataset
    writer.write([ds_b, ds_c])           # a list of datasets

# Append dask chunks, one row per chunk.
with lance_nd.append("data.lance") as writer:
    writer.write(ds_d, chunks=["time"])

table = lance_nd.open("data.lance")
for batch in table.scan(filter="__global_attrs__.title = 'a'", prune="sst.values > 300"):
    ds = batch.to_xarray()               # an nd_arrow_array.NdRecordBatch
```

## Command line

The `lance-nd` command writes files that xarray can open to a table. Each
file is one row with all its variables. The command sorts the files and
commits all rows in one version:

```bash
lance-nd create "data/**/*.nc" data.lance --stats
```

```bash
lance-nd append "new/*.nc" data.lance --skip-errors --log-file append.log
```

Use these options for files that the table cannot store:

- `--skip-variables` drops each variable that cannot be stored, such as a
  complex variable.
- `--skip-attributes` drops each attribute that cannot be stored, such as a
  dict.
- `--skip-errors` skips a file that fails and continues with the next file.
  Without it, the first error stops the command and it commits nothing.
- `--casting` sets the casting rule of the writer, see `create()`.

The command adds three global attributes to each row: `source_file` is the
file path with `/` separators, `source_mtime` is the modification time in
nanoseconds and `source_size` is the size in bytes. Use `source_file` to
select or delete the rows of a file:

```python
lance_nd.open("data.lance").delete("__global_attrs__.source_file = 'data/2024/a.nc'")
```

`--file-attribute NAME` sets another name for `source_file`, and
`--no-file-attribute` adds none of the three. A file attribute with the same
name gets the new value.

### Sync

`lance-nd sync` mirrors the files as the rows of a table. It adds a row for
each new file, writes a changed file again, and deletes the rows of deleted
files. A file is changed when its modification time or size is not the same
as in its row. The first sync creates the table:

```bash
lance-nd sync "data/**/*.nc" data.lance --interval 600 --min-age 60 --log-file sync.log
```

- `--interval SECONDS` syncs again after each interval, until you press
  Ctrl+C. Without it, the command syncs one time.
- `--min-age SECONDS` skips files that changed less than this time ago, so
  that a file that is still being written waits for the next sync.
- `--no-compact` turns off the compaction after each sync that changes the
  table.

A sync appends the new rows in one commit, then deletes the old rows in a
second commit. Between the two commits, a reader can see two rows for a
changed file. A file that fails keeps its old row, and the next sync tries
it again. When the pattern matches no file, the sync deletes nothing,
because a missing mount gives the same result as an empty folder. Use the
same pattern form for each sync: a relative and an absolute pattern give
different `source_file` values. The sync does not remove old table
versions, see `Table.cleanup_old_versions()`.

The log names each skipped file, variable and attribute. Run
`lance-nd create --help` or `lance-nd sync --help` for all options. `python -m lance_nd` is the same
command. The package installs the `netCDF4` and `h5netcdf` engines, so
xarray can open netCDF 3 and netCDF 4 files.

The [Python wiki page](https://github.com/robinskil/lance-nd/wiki/Python)
explains the writer rules, chunks, encoding options, stats, scans, casting,
object stores and the encoding rules.

## Tests

Build from source first. Then run the tests:

```bash
.venv/bin/python -m pytest tests
```

`tests/test_s3.py` runs when `LANCE_ND_S3_ENDPOINT` points to an S3 server
with the bucket `lance-nd-test`, see the [Object stores](https://github.com/robinskil/lance-nd/wiki/Object-Stores) wiki page.
