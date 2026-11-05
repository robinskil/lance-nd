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

The [Python wiki page](https://github.com/robinskil/lance-nd/wiki/Python)
explains the writer rules, chunks, encoding options, stats, scans, casting,
object stores and the encoding rules.

## Tests

```bash
.venv/bin/python -m pytest tests
```

`tests/test_s3.py` runs when `LANCE_ND_S3_ENDPOINT` points to an S3 server
with the bucket `lance-nd-test`, see the [Object stores](https://github.com/robinskil/lance-nd/wiki/Object-Stores) wiki page.
