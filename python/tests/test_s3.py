"""Tables on S3. Set ``LANCE_ND_S3_ENDPOINT`` to an S3 server, for example
moto, to run these tests."""

import os
import time

import numpy as np
import pytest
import xarray as xr

import lance_nd

ENDPOINT = os.environ.get("LANCE_ND_S3_ENDPOINT")
pytestmark = pytest.mark.skipif(ENDPOINT is None, reason="LANCE_ND_S3_ENDPOINT is not set")


def storage_options():
    return {
        "aws_endpoint": ENDPOINT,
        "aws_access_key_id": "test",
        "aws_secret_access_key": "test",
        "aws_region": "us-east-1",
        "allow_http": "true",
    }


def table_uri(test):
    bucket = os.environ.get("LANCE_ND_S3_BUCKET", "lance-nd-test")
    return f"s3://{bucket}/python-{test}-{time.time_ns()}/table.lance"


def test_round_trip_on_s3():
    uri = table_uri("round-trip")
    ds = xr.Dataset({"sst": ("time", np.arange(3.0))}, attrs={"title": "a"})
    with lance_nd.create(uri, stats=True, storage_options=storage_options()) as writer:
        writer.write(ds)
    with lance_nd.append(uri, storage_options=storage_options()) as writer:
        writer.write(ds)
    table = lance_nd.open(uri, storage_options=storage_options())
    assert table.count_rows() == 2
    assert len(list(table.scan())) == 2


def test_open_a_missing_table_fails():
    with pytest.raises(lance_nd.TableNotFoundError):
        lance_nd.open(table_uri("missing"), storage_options=storage_options())
