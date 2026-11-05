"""Round trips of xarray datasets through a table."""

import numpy as np
import pyarrow as pa
import pytest
import xarray as xr
from nd_arrow_array import NdRecordBatch

from conftest import GLOBAL_ATTRIBUTES, ND_ARRAY
import lance_nd
from lance_nd._lance_nd import _scan_batch


def round_trip(uri, *datasets):
    """Write `datasets` to a new table and read them back as xarray datasets."""
    with lance_nd.create(uri) as writer:
        writer.write(list(datasets))
    return [row.to_xarray() for row in lance_nd.open(uri).scan()]


def sea_surface():
    return xr.Dataset(
        {"sst": (("time", "lat"), np.arange(6, dtype="float64").reshape(2, 3), {"units": "K"})},
        coords={"time": [0, 1], "lat": [10.0, 20.0, 30.0]},
        attrs={"title": "t"},
    )


def test_structure(table_uri):
    with lance_nd.create(table_uri) as writer:
        writer.write(sea_surface())
    batch = _scan_batch(table_uri)
    assert sorted(batch.schema.names) == ["__global_attrs__", "lat", "sst", "time"]
    for name in ["time", "lat", "sst"]:
        assert batch.schema.field(name).metadata == ND_ARRAY
    assert batch.schema.field("__global_attrs__").metadata == GLOBAL_ATTRIBUTES
    sst = batch.column("sst")[0].as_py()
    assert sst["dim_names"] == ["time", "lat"]
    assert sst["dim_sizes"] == [2, 3]
    assert sst["values"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert sst["attrs"]["units"] == "K"


def test_round_trip(table_uri):
    (read,) = round_trip(table_uri, sea_surface())
    xr.testing.assert_identical(read, sea_surface())


@pytest.mark.parametrize(
    "values",
    [
        np.array([True, False]),
        np.array([1, 2], dtype="int8"),
        np.array([1, 2], dtype="uint64"),
        np.array([1, 2], dtype="float32"),
        np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
        np.array(["a", "bc"]),
    ],
)
def test_value_types(table_uri, values):
    (read,) = round_trip(table_uri, xr.Dataset({"v": (("n",), values)}))
    assert read["v"].dtype == values.dtype
    np.testing.assert_array_equal(read["v"].values, values)


def test_missing_values(table_uri):
    ds = xr.Dataset({"f": (("n",), np.array([1.0, np.nan, 3.0]))})
    (read,) = round_trip(table_uri, ds)
    xr.testing.assert_identical(read, ds)


def test_datasets_with_other_variables(table_uri):
    a = sea_surface().assign(extra=(("time",), np.array([7.0, 8.0])))
    b = sea_surface()
    read = round_trip(table_uri, a, b)
    assert "extra" in read[0] and "extra" not in read[1]
    batch = _scan_batch(table_uri)
    assert batch.column("extra").is_valid().to_pylist() == [True, False]


def test_type_conflict_between_datasets(table_uri):
    a = xr.Dataset({"sst": (("n",), np.zeros(2, dtype="float64"))})
    b = xr.Dataset({"sst": (("n",), np.zeros(2, dtype="float32"))})
    with lance_nd.create(table_uri) as writer:
        writer.write(a)
        with pytest.raises(lance_nd.SchemaMismatchError, match="sst"):
            writer.write(b)


def test_cf_coordinates(table_uri):
    ds = xr.Dataset(
        {"temp": (("y", "x"), np.zeros((2, 3)))},
        coords={
            "x": [0, 1, 2],
            "lat2d": (("y", "x"), np.ones((2, 3))),
            "station_lat": (("station",), np.zeros(4)),
        },
    )
    (read,) = round_trip(table_uri, ds)
    xr.testing.assert_identical(read, ds)


@pytest.mark.parametrize("name", ["__global_attrs__", "__stats__", "__stats__x"])
def test_reserved_names(table_uri, name):
    with lance_nd.create(table_uri) as writer:
        with pytest.raises(lance_nd.SchemaMismatchError, match=name):
            writer.write(xr.Dataset({name: (("n",), np.zeros(2))}))
        writer.write(sea_surface())


def test_fortran_order_and_big_endian(table_uri):
    data = np.asfortranarray(np.arange(6, dtype=">f8").reshape(2, 3))
    ds = xr.Dataset({"sst": (("time", "lat"), data)})
    (read,) = round_trip(table_uri, ds)
    np.testing.assert_array_equal(read["sst"].values, data)


def test_scalar_and_empty(table_uri):
    ds = xr.Dataset({"depth": ((), 0.5), "none": (("n",), np.zeros(0))})
    (read,) = round_trip(table_uri, ds)
    assert float(read["depth"]) == 0.5
    assert read["none"].shape == (0,)


def test_dask_is_loaded(table_uri):
    pytest.importorskip("dask")
    (read,) = round_trip(table_uri, sea_surface().chunk({"time": 1}))
    xr.testing.assert_identical(read, sea_surface())


def test_julian_dates_keep_their_instant(table_uri):
    raw = xr.Dataset(
        {"temp": (("time",), np.array([1.0, 2.0]))},
        coords={"time": ("time", np.array([5.0, 6.5]), {"units": "days since 1973-02-20", "calendar": "julian"})},
    )
    ds = xr.decode_cf(raw, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    (read,) = round_trip(table_uri, ds)
    # Julian 1973-02-25 is the Gregorian 1973-03-10.
    assert [str(t) for t in read["time"].values] == [
        "1973-03-10T00:00:00.000000",
        "1973-03-11T12:00:00.000000",
    ]


def test_write_nd_record_batches_and_arrow(table_uri):
    batch = NdRecordBatch.from_xarray(sea_surface())
    with lance_nd.create(table_uri) as writer:
        writer.write(batch)
        writer.write([batch, sea_surface()])
        writer.write(batch.to_arrow())
        writer.write(pa.Table.from_batches([batch.to_arrow()]))
    assert lance_nd.open(table_uri).count_rows() == 5


def test_rejects_other_input(table_uri):
    with lance_nd.create(table_uri) as writer:
        with pytest.raises(TypeError):
            writer.write([42])
        with pytest.raises(TypeError):
            writer.write(42)
        writer.write(sea_surface())
