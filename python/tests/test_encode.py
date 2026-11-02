import numpy as np
import pyarrow as pa
import pytest
import xarray as xr

from conftest import GLOBAL_ATTRIBUTES, ND_ARRAY
from lance_nd import encode
from lance_nd._lance_nd import Writer, _scan_batch


def write_and_scan(uri, batch):
    """Write `batch` to a new table and read it back as one nd-arrow batch."""
    writer = Writer.create(uri)
    writer.write_batch(batch)
    writer.finish()
    return _scan_batch(uri)


def storage_names(field):
    return [field.type.field(i).name for i in range(field.type.num_fields)]


def sea_surface():
    return xr.Dataset(
        {"sst": (("time", "lat"), np.arange(6, dtype="float64").reshape(2, 3), {"units": "K"})},
        coords={"time": [0, 1], "lat": [10.0, 20.0, 30.0]},
        attrs={"title": "t"},
    )


def test_structure(table_uri):
    batch = encode(sea_surface())
    assert batch.num_rows == 1
    assert sorted(batch.schema.names) == ["__global_attrs__", "lat", "sst", "time"]
    for name in ["time", "lat", "sst"]:
        field = batch.schema.field(name)
        assert field.metadata == ND_ARRAY
        assert storage_names(field)[:3] == ["values", "dim_names", "dim_sizes"]
    assert batch.schema.field("__global_attrs__").metadata == GLOBAL_ATTRIBUTES
    sst = batch.column("sst")[0].as_py()
    assert sst["dim_names"] == ["time", "lat"]
    assert sst["dim_sizes"] == [2, 3]
    assert sst["values"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert sst["attrs"]["units"] == "K"
    assert batch.column("__global_attrs__")[0].as_py() == {"title": "t"}

    read = write_and_scan(table_uri, batch)
    assert read.column("sst")[0].as_py()["values"] == sst["values"]
    assert read.column("__global_attrs__")[0].as_py()["title"] == "t"


def test_no_attrs_field_when_empty():
    ds = xr.Dataset({"sst": (("time",), np.zeros(2))})
    batch = encode(ds)
    assert "attrs" not in storage_names(batch.schema.field("sst"))
    assert "__global_attrs__" not in batch.schema.names


@pytest.mark.parametrize(
    "values, arrow_type",
    [
        (np.array([True, False]), pa.bool_()),
        (np.array([1, 2], dtype="int8"), pa.int8()),
        (np.array([1, 2], dtype="int16"), pa.int16()),
        (np.array([1, 2], dtype="int32"), pa.int32()),
        (np.array([1, 2], dtype="int64"), pa.int64()),
        (np.array([1, 2], dtype="uint8"), pa.uint8()),
        (np.array([1, 2], dtype="uint16"), pa.uint16()),
        (np.array([1, 2], dtype="uint32"), pa.uint32()),
        (np.array([1, 2], dtype="uint64"), pa.uint64()),
        (np.array([1, 2], dtype="float16"), pa.float16()),
        (np.array([1, 2], dtype="float32"), pa.float32()),
        (np.array([1, 2], dtype="float64"), pa.float64()),
        (np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"), pa.timestamp("ns")),
        (np.array([1, 2], dtype="timedelta64[s]"), pa.duration("s")),
        (np.array(["a", "bc"]), pa.string()),
        (np.array([b"a", b"bc"]), pa.string()),
        (np.array(["a", "bc"], dtype=object), pa.string()),
    ],
)
def test_value_types(values, arrow_type):
    ds = xr.Dataset({"v": (("n",), values)})
    batch = encode(ds)
    field = batch.schema.field("v")
    assert field.type.field("values").type.value_type == arrow_type
    assert len(batch.column("v")[0].as_py()["values"]) == 2


def test_unsupported_dtype_raises():
    ds = xr.Dataset({"z": (("n",), np.array([1 + 2j]))})
    with pytest.raises(TypeError, match="'z'"):
        encode(ds)


def test_missing_values():
    ds = xr.Dataset(
        {
            "f": (("n",), np.array([1.0, np.nan, 3.0])),
            "t": (("n",), np.array(["2024-01-01", "NaT", "2024-01-03"], dtype="datetime64[s]")),
            "s": (("n",), np.array(["a", None, "c"], dtype=object)),
        }
    )
    batch = encode(ds)
    assert batch.column("f")[0].as_py()["values"] == [1.0, None, 3.0]
    times = batch.column("t")[0].as_py()["values"]
    assert times[1] is None and times[0] is not None and times[2] is not None
    assert batch.column("s")[0].as_py()["values"] == ["a", None, "c"]


@pytest.mark.parametrize(
    "value, arrow_type",
    [
        ("K", pa.string()),
        (True, pa.bool_()),
        (np.bool_(True), pa.bool_()),
        (3, pa.int64()),
        (2.5, pa.float64()),
        (np.int16(3), pa.int16()),
        (np.array([1.0, 2.0]), pa.list_(pa.float64())),
        ([1, 2], pa.list_(pa.int64())),
        (("a", "b"), pa.list_(pa.string())),
    ],
)
def test_attr_types(value, arrow_type):
    ds = xr.Dataset({"v": (("n",), np.zeros(2), {"key": value})})
    attrs = encode(ds).schema.field("v").type.field("attrs").type
    assert attrs.field("key").type.equals(arrow_type, check_metadata=False) or (
        pa.types.is_list(arrow_type) and attrs.field("key").type.value_type == arrow_type.value_type
    )


def test_unsupported_attr_raises():
    ds = xr.Dataset({"v": (("n",), np.zeros(2), {"bad": {"a": 1}})})
    with pytest.raises(TypeError, match="'bad'"):
        encode(ds)


def test_multiple_datasets(table_uri):
    ds_a = sea_surface().assign(extra=(("time",), np.array([7.0, 8.0])))
    ds_b = sea_surface()
    batch = encode([ds_a, ds_b])
    assert batch.num_rows == 2
    assert batch.column("extra").is_valid().to_pylist() == [True, False]

    read = write_and_scan(table_uri, batch)
    assert read.column("extra").is_valid().to_pylist() == [True, False]
    assert read.column("sst").is_valid().to_pylist() == [True, True]


def test_type_conflict_between_datasets():
    ds_a = xr.Dataset({"sst": (("n",), np.zeros(2, dtype="float64"))})
    ds_b = xr.Dataset({"sst": (("n",), np.zeros(2, dtype="float32"))})
    with pytest.raises(TypeError, match="'sst'"):
        encode([ds_a, ds_b])
    ds_c = xr.Dataset({"v": (("n",), np.zeros(2), {"key": "text"})})
    ds_d = xr.Dataset({"v": (("n",), np.zeros(2), {"key": 1})})
    with pytest.raises(TypeError, match="'key'"):
        encode([ds_c, ds_d])


def test_cf_coordinates():
    ds = xr.Dataset(
        {"temp": (("y", "x"), np.zeros((2, 3)))},
        coords={
            "x": [0, 1, 2],
            "lat2d": (("y", "x"), np.zeros((2, 3))),
            "lon2d": (("y", "x"), np.zeros((2, 3))),
            "ref_time": ((), np.datetime64("2024-01-01", "ns")),
            "station_lat": (("station",), np.zeros(4)),
        },
    )
    batch = encode(ds)
    assert batch.column("temp")[0].as_py()["attrs"]["coordinates"] == "lat2d lon2d ref_time"
    assert batch.column("__global_attrs__")[0].as_py()["coordinates"] == "station_lat"
    x_attrs = batch.column("x")[0].as_py().get("attrs") or {}
    assert x_attrs.get("coordinates") is None


def test_existing_coordinates_attr_raises():
    ds = xr.Dataset({"temp": (("y",), np.zeros(2), {"coordinates": "lat"})})
    with pytest.raises(ValueError, match="coordinates"):
        encode(ds)
    ds = xr.Dataset({"temp": (("y",), np.zeros(2))}, attrs={"coordinates": "lat"})
    with pytest.raises(ValueError, match="coordinates"):
        encode(ds)


def test_encoding_ignored():
    ds = xr.Dataset({"sst": (("n",), np.zeros(2), {"units": "K"})})
    ds["sst"].encoding = {"_FillValue": -999, "scale_factor": 0.1}
    attrs = encode(ds).column("sst")[0].as_py()["attrs"]
    assert attrs == {"units": "K"}


def test_reserved_name():
    ds = xr.Dataset({"__global_attrs__": (("n",), np.zeros(2))})
    with pytest.raises(ValueError, match="__global_attrs__"):
        encode(ds)


def test_fortran_order(table_uri):
    data = np.asfortranarray(np.arange(6, dtype="float64").reshape(2, 3))
    ds = xr.Dataset({"sst": (("time", "lat"), data)})
    assert encode(ds).column("sst")[0].as_py()["values"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    transposed = sea_surface().transpose("lat", "time")
    batch = encode(transposed)
    sst = batch.column("sst")[0].as_py()
    assert sst["dim_names"] == ["lat", "time"]
    assert sst["values"] == transposed["sst"].values.ravel(order="C").tolist()
    read = write_and_scan(table_uri, batch)
    assert read.column("sst")[0].as_py()["values"] == sst["values"]


def test_big_endian(table_uri):
    ds = xr.Dataset({"sst": (("n",), np.array([1.5, 2.5], dtype=">f4"))})
    batch = encode(ds)
    assert batch.schema.field("sst").type.field("values").type.value_type == pa.float32()
    assert batch.column("sst")[0].as_py()["values"] == [1.5, 2.5]
    read = write_and_scan(table_uri, batch)
    assert read.column("sst")[0].as_py()["values"] == [1.5, 2.5]


def test_scalar_and_empty(table_uri):
    ds = xr.Dataset({"depth": ((), 0.5), "none": (("n",), np.zeros(0))})
    batch = encode(ds)
    depth = batch.column("depth")[0].as_py()
    assert depth["dim_names"] == [] and depth["values"] == [0.5]
    empty = batch.column("none")[0].as_py()
    assert empty["dim_sizes"] == [0] and empty["values"] == []
    read = write_and_scan(table_uri, batch)
    assert read.column("depth")[0].as_py()["values"] == [0.5]


def test_dask_is_loaded():
    pytest.importorskip("dask")
    ds = sea_surface()
    chunked = ds.chunk({"time": 1})
    assert encode(chunked).column("sst").to_pylist() == encode(ds).column("sst").to_pylist()


def test_rejects_other_input():
    with pytest.raises(TypeError):
        encode([42])


def test_nan_attribute_is_kept():
    ds = xr.Dataset(
        {"v": (("n",), np.zeros(2), {"_FillValue": np.float32("nan"), "range": np.array([np.nan, 1.0])})}
    )
    attrs = encode(ds).column("v")[0].as_py()["attrs"]
    assert attrs["_FillValue"] is not None and np.isnan(attrs["_FillValue"])
    assert np.isnan(attrs["range"][0]) and attrs["range"][1] == 1.0


def test_too_many_values_raises():
    from lance_nd._encode import _column

    class Big:
        type = pa.float32()

        def __len__(self):
            return 2**30

    rows = [(Big(), ["n"], (2**30,), {}), (Big(), ["n"], (2**30,), {})]
    with pytest.raises(ValueError, match="'v'"):
        _column("v", rows)
