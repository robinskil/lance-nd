import warnings

import dask.array as da
import numpy as np
import pytest
import xarray as xr

import lance_nd
from lance_nd._chunks import split_chunks
from nd_arrow_array import NdRecordBatch, encode_batches


def encode(ds, chunks, on_drop="warn"):
    """The chunks of `ds` as one record batch, one row per chunk."""
    return encode_batches([NdRecordBatch.from_xarray(part) for part in split_chunks(ds, chunks, on_drop)])


def ocean():
    """time=4 in chunks of 2, depth=3 in chunks of 2 (sizes 2 and 1)."""
    temp = np.arange(4 * 3 * 5, dtype="float64").reshape(4, 3, 5)
    ds = xr.Dataset(
        {
            "temp": (("time", "depth", "position"), temp, {"units": "C"}),
            "level": (("time", "depth"), np.arange(12, dtype="float64").reshape(4, 3)),
            "flag": ((), 1),
            "sst": (("time", "lat"), np.zeros((4, 2))),
            "station_name": (("station",), np.array(["a", "b"])),
        },
        coords={
            "time": np.arange(4),
            "depth": [0.0, 10.0, 20.0],
            "position": np.arange(5),
        },
        attrs={"title": "ocean"},
    )
    return ds.chunk({"time": 2, "depth": 2})


def test_one_row_per_chunk():
    batch = encode(ocean(), chunks=["time", "depth"], on_drop="ignore")
    assert batch.num_rows == 4
    times = [row["values"] for row in batch.column("time").to_pylist()]
    depths = [row["values"] for row in batch.column("depth").to_pylist()]
    assert times == [[0, 1], [0, 1], [2, 3], [2, 3]]
    assert depths == [[0.0, 10.0], [20.0], [0.0, 10.0], [20.0]]


def test_chunk_values_and_shapes():
    ds = ocean()
    batch = encode(ds, chunks=["time", "depth"], on_drop="ignore")
    last = batch.column("temp")[3].as_py()
    assert last["dim_names"] == ["time", "depth", "position"]
    assert last["dim_sizes"] == [2, 1, 5]
    expected = ds["temp"].isel(time=slice(2, 4), depth=slice(2, 3)).values.ravel().tolist()
    assert last["values"] == expected


def test_kept_and_dropped_variables():
    batch = encode(ocean(), chunks=["time", "depth"], on_drop="ignore")
    assert set(batch.schema.names) == {"temp", "level", "flag", "time", "depth", "__global_attrs__"}


def test_attributes_on_every_row():
    batch = encode(ocean(), chunks=["time", "depth"], on_drop="ignore")
    assert all(row["attrs"]["units"] == "C" for row in batch.column("temp").to_pylist())
    assert all(row["title"] == "ocean" for row in batch.column("__global_attrs__").to_pylist())


def test_on_drop_warn_is_default():
    with pytest.warns(UserWarning, match="position.*sst.*station_name"):
        encode(ocean(), chunks=["time", "depth"])


def test_on_drop_raise():
    with pytest.raises(ValueError, match="station_name"):
        encode(ocean(), chunks=["time", "depth"], on_drop="raise")


def test_on_drop_ignore_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        encode(ocean(), chunks=["time", "depth"], on_drop="ignore")


def test_invalid_on_drop():
    with pytest.raises(ValueError, match="on_drop"):
        encode(ocean(), chunks=["time"], on_drop="maybe")


def test_unknown_chunk_dimension():
    with pytest.raises(ValueError, match="'height'"):
        encode(ocean(), chunks=["height"], on_drop="ignore")


def test_no_kept_variable_raises():
    ds = xr.Dataset(
        {"x": (("a", "c"), np.zeros((2, 2))), "y": (("b", "c"), np.zeros((2, 2)))}
    ).chunk({"a": 1, "b": 1})
    with pytest.raises(ValueError, match="no variable"):
        encode(ds, chunks=["a", "b"], on_drop="ignore")


def test_cf_coordinates_list_only_kept_variables():
    ds = ocean().assign_coords(depth_label=(("depth",), np.array(["s", "m", "d"])))
    batch = encode(ds, chunks=["time", "depth"], on_drop="ignore")
    assert batch.column("temp")[0].as_py()["attrs"]["coordinates"] == "depth_label"


def test_split_chunks_is_lazy():
    computed = []

    def block(x, block_info=None):
        computed.append(block_info[None]["chunk-location"])
        return x

    data = da.map_blocks(block, da.zeros((4,), chunks=2), dtype="float64")
    ds = xr.Dataset({"v": (("time",), data)})
    batches = (NdRecordBatch.from_xarray(part) for part in split_chunks(ds, chunks=["time"]))
    assert computed == []
    next(batches)
    assert len(computed) == 1
    assert len(list(batches)) == 1
    assert len(computed) == 2


def test_lazy_write(table_uri):
    ds = ocean()
    with lance_nd.create(table_uri) as writer:
        writer.write(ds, chunks=["time", "depth"], on_drop="ignore")
    table = lance_nd.open(table_uri)
    assert table.count_rows() == 4
    assert sorted(table.variables()) == ["depth", "flag", "level", "temp", "time"]


def test_write_record_batch_with_chunks_raises(table_uri):
    writer = lance_nd.create(table_uri)
    with pytest.raises(TypeError):
        writer.write(encode(ocean(), chunks=["time"], on_drop="ignore"), chunks=["time"])
    with pytest.raises(TypeError):
        writer.write(NdRecordBatch.from_xarray(ocean().drop_vars(["sst", "station_name"])), chunks=["time"])
    writer.abort()


def test_list_of_datasets(table_uri):
    with lance_nd.create(table_uri) as writer:
        writer.write([ocean(), ocean()], chunks=["time", "depth"], on_drop="ignore")
    assert lance_nd.open(table_uri).count_rows() == 8
