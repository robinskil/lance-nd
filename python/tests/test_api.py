import numpy as np
import pytest
import xarray as xr

import lance_nd
from lance_nd import LanceNdError, encode


def dataset(title):
    return xr.Dataset(
        {"sst": (("time",), np.array([1.0, 2.0]), {"units": "K"})},
        coords={"time": [0, 1]},
        attrs={"title": title},
    )


def test_spec_example(table_uri):
    writer = lance_nd.create(table_uri)
    writer.write(dataset("a"))
    writer.write([dataset("b"), dataset("c")])
    writer.write(encode(dataset("d")))
    table = writer.finish()
    assert table.count_rows() == 4
    assert table.version == 1


def test_append_api(table_uri, tmp_path):
    with lance_nd.create(table_uri) as writer:
        writer.write(dataset("a"))
    writer = lance_nd.append(table_uri)
    writer.write([dataset("b"), dataset("c")])
    table = writer.finish()
    assert table.version == 2
    assert table.count_rows() == 3
    with pytest.raises(LanceNdError):
        lance_nd.append(str(tmp_path / "missing.lance"))


def test_context_manager_finishes(table_uri):
    with lance_nd.create(table_uri) as writer:
        writer.write(dataset("a"))
    assert lance_nd.open(table_uri).count_rows() == 1


def test_context_manager_aborts(table_uri):
    with lance_nd.create(table_uri) as writer:
        writer.write(dataset("a"))
    with pytest.raises(RuntimeError, match="boom"):
        with lance_nd.append(table_uri) as writer:
            writer.write(dataset("b"))
            raise RuntimeError("boom")
    table = lance_nd.open(table_uri)
    assert table.version == 1
    assert table.count_rows() == 1


def test_empty_with_block(table_uri):
    with pytest.raises(LanceNdError):
        with lance_nd.create(table_uri):
            pass
    with pytest.raises(LanceNdError):
        lance_nd.open(table_uri)
    with lance_nd.create(table_uri) as writer:
        writer.write(dataset("a"))
    with lance_nd.append(table_uri):
        pass
    assert lance_nd.open(table_uri).version == 1


def test_overwrite(table_uri):
    with lance_nd.create(table_uri) as writer:
        writer.write([dataset("a"), dataset("b")])
    with lance_nd.create(table_uri, mode="overwrite") as writer:
        writer.write(dataset("c"))
    assert lance_nd.open(table_uri).count_rows() == 1


def test_create_existing_raises(table_uri):
    with lance_nd.create(table_uri) as writer:
        writer.write(dataset("a"))
    with pytest.raises(LanceNdError):
        lance_nd.create(table_uri)


def test_write_rejects_other_types(table_uri):
    writer = lance_nd.create(table_uri)
    with pytest.raises(TypeError):
        writer.write(42)
    writer.abort()


def test_options_pass_through(table_uri):
    writer = lance_nd.create(
        table_uri,
        column_defaults={"compression": "zstd"},
        column_options={"sst": {"compression": "lz4"}},
    )
    writer.write(dataset("a"))
    assert sorted(writer.finish().variables()) == ["sst", "time"]
