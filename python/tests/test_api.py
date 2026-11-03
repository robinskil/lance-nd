from datetime import timedelta

import numpy as np
import pytest
import xarray as xr

import lance_nd
from lance_nd import LanceNdError, encode
from lance_nd._lance_nd import _scan_batch


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


def test_finish_inside_with_block(table_uri):
    with lance_nd.create(table_uri) as writer:
        writer.write(dataset("a"))
        table = writer.finish()
    assert table.count_rows() == 1
    assert writer.table.version == 1
    assert lance_nd.open(table_uri).count_rows() == 1


def test_failed_finish_inside_with_block_keeps_its_error(table_uri):
    late = lance_nd.create(table_uri)
    with lance_nd.create(table_uri) as writer:
        writer.write(dataset("a"))
    with pytest.raises(LanceNdError, match="a table exists"):
        with late:
            late.write(dataset("b"))
            late.finish()
    assert lance_nd.open(table_uri).count_rows() == 1


def test_create_with_stats(table_uri):
    with lance_nd.create(table_uri, stats=True) as writer:
        writer.write(dataset("a"))
    table = lance_nd.open(table_uri)
    assert table.has_stats
    assert "__stats__" not in table.variables()
    assert "__stats__" not in _scan_batch(table_uri).schema.names


def test_stats_off_by_default(table_uri):
    with lance_nd.create(table_uri) as writer:
        writer.write(dataset("a"))
    assert not lance_nd.open(table_uri).has_stats


def test_small_writes_share_a_fragment(table_uri):
    with lance_nd.create(table_uri) as writer:
        for title in "abc":
            writer.write(dataset(title))
    assert lance_nd.open(table_uri).num_fragments == 1


def test_fragment_options(table_uri):
    with lance_nd.create(table_uri, target_fragment_bytes=1, write_parallelism=2) as writer:
        for title in "abc":
            writer.write(dataset(title))
    with lance_nd.append(table_uri, target_fragment_bytes=1) as writer:
        writer.write([dataset("d"), dataset("e")])
    table = lance_nd.open(table_uri)
    assert table.num_fragments == 4
    assert table.count_rows() == 5


def test_delete(table_uri):
    with lance_nd.create(table_uri) as writer:
        writer.write([dataset(title) for title in "abc"])
    table = lance_nd.open(table_uri)
    assert table.delete("__global_attrs__.title = 'b'") == 1
    assert table.version == 2
    assert table.delete("__global_attrs__.title = 'nope'") == 0
    assert table.count_rows() == 2
    assert lance_nd.open(table_uri).count_rows() == 2


def test_delete_bad_predicate_raises(table_uri):
    with lance_nd.create(table_uri) as writer:
        writer.write(dataset("a"))
    with pytest.raises(LanceNdError):
        lance_nd.open(table_uri).delete("nope = 1")


def test_compact_and_cleanup(table_uri):
    with lance_nd.create(table_uri, target_fragment_bytes=1) as writer:
        for title in "abcd":
            writer.write(dataset(title))
    table = lance_nd.open(table_uri)
    assert table.num_fragments == 4
    table.delete("__global_attrs__.title = 'a'")
    metrics = table.compact()
    assert metrics["fragments_removed"] == 3
    assert metrics["fragments_added"] == 1
    assert table.num_fragments == 1
    assert table.count_rows() == 3
    assert table.compact()["fragments_removed"] == 0
    removed = table.cleanup_old_versions(timedelta(0))
    assert removed["old_versions"] > 0
    assert removed["bytes_removed"] > 0
    assert lance_nd.open(table_uri).count_rows() == 3
