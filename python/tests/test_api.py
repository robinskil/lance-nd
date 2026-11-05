from datetime import timedelta

import numpy as np
import pyarrow as pa
import pytest
import xarray as xr

import lance_nd
from nd_arrow_array import NdRecordBatch

from lance_nd import LanceNdError
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
    writer.write(NdRecordBatch.from_xarray(dataset("d")))
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
    assert not any(name.startswith("__stats__") for name in table.variables())
    assert not any(name.startswith("__stats__") for name in _scan_batch(table_uri).schema.names)


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


def sst_dataset(title, values):
    return xr.Dataset(
        {"sst": (("time",), np.array(values), {"units": "K"})},
        coords={"time": np.arange(len(values))},
        attrs={"title": title},
    )


def titles(batches):
    return [batch.attrs["title"] for batch in batches]


@pytest.fixture
def ocean(table_uri):
    with lance_nd.create(table_uri, stats=True) as writer:
        writer.write(sst_dataset("a", [1.0, 2.0]))
        writer.write(sst_dataset("b", [5.0, 6.0, 7.0]))
        writer.write(sst_dataset("c", [3.0]))
    return lance_nd.open(table_uri)


def test_scan_streams_nd_record_batches(ocean):
    scanner = ocean.scan()
    first = next(scanner)
    assert isinstance(first, NdRecordBatch)
    assert set(first.keys()) == {"sst", "time"}
    rest = list(scanner)
    assert titles([first, *rest]) == ["a", "b", "c"]
    assert rest[0]["sst"].to_numpy().tolist() == [5.0, 6.0, 7.0]
    xr.testing.assert_identical(first.to_xarray(), sst_dataset("a", [1.0, 2.0]))


def test_scan_project_filter_and_limit(ocean):
    batches = list(ocean.scan(project=["sst"]))
    assert all(batch.keys() == ["sst"] for batch in batches)
    assert titles(ocean.scan(filter="__global_attrs__.title = 'b'")) == ["b"]
    assert titles(ocean.scan(filter="sst.attrs.units = 'K'", limit=2)) == ["a", "b"]


def test_scan_empty_projection_reads_global_attributes(ocean):
    batches = list(ocean.scan(project=[]))
    assert [batch.keys() for batch in batches] == [[]] * 3
    assert titles(batches) == ["a", "b", "c"]


def test_scan_prune_uses_stats(ocean):
    assert titles(ocean.scan(prune="sst.values > 4.5")) == ["b"]
    assert titles(ocean.scan(prune="sst.values BETWEEN 2.5 AND 3.5")) == ["c"]
    both = ocean.scan(filter="__global_attrs__.title <> 'b'", prune="sst.values < 2.5")
    assert titles(both) == ["a"]


def test_scan_errors(ocean):
    with pytest.raises(LanceNdError):
        next(ocean.scan(project=["sst.attrs.units"]))
    with pytest.raises(LanceNdError):
        ocean.scan(filter="nope = 1 AND")


def test_scan_keeps_the_case_of_names(table_uri):
    ds = xr.Dataset({"TEMP": (("n",), np.array([1.0, 9.0]))}, attrs={"title": "a"})
    with lance_nd.create(table_uri, stats=True) as writer:
        writer.write(ds)
        writer.write(ds.assign(TEMP=ds.TEMP * 10).assign_attrs(title="b"))
    table = lance_nd.open(table_uri)
    assert titles(table.scan(prune="TEMP.values > 50")) == ["b"]
    assert titles(table.scan(project=["TEMP"], filter="__global_attrs__.title = 'a'")) == ["a"]


def resolution_dataset(title, resolution):
    ds = dataset(title)
    ds["sst"].attrs["resolution"] = resolution
    return ds


def test_rejected_write_keeps_the_writer(table_uri):
    with lance_nd.create(table_uri) as writer:
        writer.write(resolution_dataset("a", 0.5))
        with pytest.raises(LanceNdError, match="resolution"):
            writer.write(resolution_dataset("b", np.array([1.0, 2.0])))
        writer.write(resolution_dataset("c", 0.25))
    table = lance_nd.open(table_uri)
    assert titles(table.scan(project=[])) == ["a", "c"]


def test_empty_array_attribute_is_dropped(table_uri):
    ds = resolution_dataset("b", np.array([], dtype=np.float64))
    ds.attrs["history"] = []
    only_empty = xr.Dataset({"x": (("n",), np.array([1.0]), {"res": np.array([])})})
    with lance_nd.create(table_uri) as writer:
        writer.write(resolution_dataset("a", 0.5))
        writer.write(ds)
        writer.write(only_empty)
    rows = list(lance_nd.open(table_uri).scan())
    assert len(rows) == 3
    assert "resolution" not in rows[1]["sst"].attrs
    assert "history" not in rows[1].attrs
    assert "res" not in rows[2]["x"].attrs
    # The dataset of the caller keeps its attributes.
    assert "history" in ds.attrs


def typed_dataset(title, dtype):
    return xr.Dataset({"sst": (("time",), np.array([1.5, 2.5], dtype=dtype))}, attrs={"title": title})


def sst_types(table):
    return [batch["sst"].dtype for batch in table.scan()]


def test_casting_default_rejects_another_type(table_uri):
    with lance_nd.create(table_uri) as writer:
        writer.write(typed_dataset("a", np.float64))
        with pytest.raises(LanceNdError, match="sst"):
            writer.write(typed_dataset("b", np.float32))
    assert lance_nd.open(table_uri).count_rows() == 1


def test_casting_safe_widens_new_rows(table_uri):
    with lance_nd.create(table_uri, stats=True, casting="safe") as writer:
        writer.write(typed_dataset("a", np.float64))
        writer.write(typed_dataset("b", np.float32))
    with lance_nd.append(table_uri, casting="safe") as writer:
        writer.write(typed_dataset("c", np.int16))
        with pytest.raises(LanceNdError):
            writer.write(xr.Dataset({"sst": (("time",), np.array(["x"]))}))
    table = lance_nd.open(table_uri)
    assert sst_types(table) == ["double"] * 3
    assert titles(table.scan(prune="sst.values > 1.5")) == ["a", "b", "c"]


def test_casting_same_kind_narrows_new_rows(table_uri):
    with lance_nd.create(table_uri, casting="same_kind") as writer:
        writer.write(typed_dataset("a", np.float32))
        writer.write(typed_dataset("b", np.float64))
    assert sst_types(lance_nd.open(table_uri)) == ["float", "float"]


def test_casting_must_be_known(table_uri):
    with pytest.raises(ValueError, match="casting"):
        lance_nd.create(table_uri, casting="unsafe")


def test_widen(table_uri):
    with lance_nd.create(table_uri, stats=True) as writer:
        writer.write(typed_dataset("a", np.float32))
    table = lance_nd.open(table_uri)
    with pytest.raises(LanceNdError, match="could change"):
        table.widen("sst", pa.float16())
    table.widen("sst", pa.float64())
    assert table.version > 1
    with lance_nd.append(table_uri) as writer:
        writer.write(typed_dataset("b", np.float64))
    table = lance_nd.open(table_uri)
    assert sst_types(table) == ["double", "double"]
    assert titles(table.scan(prune="sst.values > 2")) == ["a", "b"]


def test_casting_promote_widens_the_table(table_uri):
    with lance_nd.create(table_uri, stats=True, casting="promote") as writer:
        writer.write(typed_dataset("a", np.float32))
        writer.write(typed_dataset("b", np.float64))
        writer.write(typed_dataset("c", np.float32))
    table = lance_nd.open(table_uri)
    assert sst_types(table) == ["double"] * 3
    assert titles(table.scan(prune="sst.values > 2")) == ["a", "b", "c"]


def test_error_classes(table_uri, tmp_path):
    missing = str(tmp_path / "missing.lance")
    with pytest.raises(lance_nd.TableNotFoundError):
        lance_nd.open(missing)
    with pytest.raises(lance_nd.TableNotFoundError):
        lance_nd.append(missing)
    with lance_nd.create(table_uri) as writer:
        writer.write(dataset("a"))
    with pytest.raises(lance_nd.TableExistsError):
        lance_nd.create(table_uri)
    # Each class is a LanceNdError.
    assert issubclass(lance_nd.CommitConflictError, LanceNdError)


def test_versions(table_uri):
    with lance_nd.create(table_uri) as writer:
        writer.write(dataset("a"))
    with lance_nd.append(table_uri) as writer:
        writer.write(dataset("b"))
    table = lance_nd.open(table_uri)
    history = table.versions()
    assert [version for version, _ in history] == [1, 2]
    assert history[0][1] <= history[1][1]
    assert table.checkout(1).count_rows() == 1
    table.restore(1)
    assert table.version == 3
    assert table.count_rows() == 1
