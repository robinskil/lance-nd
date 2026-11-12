"""The lance-nd sync command."""

import os
import threading

import numpy as np
import pytest
import xarray as xr

import lance_nd
from lance_nd.cli import _parser, _sync, main

MTIME = 1_700_000_000_000_000_000


def put(path, title, mtime=MTIME):
    """Write a netCDF file with ``title``, and set its modification time unless it is None."""
    xr.Dataset({"sst": ("time", np.arange(3.0))}, attrs={"title": title}).to_netcdf(path)
    if mtime is not None:
        os.utime(path, ns=(mtime, mtime))


def rows(uri):
    """The (source_file name, title) of each row, sorted."""
    return sorted(
        (os.path.basename(ds.attrs["source_file"]), ds.attrs["title"])
        for ds in (row.to_xarray() for row in lance_nd.open(uri).scan())
    )


@pytest.fixture
def folder(tmp_path):
    path = tmp_path / "data"
    path.mkdir()
    for name in ["a", "b", "c"]:
        put(path / f"{name}.nc", name)
    return path


def sync(folder, uri, *options):
    return main(["sync", *options, str(folder / "*.nc"), uri])


def test_first_sync_creates_the_table(folder, table_uri):
    assert sync(folder, table_uri) == 0
    assert rows(table_uri) == [("a.nc", "a"), ("b.nc", "b"), ("c.nc", "c")]


def test_no_change_makes_no_version(folder, table_uri):
    assert sync(folder, table_uri) == 0
    version = lance_nd.open(table_uri).version
    assert sync(folder, table_uri) == 0
    assert lance_nd.open(table_uri).version == version


def test_add_change_and_delete(folder, table_uri, capsys):
    assert sync(folder, table_uri) == 0
    put(folder / "a.nc", "a2", mtime=MTIME + 1_000_000_000)
    (folder / "b.nc").unlink()
    put(folder / "d.nc", "d")
    capsys.readouterr()
    assert sync(folder, table_uri) == 0
    assert rows(table_uri) == [("a.nc", "a2"), ("c.nc", "c"), ("d.nc", "d")]
    assert "1 files added, 1 files changed, 1 files deleted, 0 files failed" in capsys.readouterr().err


def test_size_change_without_mtime_change(folder, table_uri):
    assert sync(folder, table_uri) == 0
    xr.Dataset({"sst": ("time", np.arange(9.0))}, attrs={"title": "big"}).to_netcdf(folder / "a.nc")
    os.utime(folder / "a.nc", ns=(MTIME, MTIME))
    assert sync(folder, table_uri) == 0
    assert rows(table_uri) == [("a.nc", "big"), ("b.nc", "b"), ("c.nc", "c")]


def test_failed_file_keeps_its_old_row(folder, table_uri):
    assert sync(folder, table_uri) == 0
    (folder / "a.nc").write_text("not netCDF")
    put(folder / "d.nc", "d")
    assert sync(folder, table_uri) == 1
    assert rows(table_uri) == [("a.nc", "a"), ("b.nc", "b"), ("c.nc", "c"), ("d.nc", "d")]


def test_failed_first_sync_creates_no_table(tmp_path, table_uri):
    (tmp_path / "a.nc").write_text("not netCDF")
    assert sync(tmp_path, table_uri) == 1
    with pytest.raises(lance_nd.TableNotFoundError):
        lance_nd.open(table_uri)


def test_min_age_waits_for_new_files(folder, table_uri):
    put(folder / "d.nc", "d", mtime=None)
    assert sync(folder, table_uri, "--min-age", "3600") == 0
    assert rows(table_uri) == [("a.nc", "a"), ("b.nc", "b"), ("c.nc", "c")]
    assert sync(folder, table_uri) == 0
    assert ("d.nc", "d") in rows(table_uri)


def test_empty_match_keeps_all_rows(folder, table_uri, capsys):
    assert sync(folder, table_uri) == 0
    for path in folder.iterdir():
        path.unlink()
    assert sync(folder, table_uri) == 0
    assert "keep all rows" in capsys.readouterr().err
    assert len(rows(table_uri)) == 3


def test_rows_without_mtime_are_written_again(folder, table_uri):
    with lance_nd.create(table_uri) as writer:
        writer.write(
            xr.Dataset({"sst": ("time", np.zeros(3))}, attrs={"title": "old", "source_file": (folder / "a.nc").as_posix()})
        )
    assert sync(folder, table_uri) == 0
    assert rows(table_uri) == [("a.nc", "a"), ("b.nc", "b"), ("c.nc", "c")]


def test_stale_rows_after_a_stop_are_deleted(folder, table_uri):
    assert sync(folder, table_uri) == 0
    with lance_nd.append(table_uri) as writer:
        writer.write(
            xr.Dataset(
                {"sst": ("time", np.zeros(3))},
                attrs={
                    "title": "stale",
                    "source_file": (folder / "a.nc").as_posix(),
                    "source_mtime": 1,
                    "source_size": 1,
                },
            )
        )
    assert sync(folder, table_uri) == 0
    assert rows(table_uri) == [("a.nc", "a"), ("b.nc", "b"), ("c.nc", "c")]


def test_quote_in_a_file_name(folder, table_uri):
    put(folder / "it's.nc", "q")
    assert sync(folder, table_uri) == 0
    (folder / "it's.nc").unlink()
    assert sync(folder, table_uri) == 0
    assert rows(table_uri) == [("a.nc", "a"), ("b.nc", "b"), ("c.nc", "c")]


def test_compact_merges_fragments(folder, table_uri):
    for name in ["d", "e", "f"]:
        put(folder / f"{name}.nc", name)
        assert sync(folder, table_uri, "--no-compact") == 0
    assert lance_nd.open(table_uri).num_fragments == 3
    put(folder / "g.nc", "g")
    assert sync(folder, table_uri) == 0
    assert lance_nd.open(table_uri).num_fragments == 1


def test_interval_runs_until_stop(folder, table_uri, monkeypatch):
    stop = threading.Event()
    calls = []

    def counted(args):
        calls.append(args.table)
        if len(calls) == 3:
            stop.set()
        return True

    monkeypatch.setattr(_sync, "cycle", counted)
    args = _parser().parse_args(["sync", "--interval", "0.01", str(folder / "*.nc"), table_uri])
    assert _sync.run(args, stop) == 0
    assert calls == [table_uri] * 3


def test_interval_continues_after_a_failed_cycle(folder, table_uri, monkeypatch):
    stop = threading.Event()
    calls = []

    def failing(args):
        calls.append(1)
        if len(calls) == 2:
            stop.set()
        raise OSError("table offline")

    monkeypatch.setattr(_sync, "cycle", failing)
    args = _parser().parse_args(["sync", "--interval", "0.01", str(folder / "*.nc"), table_uri])
    assert _sync.run(args, stop) == 0
    assert len(calls) == 2
