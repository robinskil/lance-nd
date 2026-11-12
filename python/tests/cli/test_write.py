"""The lance-nd create and append commands."""

import os
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

import lance_nd
from lance_nd.cli import _files, main


def sea_surface(title, dtype="float64"):
    return xr.Dataset(
        {"sst": (("time",), np.arange(3, dtype=dtype), {"units": "K"})},
        coords={"time": [0, 1, 2]},
        attrs={"title": title},
    )


def identity(path, name="source_file"):
    """The global attributes that the command adds for ``path``."""
    result = os.stat(path)
    return {name: path.as_posix(), "source_mtime": result.st_mtime_ns, "source_size": result.st_size}


def titles(uri):
    return [row.to_xarray().attrs["title"] for row in lance_nd.open(uri).scan()]


@pytest.fixture
def files(tmp_path):
    """Three netCDF files, in reverse order of creation."""
    folder = tmp_path / "data"
    folder.mkdir()
    for title in ["c", "a", "b"]:
        sea_surface(title).to_netcdf(folder / f"{title}.nc")
    return folder


def test_create_writes_one_sorted_row_per_file(files, table_uri):
    assert main(["create", str(files / "*.nc"), table_uri]) == 0
    assert titles(table_uri) == ["a", "b", "c"]


def test_accepts_paths_expanded_by_the_shell(files, table_uri):
    paths = [str(files / "b.nc"), str(files / "a.nc")]
    assert main(["create", *paths, table_uri]) == 0
    assert titles(table_uri) == ["a", "b"]


def test_no_match_fails(tmp_path, table_uri, capsys):
    assert main(["create", str(tmp_path / "*.nc"), table_uri]) == 1
    assert "no file matches" in capsys.readouterr().err
    with pytest.raises(lance_nd.TableNotFoundError):
        lance_nd.open(table_uri)


def test_append_and_overwrite(files, table_uri):
    pattern = str(files / "*.nc")
    assert main(["create", pattern, table_uri]) == 0
    assert main(["create", pattern, table_uri]) == 1
    assert main(["append", str(files / "a.nc"), table_uri]) == 0
    assert titles(table_uri) == ["a", "b", "c", "a"]
    assert main(["create", "--overwrite", "--stats", str(files / "c.nc"), table_uri]) == 0
    assert titles(table_uri) == ["c"]


def test_casting(files, table_uri):
    sea_surface("d", dtype="float32").to_netcdf(files / "d.nc")
    pattern = str(files / "*.nc")
    assert main(["create", pattern, table_uri]) == 1
    assert main(["create", "--casting", "safe", pattern, table_uri]) == 0
    assert titles(table_uri) == ["a", "b", "c", "d"]


@pytest.fixture
def unsupported(monkeypatch, files):
    """b.nc opens with a complex variable and unsupported attributes."""
    real = _files.open_dataset

    def fake(path):
        ds = real(path)
        if path.endswith("b.nc"):
            ds["wave"] = ("time", np.zeros(3, dtype=complex))
            ds.attrs["meta"] = {"x": 1}
            ds["sst"].attrs["flag"] = 1j
        return ds

    monkeypatch.setattr(_files, "open_dataset", fake)
    return files


def test_unsupported_parts_fail_by_default(unsupported, table_uri, capsys):
    assert main(["create", str(unsupported / "*.nc"), table_uri]) == 1
    assert "b.nc" in capsys.readouterr().err
    with pytest.raises(lance_nd.TableNotFoundError):
        lance_nd.open(table_uri)


def test_skip_variables_and_attributes(unsupported, table_uri, capsys):
    args = ["create", "--skip-variables", "--skip-attributes", str(unsupported / "*.nc"), table_uri]
    assert main(args) == 0
    err = capsys.readouterr().err
    assert "wave" in err and "meta" in err and "flag" in err
    assert "3 files written, 0 files skipped, 1 variables dropped, 2 attributes dropped" in err
    rows = [row.to_xarray() for row in lance_nd.open(table_uri).scan()]
    b = rows[1]
    assert b.attrs == {"title": "b", **identity(unsupported / "b.nc")}
    assert "wave" not in b
    assert b["sst"].attrs == {"units": "K"}


def test_skip_attributes_alone_keeps_failure(unsupported, table_uri):
    assert main(["create", "--skip-attributes", str(unsupported / "*.nc"), table_uri]) == 1


def test_skip_errors(files, table_uri, capsys):
    (files / "broken.nc").write_text("not netCDF")
    assert main(["create", str(files / "*.nc"), table_uri]) == 1
    assert main(["create", "--skip-errors", str(files / "*.nc"), table_uri]) == 0
    assert "broken.nc" in capsys.readouterr().err
    assert titles(table_uri) == ["a", "b", "c"]


def test_skip_errors_with_no_good_file_fails(tmp_path, table_uri):
    (tmp_path / "broken.nc").write_text("not netCDF")
    assert main(["create", "--skip-errors", str(tmp_path / "*.nc"), table_uri]) == 1
    with pytest.raises(lance_nd.TableNotFoundError):
        lance_nd.open(table_uri)


def test_skip_errors_skips_schema_conflicts(files, table_uri):
    sea_surface("d", dtype="float32").to_netcdf(files / "d.nc")
    assert main(["create", "--skip-errors", str(files / "*.nc"), table_uri]) == 0
    assert titles(table_uri) == ["a", "b", "c"]


def test_log_file_and_quiet(files, table_uri, tmp_path, capsys):
    log = tmp_path / "run.log"
    assert main(["create", "-q", "--log-file", str(log), str(files / "*.nc"), table_uri]) == 0
    assert capsys.readouterr().err == ""
    text = log.read_text()
    assert "a.nc" in text
    assert "3 files written" in text


def test_file_attribute_selects_rows(files, table_uri):
    assert main(["create", str(files / "*.nc"), table_uri]) == 0
    rows = [row.to_xarray() for row in lance_nd.open(table_uri).scan()]
    assert [ds.attrs["source_file"] for ds in rows] == [
        (files / name).as_posix() for name in ["a.nc", "b.nc", "c.nc"]
    ]
    table = lance_nd.open(table_uri)
    assert table.delete(f"__global_attrs__.source_file = '{(files / 'b.nc').as_posix()}'") == 1
    assert titles(table_uri) == ["a", "c"]


def test_file_attribute_name_and_off(files, table_uri, tmp_path):
    assert main(["create", "--file-attribute", "path", str(files / "a.nc"), table_uri]) == 0
    (ds,) = [row.to_xarray() for row in lance_nd.open(table_uri).scan()]
    assert ds.attrs == {"title": "a", **identity(files / "a.nc", name="path")}
    other = str(tmp_path / "other.lance")
    assert main(["create", "--no-file-attribute", str(files / "a.nc"), other]) == 0
    (ds,) = [row.to_xarray() for row in lance_nd.open(other).scan()]
    assert ds.attrs == {"title": "a"}


def test_file_attribute_replaces_an_existing_attribute(files, table_uri, capsys):
    assert main(["create", "--file-attribute", "title", str(files / "a.nc"), table_uri]) == 0
    assert "replace global attribute 'title'" in capsys.readouterr().err
    assert titles(table_uri) == [(files / "a.nc").as_posix()]


def test_module_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "lance_nd", "--help"], capture_output=True, text=True, check=True
    )
    assert "create" in result.stdout and "append" in result.stdout
