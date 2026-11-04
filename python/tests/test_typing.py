import textwrap

import pytest

api = pytest.importorskip("mypy.api")

GOOD = """
import datetime

import lance_nd
import xarray as xr

writer = lance_nd.create(
    "t.lance",
    mode="overwrite",
    column_options={"sst": {"compression": "zstd"}},
    stats=True,
    target_fragment_bytes=64 << 20,
    write_parallelism=2,
    casting="safe",
)
writer.write(xr.Dataset(), chunks=["time"], on_drop="ignore")
table = writer.finish()
version: int = table.version
names: list[str] = lance_nd.open("t.lance").variables()
flags: tuple[bool, int] = (table.has_stats, table.num_fragments)
deleted: int = table.delete("__global_attrs__.title = 'a'")
compacted: dict[str, int] = table.compact(target_fragment_bytes=64 << 20, deleted_fraction=0.2)
removed: dict[str, int] = table.cleanup_old_versions(datetime.timedelta(days=1))
scanner: lance_nd.Scanner = table.scan(project=["sst"], filter="sst.attrs.units = 'K'", prune="sst.values > 1", limit=10)
for batch in scanner:
    rows: int = batch.num_rows
"""

BAD = """
import lance_nd

lance_nd.create("t.lance", mode="replace")
lance_nd.append("t.lance", casting="unsafe")
lance_nd.append("t.lance", column_options={"sst": {"compression": "gzip"}})
count: str = lance_nd.open("t.lance").count_rows()
"""


def check(tmp_path, source):
    path = tmp_path / "usage.py"
    path.write_text(textwrap.dedent(source))
    stdout, _, status = api.run([str(path), "--strict", "--ignore-missing-imports", "--no-incremental"])
    return stdout, status


def test_typed_usage_passes(tmp_path):
    stdout, status = check(tmp_path, GOOD)
    assert status == 0, stdout


def test_wrong_usage_fails(tmp_path):
    stdout, status = check(tmp_path, BAD)
    assert status == 1
    assert "usage.py:4" in stdout and "replace" in stdout
    assert "usage.py:5" in stdout and "gzip" in stdout
    assert "usage.py:6" in stdout and '"int"' in stdout
