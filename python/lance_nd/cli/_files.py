"""Find the input files and add their identity to the global attributes."""

import glob
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePath

import xarray as xr

log = logging.getLogger("lance_nd.cli")

FILE_ATTRIBUTE = "source_file"
MTIME_ATTRIBUTE = "source_mtime"
SIZE_ATTRIBUTE = "source_size"


@dataclass(frozen=True)
class FileInfo:
    """A file to write. ``name`` is its path with ``/`` separators."""

    path: str
    name: str
    mtime_ns: int
    size: int


def paths(sources: Sequence[str]) -> list[str]:
    """The sorted files of ``sources``. A source with glob characters expands."""
    found: set[str] = set()
    for source in sources:
        if glob.has_magic(source):
            found.update(path for path in glob.glob(source, recursive=True) if os.path.isfile(path))
        else:
            found.add(source)
    return sorted(found)


def stat(path: str) -> FileInfo:
    result = os.stat(path)
    return FileInfo(path, PurePath(path).as_posix(), result.st_mtime_ns, result.st_size)


def open_dataset(path: str) -> xr.Dataset:
    return xr.open_dataset(path)


def add_identity(ds: xr.Dataset, info: FileInfo, attribute: str) -> None:
    """Put the path, modification time and size of ``info`` in the global attributes."""
    for key, value in (
        (attribute, info.name),
        (MTIME_ATTRIBUTE, info.mtime_ns),
        (SIZE_ATTRIBUTE, info.size),
    ):
        old = ds.attrs.get(key)
        if old is not None and old != value:
            log.warning("%s: replace global attribute '%s' (%r)", info.path, key, old)
        ds.attrs[key] = value
