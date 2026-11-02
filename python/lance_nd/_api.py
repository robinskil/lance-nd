"""Create, append to and open lance-nd tables."""

from collections.abc import Sequence

import pyarrow as pa
import xarray as xr

from lance_nd import _lance_nd
from lance_nd._encode import encode

Table = _lance_nd.Table


class Writer:
    """Writes datasets to a table and commits them all at once in ``finish()``.

    Readers see none of the written datasets before ``finish()``. As a
    context manager, the writer finishes at the end of the block, or aborts
    when the block raises.
    """

    def __init__(self, native: _lance_nd.Writer):
        self._native = native

    def write(self, data: xr.Dataset | Sequence[xr.Dataset] | pa.RecordBatch) -> None:
        """Write a dataset, a list of datasets or an nd-arrow record batch."""
        if isinstance(data, pa.RecordBatch):
            batch = data
        elif isinstance(data, xr.Dataset) or (
            isinstance(data, Sequence) and not isinstance(data, str | bytes)
        ):
            batch = encode(data)
        else:
            raise TypeError(
                "expected an xarray.Dataset, a list of them or a pyarrow.RecordBatch, "
                f"not {type(data).__name__}"
            )
        self._native.write_batch(batch)

    def finish(self) -> Table:
        """Commit all written datasets in one new table version."""
        return self._native.finish()

    def abort(self) -> None:
        """Drop all written datasets without a commit."""
        self._native.abort()

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.finish()
        else:
            self.abort()


def create(
    uri: str,
    mode: str = "create",
    column_defaults: dict | None = None,
    column_options: dict | None = None,
) -> Writer:
    """A writer for a new table. ``mode`` is "create" or "overwrite"."""
    return Writer(_lance_nd.Writer.create(uri, mode, column_defaults, column_options))


def append(uri: str, column_options: dict | None = None) -> Writer:
    """A writer that appends to an existing table."""
    return Writer(_lance_nd.Writer.append(uri, column_options))


def open(uri: str) -> Table:
    """Open the table at ``uri``."""
    return Table.open(uri)
