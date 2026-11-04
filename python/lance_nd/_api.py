"""Create, append to and open lance-nd tables."""

from collections.abc import Sequence
from types import TracebackType

import pyarrow as pa
import xarray as xr

from lance_nd import _lance_nd
from lance_nd._encode import encode, encode_chunks
from lance_nd._typing import Casting, ColumnOptions, Mode, OnDrop

Table = _lance_nd.Table
Scanner = _lance_nd.Scanner


class Writer:
    """Writes datasets to a table and commits them all at once in ``finish()``.

    Readers see none of the written datasets before ``finish()``. As a
    context manager, the writer finishes at the end of the block, or aborts
    when the block raises.
    """

    def __init__(self, native: _lance_nd.Writer):
        self._native = native
        self._closed = False
        #: The committed table after ``finish()``, else ``None``.
        self.table: Table | None = None

    def write(
        self,
        data: xr.Dataset | Sequence[xr.Dataset] | pa.RecordBatch,
        chunks: Sequence[str] | None = None,
        on_drop: OnDrop = "warn",
        convert_calendar: bool = False,
    ) -> None:
        """Write a dataset, a list of datasets or an nd-arrow record batch.

        With ``chunks``, each dask chunk of each dataset is one row. The
        writer computes and writes one chunk at a time, see
        :func:`lance_nd.encode_chunks`. ``convert_calendar`` turns cftime
        values into ``datetime64[ns]``, see :func:`lance_nd.encode`.
        """
        if isinstance(data, pa.RecordBatch):
            if chunks is not None:
                raise TypeError("chunks needs xarray datasets, not a pyarrow.RecordBatch")
            self._native.write_batch(data)
        elif isinstance(data, xr.Dataset) or (
            isinstance(data, Sequence) and not isinstance(data, str | bytes)
        ):
            if chunks is None:
                self._native.write_batch(encode(data, convert_calendar=convert_calendar))
                return
            for ds in [data] if isinstance(data, xr.Dataset) else data:
                for batch in encode_chunks(ds, chunks, on_drop, convert_calendar):
                    self._native.write_batch(batch)
        else:
            raise TypeError(
                "expected an xarray.Dataset, a list of them or a pyarrow.RecordBatch, "
                f"not {type(data).__name__}"
            )

    def finish(self) -> Table:
        """Commit all written datasets in one new table version."""
        try:
            self.table = self._native.finish()
        finally:
            self._closed = True
        return self.table

    def abort(self) -> None:
        """Drop all written datasets without a commit."""
        try:
            self._native.abort()
        finally:
            self._closed = True

    def __enter__(self) -> "Writer":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # A finish() or abort() inside the block already closed the writer.
        if self._closed:
            return
        if exc_type is None:
            self.finish()
        else:
            self.abort()


def create(
    uri: str,
    mode: Mode = "create",
    column_defaults: ColumnOptions | None = None,
    column_options: dict[str, ColumnOptions] | None = None,
    stats: bool = False,
    target_fragment_bytes: int | None = None,
    write_parallelism: int | None = None,
    casting: Casting = "no",
) -> Writer:
    """A writer for a new table. ``mode`` is "create" or "overwrite".

    ``stats`` stores the min and max of each variable per row, so that a
    scan can skip rows. An append follows the table.

    ``casting`` decides what happens when a variable or attribute has
    another type than the table, as the numpy ``casting`` argument: "no"
    rejects the write, "safe" casts the new values when no value can change,
    such as float32 to float64, and "same_kind" also casts within a kind,
    such as float64 to float32. With these, the table type always wins.
    "promote" follows numpy type promotion: when the new values do not fit,
    the writer widens the variable in the table to the smallest type that
    holds both, see :meth:`Table.widen`. A writer first commits the
    datasets that it wrote before such a widening.

    The writer buffers datasets until they have about
    ``target_fragment_bytes`` (default 128 MiB) in memory, then writes them
    as one Lance fragment. ``write_parallelism`` (default 4) fragments are
    written at the same time.
    """
    native = _lance_nd.Writer.create(
        uri,
        mode,
        column_defaults,
        column_options,
        stats,
        target_fragment_bytes,
        write_parallelism,
        casting,
    )
    return Writer(native)


def append(
    uri: str,
    column_options: dict[str, ColumnOptions] | None = None,
    target_fragment_bytes: int | None = None,
    write_parallelism: int | None = None,
    casting: Casting = "no",
) -> Writer:
    """A writer that appends to an existing table. See :func:`create` for
    the fragment options and ``casting``."""
    native = _lance_nd.Writer.append(
        uri, column_options, target_fragment_bytes, write_parallelism, casting
    )
    return Writer(native)


def open(uri: str) -> Table:
    """Open the table at ``uri``."""
    return Table.open(uri)
