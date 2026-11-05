"""Create, append to and open lance-nd tables."""

from collections.abc import Sequence
from types import TracebackType
from typing import Any, TypeGuard

import numpy as np
import xarray as xr
from nd_arrow_array import NdRecordBatch

from lance_nd import _lance_nd
from lance_nd._chunks import split_chunks
from lance_nd._typing import ArrowData, Casting, ColumnOptions, Mode, OnDrop

Table = _lance_nd.Table
Scanner = _lance_nd.Scanner


def _is_arrow(data: object) -> TypeGuard[ArrowData]:
    return hasattr(data, "__arrow_c_stream__") or hasattr(data, "__arrow_c_array__")


def _row(item: xr.Dataset | NdRecordBatch) -> NdRecordBatch:
    if isinstance(item, NdRecordBatch):
        return item
    return NdRecordBatch.from_xarray(_bytes_as_text(_without_empty_attrs(item)))


def _text(value: Any) -> Any:
    """``bytes`` and fixed-width byte arrays as UTF-8 text, other values as is."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return np.char.decode(value, "utf-8")
    return value


def _bytes_as_text(ds: xr.Dataset) -> xr.Dataset:
    """``ds`` with byte variables and attributes as text, which nd-arrow-array stores."""
    owners: list[xr.Dataset | xr.Variable] = [ds, *ds.variables.values()]
    has_bytes = any(var.dtype.kind == "S" for var in ds.variables.values()) or any(
        _text(value) is not value for owner in owners for value in owner.attrs.values()
    )
    if not has_bytes:
        return ds
    ds = ds.copy(deep=False)
    for name, var in ds.variables.items():
        if var.dtype.kind == "S":
            ds[name] = var.copy(data=_text(var.values))
    owners = [ds, *ds.variables.values()]
    for owner in owners:
        owner.attrs = {key: _text(value) for key, value in owner.attrs.items()}
    return ds


def _is_empty(value: object) -> bool:
    """``True`` for an attribute with no items, such as ``[]``."""
    if isinstance(value, np.ndarray):
        return value.ndim == 1 and value.size == 0
    return isinstance(value, list | tuple) and len(value) == 0


def _without_empty_attrs(ds: xr.Dataset) -> xr.Dataset:
    """``ds`` without empty array attributes, which hold no value."""
    owners: list[xr.Dataset | xr.Variable] = [ds, *ds.variables.values()]
    if not any(_is_empty(value) for owner in owners for value in owner.attrs.values()):
        return ds
    ds = ds.copy(deep=False)
    owners = [ds, *ds.variables.values()]
    for owner in owners:
        owner.attrs = {key: value for key, value in owner.attrs.items() if not _is_empty(value)}
    return ds


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
        data: xr.Dataset | NdRecordBatch | ArrowData | Sequence[xr.Dataset | NdRecordBatch],
        chunks: Sequence[str] | None = None,
        on_drop: OnDrop = "warn",
    ) -> None:
        """Write one or more datasets, without a commit.

        ``data`` is an xarray dataset, an ``nd_arrow_array.NdRecordBatch``,
        a list of these, or Arrow data in the nd-arrow format, such as a
        ``pyarrow.RecordBatch`` with one dataset per row. Each dataset is
        one row of the table.

        With ``chunks``, each dask chunk over those dimensions is one row.
        The writer computes and writes one chunk at a time. A variable is
        kept when all its dimensions are chunk dimensions, or when it has
        all chunk dimensions. ``on_drop`` is "warn", "raise" or "ignore" for
        the other variables.
        """
        if isinstance(data, xr.Dataset | NdRecordBatch):
            items: Sequence[Any] = [data]
        elif _is_arrow(data):
            if chunks is not None:
                raise TypeError("chunks needs xarray datasets, not Arrow data")
            self._native.write(data)
            return
        elif isinstance(data, Sequence) and not isinstance(data, str | bytes):
            items = data
        else:
            raise TypeError(
                "expected an xarray.Dataset, an NdRecordBatch, Arrow data or a list, "
                f"not {type(data).__name__}"
            )
        for item in items:
            if not isinstance(item, xr.Dataset | NdRecordBatch):
                raise TypeError(
                    f"expected an xarray.Dataset or an NdRecordBatch, not {type(item).__name__}"
                )
        if chunks is None:
            self._native.write_rows([_row(item) for item in items])
            return
        for item in items:
            if isinstance(item, NdRecordBatch):
                raise TypeError("chunks needs xarray datasets, not an NdRecordBatch")
            for part in split_chunks(item, chunks, on_drop):
                self._native.write(_row(part))

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
    storage_options: dict[str, str] | None = None,
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

    ``storage_options`` are the Lance storage options, for example
    ``aws_region`` or ``aws_endpoint``. An empty value uses the defaults of
    the environment.

    The writer buffers datasets until they have about
    ``target_fragment_bytes`` (default 128 MiB) on disk, as estimated from
    the fragments written so far, then writes them as one Lance fragment. ``write_parallelism`` (default 4) fragments are
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
        storage_options,
    )
    return Writer(native)


def append(
    uri: str,
    column_options: dict[str, ColumnOptions] | None = None,
    target_fragment_bytes: int | None = None,
    write_parallelism: int | None = None,
    casting: Casting = "no",
    storage_options: dict[str, str] | None = None,
) -> Writer:
    """A writer that appends to an existing table. See :func:`create` for
    the fragment options, ``casting`` and ``storage_options``."""
    native = _lance_nd.Writer.append(
        uri,
        column_options,
        target_fragment_bytes,
        write_parallelism,
        casting,
        storage_options,
    )
    return Writer(native)


def open(uri: str, storage_options: dict[str, str] | None = None) -> Table:
    """Open the table at ``uri``. See :func:`create` for ``storage_options``."""
    return Table.open(uri, storage_options)
