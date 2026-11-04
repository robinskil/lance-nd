"""Native classes of lance-nd."""

import datetime
from collections.abc import Iterator

import pyarrow as pa

from lance_nd._typing import Casting, ColumnOptions, Mode

class LanceNdError(Exception):
    """An error from lance-nd or Lance."""

class Writer:
    """A writer that makes all its datasets visible in one commit at ``finish()``."""

    @staticmethod
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

        ``stats`` stores the min and max of each variable per row.
        ``target_fragment_bytes`` and ``write_parallelism`` override the
        fragment size and the number of parallel fragment writes.
        ``casting`` is "no", "safe", "same_kind" or "promote", see
        ``lance_nd.create``.
        """

    @staticmethod
    def append(
        uri: str,
        column_options: dict[str, ColumnOptions] | None = None,
        target_fragment_bytes: int | None = None,
        write_parallelism: int | None = None,
        casting: Casting = "no",
    ) -> Writer:
        """A writer that appends to an existing table."""

    def write_batch(self, batch: pa.RecordBatch) -> None:
        """Write the datasets of an nd-arrow record batch, one per row, without a commit."""

    def finish(self) -> Table:
        """Commit all written datasets in one new table version."""

    def abort(self) -> None:
        """Drop all written datasets without a commit."""

class Table:
    """A lance-nd table at one version."""

    @staticmethod
    def open(uri: str) -> Table:
        """Open the table at ``uri``."""

    @property
    def uri(self) -> str:
        """The uri of the table."""

    @property
    def version(self) -> int:
        """The Lance version of the table."""

    @property
    def has_stats(self) -> bool:
        """``True`` when the table stores the min and max of each variable per row."""

    @property
    def num_fragments(self) -> int:
        """The number of Lance fragments of the table."""

    def count_rows(self) -> int:
        """The number of datasets in the table."""

    def variables(self) -> list[str]:
        """The names of the variables in the table."""

    def scan(
        self,
        project: list[str] | None = None,
        filter: str | None = None,
        prune: str | None = None,
        limit: int | None = None,
        batch_size: int | None = None,
        batch_size_bytes: int | None = None,
    ) -> Scanner:
        """Read the datasets of the table as a stream of nd record batches.

        ``project`` takes whole variable names. Each batch keeps the global
        attributes, so an empty list reads only the global attributes.
        ``filter`` is an exact Lance SQL filter, for example
        ``__global_attrs__.title = 'a'``. ``prune`` is a Lance SQL filter on
        values, for example ``sst.values > 25``: it uses the stats to skip
        datasets that surely do not match, and a kept dataset comes back
        whole. ``batch_size`` and ``batch_size_bytes`` limit the Lance
        batches that the scan reads.
        """

    def widen(self, variable: str, to: pa.DataType) -> None:
        """Change the value type of ``variable`` to the wider type ``to``.

        For example ``table.widen("sst", pa.float64())`` for float32 values.
        ``to`` must hold each old value exactly, as ``casting="safe"``. The
        stats get the new type too. lance-nd writes the variable and the
        stats column again, not the whole table.
        """

    def delete(self, predicate: str) -> int:
        """Delete the datasets that match ``predicate``, a Lance SQL filter.

        Lance gets ``predicate`` with no change, so it must be exact for a
        dataset, for example ``__global_attrs__.title = 'a'`` or
        ``sst.attrs.units = 'K'``. Returns the number of deleted datasets.
        """

    def compact(
        self,
        target_fragment_bytes: int | None = None,
        deleted_fraction: float | None = None,
    ) -> dict[str, int]:
        """Merge small fragments, split large fragments and remove deleted datasets.

        The target is ``target_fragment_bytes`` on disk (default 128 MiB).
        A fragment below half the target is merged with its small neighbors.
        A fragment above twice the target is split. A fragment with at least
        ``deleted_fraction`` deleted datasets (default 0.1) is written again
        without them. Returns the counts ``fragments_removed``,
        ``fragments_added``, ``files_removed`` and ``files_added``.
        """

    def cleanup_old_versions(self, older_than: datetime.timedelta = ...) -> dict[str, int]:
        """Remove the files of the table versions older than ``older_than``.

        The default is 14 days. The current version always stays. Returns
        ``bytes_removed`` and ``old_versions``.
        """

class Scanner(Iterator[pa.RecordBatch]):
    """A stream of nd record batches, one per dataset of the table.

    Each item is a record batch of one row: one nd column per variable of
    the dataset, and the ``__global_attrs__`` column.
    """

    def __iter__(self) -> Scanner: ...
    def __next__(self) -> pa.RecordBatch: ...

def _scan_batch(uri: str) -> pa.RecordBatch:
    """The whole table as one nd-arrow record batch. For tests only."""
