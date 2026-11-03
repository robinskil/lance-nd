"""Native classes of lance-nd."""

import datetime

import pyarrow as pa

from lance_nd._typing import ColumnOptions, Mode

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
    ) -> Writer:
        """A writer for a new table. ``mode`` is "create" or "overwrite".

        ``stats`` stores the min and max of each variable per row.
        ``target_fragment_bytes`` and ``write_parallelism`` override the
        fragment size and the number of parallel fragment writes.
        """

    @staticmethod
    def append(
        uri: str,
        column_options: dict[str, ColumnOptions] | None = None,
        target_fragment_bytes: int | None = None,
        write_parallelism: int | None = None,
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
        """Merge small fragments and remove deleted datasets.

        A fragment smaller than about ``target_fragment_bytes`` on disk
        (default 128 MiB) is merged with its neighbors. A fragment with at
        least ``deleted_fraction`` deleted datasets (default 0.1) is written
        again without them. Returns the counts ``fragments_removed``,
        ``fragments_added``, ``files_removed`` and ``files_added``.
        """

    def cleanup_old_versions(self, older_than: datetime.timedelta = ...) -> dict[str, int]:
        """Remove the files of the table versions older than ``older_than``.

        The default is 14 days. The current version always stays. Returns
        ``bytes_removed`` and ``old_versions``.
        """

def _scan_batch(uri: str) -> pa.RecordBatch:
    """The whole table as one nd-arrow record batch. For tests only."""
