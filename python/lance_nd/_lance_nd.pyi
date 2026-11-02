"""Native classes of lance-nd."""

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
    ) -> Writer:
        """A writer for a new table. ``mode`` is "create" or "overwrite"."""

    @staticmethod
    def append(uri: str, column_options: dict[str, ColumnOptions] | None = None) -> Writer:
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

    def count_rows(self) -> int:
        """The number of datasets in the table."""

    def variables(self) -> list[str]:
        """The names of the variables in the table."""

def _scan_batch(uri: str) -> pa.RecordBatch:
    """The whole table as one nd-arrow record batch. For tests only."""
