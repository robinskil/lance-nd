"""Types for the options of the public API."""

from typing import Any, Literal, Protocol, TypedDict, Union

Mode = Literal["create", "overwrite"]
Casting = Literal["no", "safe", "same_kind", "promote"]
OnDrop = Literal["warn", "raise", "ignore"]


class ArrowStream(Protocol):
    """Arrow data with the Arrow C stream interface, such as a ``pyarrow.Table``."""

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any: ...


class ArrowArray(Protocol):
    """Arrow data with the Arrow C array interface, such as a ``pyarrow.RecordBatch``."""

    def __arrow_c_array__(self, requested_schema: Any = None) -> tuple[Any, Any]: ...


ArrowData = Union[ArrowStream, ArrowArray]


class ColumnOptions(TypedDict, total=False):
    """Encoding options of one column. A missing key keeps the Lance default."""

    compression: Literal["none", "lz4", "zstd"]
    #: Only with compression "zstd".
    compression_level: int
    byte_stream_split: Literal["off", "on", "auto"]
    rle_threshold: float
    structural_encoding: Literal["miniblock", "fullzip"]
