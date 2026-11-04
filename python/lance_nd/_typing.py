"""Types for the options of the public API."""

from typing import Literal, TypedDict

Mode = Literal["create", "overwrite"]
Casting = Literal["no", "safe", "same_kind"]
OnDrop = Literal["warn", "raise", "ignore"]


class ColumnOptions(TypedDict, total=False):
    """Encoding options of one column. A missing key keeps the Lance default."""

    compression: Literal["none", "lz4", "zstd"]
    #: Only with compression "zstd".
    compression_level: int
    byte_stream_split: Literal["off", "on", "auto"]
    rle_threshold: float
    structural_encoding: Literal["miniblock", "fullzip"]
