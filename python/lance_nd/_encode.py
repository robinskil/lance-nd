"""Encode xarray datasets as nd-arrow record batches."""

import itertools
import warnings
from collections.abc import Hashable, Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

from lance_nd._typing import OnDrop

GLOBAL_ATTRS = "__global_attrs__"
# Each name that starts with it is reserved for the stats columns.
STATS_PREFIX = "__stats__"
ND_ARRAY_METADATA = {
    b"ARROW:extension:name": b"nd_arrow.array",
    b"ARROW:extension:metadata": b'{"version":1}',
}
GLOBAL_ATTRIBUTES_METADATA = {
    b"ARROW:extension:name": b"nd_arrow.global_attributes",
    b"ARROW:extension:metadata": b'{"version":1}',
}
_DIM_NAMES = pa.list_(pa.field("item", pa.string(), nullable=False))
_DIM_SIZES = pa.list_(pa.field("item", pa.uint32(), nullable=False))
# Version 1 uses List with 32-bit offsets.
_MAX_VALUES = 2**31 - 1
_ON_DROP = ("ignore", "warn", "raise")

# The values, dimension names, shape and attributes of one variable.
_Variable = tuple[pa.Array, list[str], tuple[int, ...], dict[Hashable, Any]]


def encode(
    datasets: xr.Dataset | Sequence[xr.Dataset],
    chunks: Sequence[str] | None = None,
    on_drop: OnDrop = "warn",
    convert_calendar: bool = False,
) -> pa.RecordBatch:
    """Encode one dataset or a list of datasets, one row per dataset.

    Each variable and coordinate is one ``nd_arrow.array`` column. A dataset
    that does not have a variable gets a null row in that column. The
    dataset attributes go into the ``__global_attrs__`` column.

    With ``chunks``, each dask chunk over those dimensions is one row, see
    :func:`encode_chunks`. This loads all chunks.

    xarray decodes a calendar that ``datetime64`` cannot hold, such as a
    Julian one, into cftime objects. ``convert_calendar`` turns them into
    ``datetime64[ns]`` values of the same instant: a Julian 1973-02-25
    becomes the Gregorian 1973-03-10. Without it, cftime values raise
    ``TypeError``.
    """
    datasets = _datasets(datasets)
    if chunks is not None:
        datasets = [part for ds in datasets for part in _chunk_datasets(ds, chunks, on_drop)]
    rows = [_dataset_row(ds, convert_calendar) for ds in datasets]
    names = list(dict.fromkeys(name for variables, _ in rows for name in variables))
    fields = []
    columns = []
    for name in names:
        column = _column(name, [variables.get(name) for variables, _ in rows])
        fields.append(pa.field(name, column.type, nullable=True, metadata=ND_ARRAY_METADATA))
        columns.append(column)
    global_attrs = [attrs for _, attrs in rows]
    if any(global_attrs):
        column = _attrs_struct(global_attrs, owner="the dataset")
        fields.append(
            pa.field(GLOBAL_ATTRS, column.type, nullable=True, metadata=GLOBAL_ATTRIBUTES_METADATA)
        )
        columns.append(column)
    return pa.RecordBatch.from_arrays(columns, schema=pa.schema(fields))


def encode_chunks(
    ds: xr.Dataset,
    chunks: Sequence[str],
    on_drop: OnDrop = "warn",
    convert_calendar: bool = False,
) -> Iterator[pa.RecordBatch]:
    """Encode each dask chunk of ``ds`` over the ``chunks`` dimensions as one row.

    Yields one record batch per chunk and computes one chunk at a time.
    A variable is kept when all its dimensions are chunk dimensions, or when
    it has all chunk dimensions. ``on_drop`` is "warn", "raise" or "ignore"
    for the other variables. See :func:`encode` for ``convert_calendar``.
    """
    for part in _chunk_datasets(ds, chunks, on_drop):
        yield encode(part, convert_calendar=convert_calendar)


def _datasets(datasets: xr.Dataset | Sequence[xr.Dataset]) -> list[xr.Dataset]:
    if isinstance(datasets, xr.Dataset):
        datasets = [datasets]
    datasets = list(datasets)
    for ds in datasets:
        if not isinstance(ds, xr.Dataset):
            raise TypeError(f"expected an xarray.Dataset, not {type(ds).__name__}")
    return datasets


def _chunk_datasets(ds: xr.Dataset, chunks: Sequence[str], on_drop: OnDrop) -> Iterator[xr.Dataset]:
    """The chunks of ``ds`` in C order, each a lazy dataset of the kept variables."""
    if on_drop not in _ON_DROP:
        raise ValueError(f"on_drop must be one of {_ON_DROP}, not '{on_drop}'")
    chunks = [str(name) for name in chunks]
    sizes = ds.chunks
    for name in chunks:
        if name not in sizes:
            raise ValueError(f"'{name}' is not a dask chunked dimension of the dataset")
    grid = set(chunks)
    dropped = sorted(
        str(name)
        for name, var in ds.variables.items()
        if not (set(var.dims) <= grid or set(var.dims) >= grid)
    )
    if len(dropped) == len(ds.variables):
        raise ValueError(f"no variable fits the chunk dimensions {chunks}")
    if dropped and on_drop != "ignore":
        message = f"variables outside the chunk dimensions {chunks}: {', '.join(dropped)}"
        if on_drop == "raise":
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=3)
    kept = ds.drop_vars(dropped)
    slices = []
    for name in chunks:
        bounds = np.cumsum((0, *sizes[name]))
        slices.append([slice(int(a), int(b)) for a, b in itertools.pairwise(bounds)])
    for parts in itertools.product(*slices):
        yield kept.isel(dict(zip(chunks, parts, strict=True)))


def _dataset_row(
    ds: xr.Dataset, convert_calendar: bool = False
) -> tuple[dict[str, _Variable], dict[Hashable, Any]]:
    """The encoded variables of one dataset, and its global attributes."""
    if "coordinates" in ds.attrs:
        raise ValueError("the dataset already has a 'coordinates' attribute")
    coordinates = _cf_coordinates(ds)
    listed = {name for names in coordinates.values() for name in names}
    variables = {}
    for name, var in ds.variables.items():
        name = str(name)
        if name == GLOBAL_ATTRS or name.startswith(STATS_PREFIX):
            raise ValueError(
                f"the variable name '{name}' is reserved, also each name that "
                f"starts with {STATS_PREFIX}"
            )
        if "coordinates" in var.attrs:
            raise ValueError(f"variable '{name}' already has a 'coordinates' attribute")
        attrs = _non_empty(var.attrs)
        if coordinates.get(name):
            attrs["coordinates"] = " ".join(coordinates[name])
        values = _values(name, var, convert_calendar)
        variables[name] = (values, [str(d) for d in var.dims], var.shape, attrs)
    attrs = _non_empty(ds.attrs)
    unlisted = [name for name in _other_coordinates(ds) if name not in listed]
    if unlisted:
        attrs["coordinates"] = " ".join(unlisted)
    return variables, attrs


def _other_coordinates(ds: xr.Dataset) -> list[str]:
    """Coordinates that are not a 1-D coordinate with the name of their dimension."""
    return [str(name) for name, var in ds.coords.items() if var.dims != (name,)]


def _cf_coordinates(ds: xr.Dataset) -> dict[str, list[str]]:
    """For each data variable, the other coordinates that fit in its dimensions."""
    others = _other_coordinates(ds)
    result = {}
    for name, var in ds.data_vars.items():
        dims = set(var.dims)
        result[str(name)] = [c for c in others if set(ds.coords[c].dims) <= dims]
    return result


def _values(name: str, var: xr.Variable, convert_calendar: bool = False) -> pa.Array:
    """The flat values of a variable in C order, with nulls for missing values."""
    data = np.asarray(var.values)
    if data.dtype.kind == "O" and _holds_cftime(data):
        if not convert_calendar:
            raise TypeError(
                f"variable '{name}' holds cftime values, because xarray cannot decode "
                "its calendar into datetime64. Pass convert_calendar=True for the "
                "same instants as datetime64[ns], or open the files with "
                "decode_times=False to keep the raw numbers"
            )
        data = _cftime_to_datetime64(name, data)
    if data.dtype.kind in "biufMm" and not data.dtype.isnative:
        data = data.astype(data.dtype.newbyteorder("="))
    flat = np.ascontiguousarray(data).reshape(-1)
    kind = data.dtype.kind
    if kind in "biufMm":
        return pa.array(flat, from_pandas=True)
    if kind == "U":
        return pa.array(flat.tolist(), type=pa.string())
    if kind == "S":
        return pa.array([v.decode("utf-8") for v in flat.tolist()], type=pa.string())
    if kind == "O":
        # xarray can store a missing string as NaN, which from_pandas makes null.
        try:
            array = pa.array(flat, from_pandas=True)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            array = None
        if array is not None and (pa.types.is_string(array.type) or pa.types.is_null(array.type)):
            return array.cast(pa.string())
    raise TypeError(f"variable '{name}' has the unsupported dtype {data.dtype}")


# The range of datetime64[ns]. A value outside it wraps without an error.
_NS_MIN = np.datetime64("1677-09-21T00:12:44", "us")
_NS_MAX = np.datetime64("2262-04-11T23:47:16", "us")


def _holds_cftime(data: np.ndarray) -> bool:
    """``True`` when the first present value of an object array is a cftime."""
    flat = data.reshape(-1)
    missing = pd.isna(flat)
    if missing.all():
        return False
    sample = flat[int(np.argmin(missing))]
    return type(sample).__module__.split(".")[0] == "cftime"


def _cftime_to_datetime64(name: str, data: np.ndarray) -> np.ndarray:
    """The same instants as the cftime values of ``data``, as ``datetime64[ns]``.

    cftime maps between two real-world calendars through the Julian day, so
    each instant stays exact. One ``date2num`` call converts all values; a
    call per value costs about three times more.
    """
    import cftime

    flat = data.reshape(-1)
    missing = np.asarray(pd.isna(flat), dtype=bool)
    present = flat[~missing]
    calendar = present[0].calendar
    try:
        # The Unix epoch, written on the source calendar: the time since it
        # is the time since the Gregorian epoch.
        epoch = cftime.datetime(1970, 1, 1, calendar="proleptic_gregorian").change_calendar(
            calendar
        )
    except ValueError as error:
        raise ValueError(
            f"variable '{name}' uses the '{calendar}' calendar, which has no real "
            "instants, so no exact Gregorian date exists. Open the files with "
            "decode_times=False to keep the raw numbers"
        ) from error
    units = "microseconds since " + epoch.strftime("%Y-%m-%d %H:%M:%S")
    micros = np.rint(np.asarray(cftime.date2num(present, units, calendar=calendar), "float64"))
    stamps = np.datetime64(0, "us") + micros.astype("int64").astype("timedelta64[us]")
    outside = (stamps < _NS_MIN) | (stamps > _NS_MAX)
    if outside.any():
        raise ValueError(
            f"variable '{name}' has {int(outside.sum())} date(s) outside the range of "
            f"datetime64[ns], such as {stamps[outside][0]}. The range is {_NS_MIN} to "
            f"{_NS_MAX}. Open the files with decode_times=False to keep the raw numbers"
        )
    out = np.full(flat.size, np.datetime64("NaT", "us"), dtype="datetime64[us]")
    out[~missing] = stamps
    return out.reshape(data.shape).astype("datetime64[ns]")


def _column(name: str, rows: list[_Variable | None]) -> pa.StructArray:
    """One nd column. A row is None when the dataset does not have the variable."""
    types = {row[0].type for row in rows if row is not None}
    if len(types) > 1:
        found = ", ".join(sorted(str(t) for t in types))
        raise TypeError(f"variable '{name}' has more than one value type: {found}")
    value_type = types.pop()
    present = [row for row in rows if row is not None]
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([0 if row is None else len(row[0]) for row in rows])
    if offsets[-1] > _MAX_VALUES:
        raise ValueError(
            f"variable '{name}' has {offsets[-1]} values over all datasets, "
            f"more than the format allows in one batch ({_MAX_VALUES}); write fewer datasets at once"
        )
    flat = pa.concat_arrays([row[0] for row in present])
    values = pa.ListArray.from_arrays(
        pa.array(offsets.astype(np.int32)), flat, type=pa.list_(pa.field("item", value_type, nullable=True))
    )
    dim_names = pa.array([[] if row is None else row[1] for row in rows], type=_DIM_NAMES)
    dim_sizes = pa.array([[] if row is None else list(row[2]) for row in rows], type=_DIM_SIZES)
    arrays = [values, dim_names, dim_sizes]
    fields = [
        pa.field("values", values.type, nullable=False),
        pa.field("dim_names", _DIM_NAMES, nullable=False),
        pa.field("dim_sizes", _DIM_SIZES, nullable=False),
    ]
    row_attrs = [{} if row is None else row[3] for row in rows]
    if any(row_attrs):
        attrs = _attrs_struct(row_attrs, owner=f"variable '{name}'")
        arrays.append(attrs)
        fields.append(pa.field("attrs", attrs.type, nullable=False))
    mask = pa.array([row is None for row in rows])
    return pa.StructArray.from_arrays(arrays, fields=fields, mask=mask)


def _attrs_struct(rows: list[dict[Hashable, Any]], owner: str) -> pa.StructArray:
    """A struct with one child per key. A row without the key has a null."""
    encoded = [{str(k): _attr(str(k), v) for k, v in row.items()} for row in rows]
    keys = list(dict.fromkeys(key for row in encoded for key in row))
    arrays = []
    for key in keys:
        types = {row[key][0] for row in encoded if key in row}
        if len(types) > 1:
            found = ", ".join(sorted(str(t) for t in types))
            raise TypeError(f"attribute '{key}' of {owner} has more than one type: {found}")
        arrow_type = types.pop()
        values = [row[key][1] if key in row else None for row in encoded]
        arrays.append(pa.array(values, type=arrow_type))
    return pa.StructArray.from_arrays(arrays, names=keys)


def _non_empty(attrs: dict[Hashable, Any]) -> dict[Hashable, Any]:
    """``attrs`` without empty arrays, which hold no value."""
    return {key: value for key, value in attrs.items() if not _is_empty(value)}


def _is_empty(value: Any) -> bool:
    """``True`` for an attribute with no items, such as ``[]``."""
    if isinstance(value, np.ndarray):
        return value.ndim == 1 and value.size == 0
    return isinstance(value, list | tuple) and len(value) == 0


def _attr(key: str, value: Any) -> tuple[pa.DataType, Any]:
    """The Arrow type and the Python value of one attribute."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise TypeError(f"attribute '{key}' must be 1-D, not {value.ndim}-D")
        items = [_scalar(key, v) for v in value.tolist()] if value.dtype.kind in "OUS" else None
        if items is None:
            data = value if value.dtype.isnative else value.astype(value.dtype.newbyteorder("="))
            array = pa.array(data)
            return pa.list_(array.type), array.to_pylist()
        return _scalar_list(key, items)
    if isinstance(value, list | tuple):
        return _scalar_list(key, [_scalar(key, v) for v in value])
    return _scalar(key, value)


def _scalar_list(key: str, items: list[tuple[pa.DataType, Any]]) -> tuple[pa.DataType, list[Any]]:
    types = {t for t, _ in items}
    if len(types) != 1:
        raise TypeError(f"attribute '{key}' must be a list of one scalar type")
    return pa.list_(types.pop()), [v for _, v in items]


def _scalar(key: str, value: Any) -> tuple[pa.DataType, Any]:
    if isinstance(value, str):
        return pa.string(), value
    if isinstance(value, bytes):
        return pa.string(), value.decode("utf-8")
    if isinstance(value, bool | np.bool_):
        return pa.bool_(), bool(value)
    if isinstance(value, np.generic):
        array = pa.array(np.asarray([value]))
        return array.type, array[0].as_py()
    if isinstance(value, int):
        return pa.int64(), value
    if isinstance(value, float):
        return pa.float64(), value
    raise TypeError(f"attribute '{key}' has the unsupported type {type(value).__name__}")
