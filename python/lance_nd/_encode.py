"""Encode xarray datasets as nd-arrow record batches."""

from collections.abc import Sequence

import numpy as np
import pyarrow as pa
import xarray as xr

GLOBAL_ATTRS = "__global_attrs__"
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


def encode(datasets: xr.Dataset | Sequence[xr.Dataset]) -> pa.RecordBatch:
    """Encode one dataset or a list of datasets, one row per dataset.

    Each variable and coordinate is one ``nd_arrow.array`` column. A dataset
    that does not have a variable gets a null row in that column. The
    dataset attributes go into the ``__global_attrs__`` column.
    """
    if isinstance(datasets, xr.Dataset):
        datasets = [datasets]
    datasets = list(datasets)
    for ds in datasets:
        if not isinstance(ds, xr.Dataset):
            raise TypeError(f"expected an xarray.Dataset, not {type(ds).__name__}")

    rows = [_dataset_row(ds) for ds in datasets]
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


def _dataset_row(ds: xr.Dataset) -> tuple[dict, dict]:
    """The encoded variables of one dataset, and its global attributes."""
    if "coordinates" in ds.attrs:
        raise ValueError("the dataset already has a 'coordinates' attribute")
    coordinates = _cf_coordinates(ds)
    listed = {name for names in coordinates.values() for name in names}
    variables = {}
    for name, var in ds.variables.items():
        name = str(name)
        if name == GLOBAL_ATTRS:
            raise ValueError(f"the variable name '{GLOBAL_ATTRS}' is reserved")
        if "coordinates" in var.attrs:
            raise ValueError(f"variable '{name}' already has a 'coordinates' attribute")
        attrs = dict(var.attrs)
        if coordinates.get(name):
            attrs["coordinates"] = " ".join(coordinates[name])
        variables[name] = (_values(name, var), [str(d) for d in var.dims], var.shape, attrs)
    attrs = dict(ds.attrs)
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


def _values(name: str, var: xr.Variable) -> pa.Array:
    """The flat values of a variable in C order, with nulls for missing values."""
    data = np.asarray(var.values)
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
        # xarray can store a missing string as NaN.
        items = [None if isinstance(v, float) and np.isnan(v) else v for v in flat.tolist()]
        if all(v is None or isinstance(v, str) for v in items):
            return pa.array(items, type=pa.string())
    raise TypeError(f"variable '{name}' has the unsupported dtype {data.dtype}")


def _column(name: str, rows: list) -> pa.StructArray:
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


def _attrs_struct(rows: list[dict], owner: str) -> pa.StructArray:
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


def _attr(key: str, value) -> tuple[pa.DataType, object]:
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


def _scalar_list(key: str, items: list) -> tuple[pa.DataType, list]:
    types = {t for t, _ in items}
    if len(types) != 1:
        raise TypeError(f"attribute '{key}' must be a list of one scalar type")
    return pa.list_(types.pop()), [v for _, v in items]


def _scalar(key: str, value) -> tuple[pa.DataType, object]:
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
