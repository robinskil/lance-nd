"""Drop the variables and attributes that the converter rejects."""

from typing import Any

import xarray as xr

from lance_nd._api import _row
from lance_nd.cli._files import log


def converts(ds: xr.Dataset) -> bool:
    try:
        _row(ds)
    except TypeError:
        return False
    return True


def _good_attrs(attrs: dict[Any, Any], owner: str, path: str) -> tuple[dict[Any, Any], int]:
    kept = {}
    for key, value in attrs.items():
        if converts(xr.Dataset(attrs={key: value})):
            kept[key] = value
        else:
            log.warning("%s: drop attribute '%s' of %s (%s)", path, key, owner, type(value).__name__)
    return kept, len(attrs) - len(kept)


def clean(
    ds: xr.Dataset, skip_variables: bool, skip_attributes: bool, path: str
) -> tuple[xr.Dataset, int, int]:
    """``ds`` without the rejected parts, and the numbers of dropped variables and attributes."""
    ds = ds.copy()
    attributes = 0
    if skip_attributes:
        ds.attrs, dropped = _good_attrs(ds.attrs, "the dataset", path)
        attributes += dropped
        for name, var in ds.variables.items():
            var.attrs, dropped = _good_attrs(var.attrs, f"variable '{name}'", path)
            attributes += dropped
    bad: list[Any] = []
    if skip_variables:
        bad = [name for name, var in ds.variables.items() if not converts(xr.Dataset({name: var}))]
        for name in bad:
            log.warning("%s: drop variable '%s' (%s)", path, name, ds.variables[name].dtype)
        ds = ds.drop_vars(bad)
        if bad and not ds.data_vars:
            raise ValueError("no variable is left after the drops")
    return ds, len(bad), attributes
