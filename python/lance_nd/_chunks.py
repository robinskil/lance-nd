"""Split xarray datasets into their dask chunks."""

import itertools
import warnings
from collections.abc import Iterator, Sequence

import numpy as np
import xarray as xr

from lance_nd._typing import OnDrop

_ON_DROP = ("ignore", "warn", "raise")


def split_chunks(ds: xr.Dataset, chunks: Sequence[str], on_drop: OnDrop = "warn") -> Iterator[xr.Dataset]:
    """The dask chunks of ``ds`` over the ``chunks`` dimensions, in C order.

    Each chunk is a lazy dataset. A variable is kept when all its dimensions
    are chunk dimensions, or when it has all chunk dimensions. ``on_drop`` is
    "warn", "raise" or "ignore" for the other variables.
    """
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
