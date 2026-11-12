"""The create and append commands: one row per file."""

import argparse
from dataclasses import dataclass

import xarray as xr

from lance_nd._api import Writer, append, create
from lance_nd.cli import _files
from lance_nd.cli._clean import clean
from lance_nd.cli._files import FileInfo, log

_ENGINE_HINT = "install a netCDF engine for xarray, for example: pip install netCDF4"


@dataclass
class Dropped:
    variables: int = 0
    attributes: int = 0


def reason(error: Exception) -> str:
    if isinstance(error, ValueError) and "IO backends" in str(error):
        engines = set(xr.backends.list_engines()) - {"store"}
        if engines:
            return f"no installed xarray engine ({', '.join(sorted(engines))}) can read the file"
        return f"xarray has no engine for the file, {_ENGINE_HINT}"
    return f"{type(error).__name__}: {error}"


def write_file(writer: Writer, info: FileInfo, args: argparse.Namespace, dropped: Dropped) -> None:
    """Write the file of ``info`` as one row."""
    with _files.open_dataset(info.path) as ds:
        if args.file_attribute is not None:
            _files.add_identity(ds, info, args.file_attribute)
        try:
            writer.write(ds)
        except TypeError:
            # A TypeError comes from the conversion, before the writer gets the row.
            if not (args.skip_variables or args.skip_attributes):
                raise
            ds, variables, attributes = clean(ds, args.skip_variables, args.skip_attributes, info.path)
            dropped.variables += variables
            dropped.attributes += attributes
            writer.write(ds)


def run(args: argparse.Namespace) -> int:
    """Run create or append, and give the exit code."""
    paths = _files.paths(args.files)
    if not paths:
        log.error("no file matches %s", " ".join(args.files))
        return 1
    try:
        if args.command == "create":
            writer = create(
                args.table,
                mode="overwrite" if args.overwrite else "create",
                stats=args.stats,
                casting=args.casting,
            )
        else:
            writer = append(args.table, casting=args.casting)
    except Exception as error:
        log.error("%s: %s", args.table, reason(error))
        return 1
    dropped = Dropped()
    written = skipped = 0
    try:
        for path in paths:
            try:
                write_file(writer, _files.stat(path), args, dropped)
            except Exception as error:
                if not args.skip_errors:
                    raise
                skipped += 1
                log.warning("%s: skip file, %s", path, reason(error))
                continue
            written += 1
            log.debug("%s: written", path)
    except Exception as error:
        writer.abort()
        log.error("%s: %s", path, reason(error))
        log.error("nothing is committed")
        return 1
    if written == 0:
        writer.abort()
        log.error("no file is written, nothing is committed")
        return 1
    try:
        writer.finish()
    except Exception as error:
        log.error("%s: commit failed, %s", args.table, reason(error))
        return 1
    log.info(
        "%d files written, %d files skipped, %d variables dropped, %d attributes dropped",
        written,
        skipped,
        dropped.variables,
        dropped.attributes,
    )
    return 0
