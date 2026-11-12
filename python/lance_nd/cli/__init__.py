"""The lance-nd command: write files that xarray can open to a table, one row per file."""

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import get_args

from lance_nd._typing import Casting
from lance_nd.cli import _sync, _write
from lance_nd.cli._files import FILE_ATTRIBUTE, log

__all__ = ["main"]


def _positive(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be more than 0")
    return value


def _not_negative(text: str) -> float:
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("files", nargs="+", help="files or glob patterns, such as 'data/**/*.nc'")
    common.add_argument("table", help="the table URI")
    common.add_argument(
        "--skip-variables", action="store_true", help="drop the variables that cannot be stored"
    )
    common.add_argument(
        "--skip-attributes", action="store_true", help="drop the attributes that cannot be stored"
    )
    common.add_argument(
        "--casting",
        choices=get_args(Casting),
        default="no",
        help="what to do when a type is not the table type (default: no)",
    )
    common.add_argument("--log-file", help="also write the full log to this file")
    level = common.add_mutually_exclusive_group()
    level.add_argument("-v", "--verbose", action="store_true", help="show each written file")
    level.add_argument("-q", "--quiet", action="store_true", help="show only warnings and errors")

    writes = argparse.ArgumentParser(add_help=False)
    writes.add_argument(
        "--skip-errors", action="store_true", help="skip a file that fails and continue"
    )
    source = writes.add_mutually_exclusive_group()
    source.add_argument(
        "--file-attribute",
        default=FILE_ATTRIBUTE,
        metavar="NAME",
        help=f"the global attribute that holds the file path (default: {FILE_ATTRIBUTE})",
    )
    source.add_argument(
        "--no-file-attribute",
        dest="file_attribute",
        action="store_const",
        const=None,
        help="do not add the file path, time and size to the global attributes",
    )

    parser = argparse.ArgumentParser(
        prog="lance-nd", description="Write files that xarray can open to a lance-nd table."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser(
        "create", parents=[common, writes], help="create a table, one row per file"
    )
    make.add_argument("--overwrite", action="store_true", help="replace an existing table")
    make.add_argument("--stats", action="store_true", help="store min and max per variable")
    commands.add_parser(
        "append", parents=[common, writes], help="append one row per file to a table"
    )
    sync = commands.add_parser(
        "sync",
        parents=[common],
        help="mirror the files as the rows of a table",
        description="Add, change and delete rows so that the table has one row per file.",
    )
    sync.add_argument(
        "--interval", type=_positive, metavar="SECONDS", help="sync again after each interval"
    )
    sync.add_argument(
        "--min-age",
        type=_not_negative,
        default=0.0,
        metavar="SECONDS",
        help="skip files that changed less than this time ago (default: 0)",
    )
    sync.add_argument(
        "--stats", action="store_true", help="store min and max per variable in a new table"
    )
    sync.add_argument(
        "--no-compact", dest="compact", action="store_false", help="do not compact after a change"
    )
    sync.set_defaults(file_attribute=FILE_ATTRIBUTE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command with ``argv`` (default: ``sys.argv``) and give the exit code."""
    args = _parser().parse_args(argv)
    handlers: list[logging.Handler] = []
    terminal = logging.StreamHandler(sys.stderr)
    terminal.setLevel(logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO)
    terminal.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handlers.append(terminal)
    if args.log_file:
        to_file = logging.FileHandler(args.log_file, encoding="utf-8")
        to_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        handlers.append(to_file)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    for handler in handlers:
        log.addHandler(handler)
    try:
        if args.command == "sync":
            return _sync.run(args)
        return _write.run(args)
    except KeyboardInterrupt:
        log.error("stopped, the current commit is not done")
        return 130
    finally:
        for handler in handlers:
            log.removeHandler(handler)
            handler.close()
