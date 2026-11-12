"""The sync command: mirror the files of a pattern as the rows of a table."""

import argparse
import logging
import signal
import threading
import time
from types import FrameType

import lance_nd
from lance_nd._api import append, create
from lance_nd.cli import _files
from lance_nd.cli._files import FILE_ATTRIBUTE, MTIME_ATTRIBUTE, SIZE_ATTRIBUTE, FileInfo, log
from lance_nd.cli._write import Dropped, reason, write_file

_FILE = f"__global_attrs__.{FILE_ATTRIBUTE}"
_MTIME = f"__global_attrs__.{MTIME_ATTRIBUTE}"
_SIZE = f"__global_attrs__.{SIZE_ATTRIBUTE}"

Rows = dict[str, set[tuple[object, object]]]


def _quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _rows(uri: str) -> Rows | None:
    """The (mtime, size) pairs of the rows of each file, or None without a table."""
    try:
        table = lance_nd.open(uri)
    except lance_nd.TableNotFoundError:
        return None
    rows: Rows = {}
    for batch in table.scan(project=[]):
        attrs = batch.attrs
        name = attrs.get(FILE_ATTRIBUTE)
        if isinstance(name, str):
            rows.setdefault(name, set()).add((attrs.get(MTIME_ATTRIBUTE), attrs.get(SIZE_ATTRIBUTE)))
    return rows


def _stale(info: FileInfo) -> str:
    """A filter for the rows of the file of ``info`` with another mtime or size."""
    return (
        f"({_FILE} = {_quote(info.name)} AND ({_MTIME} IS NULL OR {_SIZE} IS NULL"
        f" OR {_MTIME} <> {info.mtime_ns} OR {_SIZE} <> {info.size}))"
    )


def cycle(args: argparse.Namespace) -> bool:
    """Sync one time. Returns False when a file failed."""
    existing = _rows(args.table)
    rows = existing or {}
    found: dict[str, FileInfo] = {}
    failed = 0
    for path in _files.paths(args.files):
        try:
            info = _files.stat(path)
        except OSError as error:
            failed += 1
            log.warning("%s: skip file, %s", path, reason(error))
            continue
        found[info.name] = info

    newest = time.time_ns() - int(args.min_age * 1e9)
    todo = []
    for name, info in found.items():
        if (info.mtime_ns, info.size) in rows.get(name, set()):
            continue
        if info.mtime_ns > newest:
            log.debug("%s: changed less than %g s ago, wait for the next sync", info.path, args.min_age)
            continue
        todo.append(info)

    dropped = Dropped()
    written: list[FileInfo] = []
    if todo:
        if existing is None:
            writer = create(args.table, stats=args.stats, casting=args.casting)
        else:
            writer = append(args.table, casting=args.casting)
        with writer:
            for info in todo:
                try:
                    write_file(writer, info, args, dropped)
                except Exception as error:
                    failed += 1
                    log.warning("%s: skip file, %s", info.path, reason(error))
                    continue
                written.append(info)
                log.debug("%s: written", info.path)
            if not written:
                writer.abort()

    # The appends come first, so a reader never sees a file without a row.
    stale = [info for info in written if info.name in rows]
    stale += [
        info
        for name, info in found.items()
        if len(rows.get(name, set())) > 1 and (info.mtime_ns, info.size) in rows[name]
    ]
    gone = sorted(name for name in rows if name not in found)
    if gone and not found:
        # An empty match is more often a missing mount than a deleted folder.
        log.warning("no file matches %s, keep all rows", " ".join(args.files))
        gone = []
    terms = [_stale(info) for info in stale]
    if gone:
        terms.append(f"{_FILE} IN ({', '.join(_quote(name) for name in gone)})")
    deleted = lance_nd.open(args.table).delete(" OR ".join(terms)) if terms else 0
    if args.compact and (written or deleted):
        lance_nd.open(args.table).compact()

    changed = sum(info.name in rows for info in written)
    # A cycle without a change logs at debug level, so a long sync keeps a short log.
    level = logging.INFO if written or gone or failed or deleted else logging.DEBUG
    log.log(
        level,
        "%d files added, %d files changed, %d files deleted, %d files failed, "
        "%d variables dropped, %d attributes dropped",
        len(written) - changed,
        changed,
        len(gone),
        failed,
        dropped.variables,
        dropped.attributes,
    )
    return failed == 0


def _cycle_or_log(args: argparse.Namespace) -> bool:
    try:
        return cycle(args)
    except Exception as error:
        log.error("%s: sync failed, %s", args.table, reason(error))
        return False


def _wait(stop: threading.Event, seconds: float) -> None:
    # Short sleeps, because a wait on an event ignores Ctrl+C on Windows.
    end = time.monotonic() + seconds
    while not stop.is_set() and (left := end - time.monotonic()) > 0:
        time.sleep(min(left, 0.5))


def run(args: argparse.Namespace, stop: threading.Event | None = None) -> int:
    """Sync one time, or at each interval until ``stop`` or Ctrl+C. Give the exit code."""
    if args.interval is None:
        return 0 if _cycle_or_log(args) else 1
    stop = stop or threading.Event()

    def on_interrupt(signum: int, frame: FrameType | None) -> None:
        if stop.is_set():
            raise KeyboardInterrupt
        log.info("stop after this sync, press Ctrl+C again to stop now")
        stop.set()

    main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGINT, on_interrupt) if main_thread else None
    try:
        log.info("sync %s to %s every %g s", " ".join(args.files), args.table, args.interval)
        while not stop.is_set():
            _cycle_or_log(args)
            _wait(stop, args.interval)
    finally:
        if main_thread:
            signal.signal(signal.SIGINT, previous)
    log.info("sync stopped")
    return 0
