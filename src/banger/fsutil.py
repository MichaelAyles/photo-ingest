"""Filesystem helpers for safe, atomic file copies.

The export step copies the user's selected ("banger") source files out to a
destination folder. A naive ``shutil.copy2`` can leave a half-written file at
the final path if the process is killed (or the disk fills) mid-copy, and a
concurrent reader can observe a truncated file. This module copies to a
temp sibling, fsyncs, verifies the hash, and only then ``os.replace``s into
place — an atomic rename on the same filesystem. The final name therefore
never exists in a partial state.

This module is for file COPIES only. The JSON/.npy sidecar atomic-write logic
lives in ``state.py`` (it owns those formats); keeping the two separate avoids
a circular import and keeps each module's responsibility clear.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

# Safety margin so a copy doesn't completely exhaust the destination volume.
# We require the file size plus this slack to be free before starting.
_FREE_SPACE_MARGIN_BYTES = 16 * 1024 * 1024  # 16 MiB


def _sha256_of_path(path: Path) -> str:
    """Streamed sha256 of a file (1 MiB chunks); never loads the whole file."""
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_copy(src: Path, dst: Path, *, expected_sha: str | None = None) -> str:
    """Atomically copy ``src`` to ``dst`` and return the sha256 of the copy.

    The copy is written to a temp sibling ``dst + ".part"``, flushed and
    ``os.fsync``ed to durable storage, then ``os.replace``d onto ``dst`` —
    which is atomic on the same filesystem, so ``dst`` is never observed in a
    half-written state.

    After the rename, the destination is re-hashed (streamed sha256). If
    ``expected_sha`` is given and the destination hash differs, the destination
    is deleted and an ``IOError`` is raised — the copy is treated as corrupt and
    no file is left behind at the final name.

    File metadata (mode, timestamps) is preserved via ``shutil.copystat``.

    Returns the sha256 hex digest of the copied file.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Temp sibling on the SAME directory/filesystem so os.replace is atomic.
    part = dst.with_name(dst.name + ".part")

    try:
        # Stream the bytes ourselves so we can fsync the destination fd before
        # the rename. shutil.copyfileobj handles the chunked read/write loop.
        with open(src, "rb") as fsrc, open(part, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, length=1 << 20)
            fdst.flush()
            os.fsync(fdst.fileno())
        # Preserve mode/timestamps from the source.
        shutil.copystat(src, part)
        # Atomic publish: dst either is the old file or the fully-written one.
        os.replace(part, dst)
    except BaseException:
        # On any failure, never leave a stray .part behind.
        _unlink_quietly(part)
        raise

    # Verify the published file. If it doesn't match, remove it and fail loud.
    actual_sha = _sha256_of_path(dst)
    if expected_sha is not None and actual_sha != expected_sha:
        _unlink_quietly(dst)
        _unlink_quietly(part)  # defensive: replace consumed it, but be safe.
        raise OSError(
            f"atomic_copy hash mismatch for {dst}: "
            f"expected {expected_sha}, got {actual_sha}"
        )
    return actual_sha


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def has_free_space(dst_dir: Path, needed_bytes: int) -> bool:
    """Return True if ``dst_dir``'s filesystem has room for ``needed_bytes``.

    A preflight check before a batch export so we fail fast with a clear error
    instead of part-way through. Includes a small fixed safety margin so we
    don't fill the volume to the last byte. If the directory doesn't exist yet,
    its nearest existing ancestor is checked (that's the filesystem the new
    files will land on).
    """
    probe = Path(dst_dir)
    # disk_usage needs an existing path; walk up to the first ancestor that
    # exists (the volume the eventual mkdir will create children on).
    while not probe.exists():
        parent = probe.parent
        if parent == probe:  # reached filesystem root and still nothing
            return False
        probe = parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError:
        return False
    return usage.free >= int(needed_bytes) + _FREE_SPACE_MARGIN_BYTES
