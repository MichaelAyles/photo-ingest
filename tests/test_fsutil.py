"""Tests for banger.fsutil: atomic file copy + free-space preflight.

These rely only on stdlib (hashlib/os/shutil) — no cv2/numpy needed — so they
run anywhere the package imports.
"""

from __future__ import annotations

import hashlib

import pytest

from banger import fsutil


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_atomic_copy_happy_path(tmp_path):
    src = tmp_path / "src.bin"
    payload = b"banger-bytes" * 4096  # a few hundred KB, exercises chunked copy
    src.write_bytes(payload)
    dst = tmp_path / "out" / "dst.bin"  # parent doesn't exist yet

    returned = fsutil.atomic_copy(src, dst)

    assert dst.exists()
    assert dst.read_bytes() == payload
    assert returned == _sha(payload)
    # No temp sibling left behind.
    assert not dst.with_name(dst.name + ".part").exists()


def test_atomic_copy_verifies_expected_sha(tmp_path):
    src = tmp_path / "src.bin"
    payload = b"hello world"
    src.write_bytes(payload)
    dst = tmp_path / "dst.bin"

    # Correct expected_sha -> succeeds and returns it.
    returned = fsutil.atomic_copy(src, dst, expected_sha=_sha(payload))
    assert returned == _sha(payload)
    assert dst.read_bytes() == payload


def test_atomic_copy_sha_mismatch_raises_and_leaves_no_file(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"real content")
    dst = tmp_path / "dst.bin"

    with pytest.raises(IOError):
        fsutil.atomic_copy(src, dst, expected_sha="0" * 64)

    # On mismatch the destination is removed entirely — no partial, no final.
    assert not dst.exists()
    assert not dst.with_name(dst.name + ".part").exists()


def test_atomic_copy_overwrites_existing_only_on_success(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"new")
    dst = tmp_path / "dst.bin"
    dst.write_bytes(b"old-existing-content")

    fsutil.atomic_copy(src, dst)
    assert dst.read_bytes() == b"new"


def test_atomic_copy_preserves_mode(tmp_path):
    import os
    import stat

    src = tmp_path / "src.bin"
    src.write_bytes(b"x")
    os.chmod(src, 0o640)
    dst = tmp_path / "dst.bin"

    fsutil.atomic_copy(src, dst)
    mode = stat.S_IMODE(os.stat(dst).st_mode)
    assert mode == 0o640


def test_has_free_space_true_for_small_need(tmp_path):
    # A handful of bytes always fits on any volume with a test tmp_path.
    assert fsutil.has_free_space(tmp_path, 1024) is True


def test_has_free_space_false_for_absurd_need(tmp_path):
    # Petabytes won't fit; preflight must say no.
    assert fsutil.has_free_space(tmp_path, 10**18) is False


def test_has_free_space_walks_up_to_existing_ancestor(tmp_path):
    # dst dir doesn't exist yet; should check the nearest existing ancestor's FS.
    nonexistent = tmp_path / "a" / "b" / "c"
    assert fsutil.has_free_space(nonexistent, 1024) is True
