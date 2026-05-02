"""Develop stage: darktable detection, fallback copy, manifest write."""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

from banger.develop import (
    DevelopResult,
    copy_fallback,
    develop_to_jpeg,
    find_darktable,
    write_manifest,
)


def test_find_darktable_returns_none_when_absent(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr("banger.develop.DARKTABLE_WINDOWS_FALLBACK_PATHS", ())
    assert find_darktable() is None


def test_find_darktable_finds_path_first(monkeypatch, tmp_path):
    fake = tmp_path / "darktable-cli"
    fake.write_text("")
    monkeypatch.setattr(shutil, "which", lambda name: str(fake) if "darktable-cli" in name else None)
    found = find_darktable()
    assert found == fake


def test_find_darktable_fallback_to_windows_path(monkeypatch, tmp_path):
    fake = tmp_path / "darktable-cli.exe"
    fake.write_text("")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr("banger.develop.DARKTABLE_WINDOWS_FALLBACK_PATHS", (fake,))
    assert find_darktable() == fake


def test_copy_fallback_produces_identical_bytes(tmp_path, make_jpeg):
    src = make_jpeg()
    dst = tmp_path / "out" / "copy.jpg"
    result = copy_fallback(src, dst)
    assert result.success
    assert result.used == "copy_fallback"
    assert dst.read_bytes() == src.read_bytes()


def test_develop_to_jpeg_falls_back_when_no_darktable(tmp_path, make_jpeg):
    src = make_jpeg()
    dst = tmp_path / "out" / "dev.jpg"
    result = develop_to_jpeg(src, dst, preset_name="bw_moody", presets_dir=tmp_path / "no_presets")
    assert result.success
    assert result.used == "copy_fallback"
    assert dst.exists()


def test_develop_to_jpeg_falls_back_when_preset_xmp_missing(tmp_path, make_jpeg):
    src = make_jpeg()
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    # No bw_moody.xmp inside.
    dst = tmp_path / "out" / "dev.jpg"
    fake_dt = tmp_path / "darktable-cli"
    fake_dt.write_text("")
    result = develop_to_jpeg(
        src, dst, preset_name="bw_moody", presets_dir=presets_dir, darktable_cli=fake_dt
    )
    assert result.success
    # No XMP and src is JPEG -> copy fallback (avoids darktable's default render).
    assert result.used == "copy_fallback"


def test_develop_calls_darktable_when_xmp_present(tmp_path, make_jpeg):
    src = make_jpeg()
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    (presets_dir / "bw_moody.xmp").write_text("<fake xmp/>")
    dst = tmp_path / "out" / "dev.jpg"
    fake_dt = tmp_path / "darktable-cli"
    fake_dt.write_text("")

    captured: dict = {}

    def fake_run(*args, **kwargs):
        # Pretend darktable wrote the output file.
        cmd = args[0]
        captured["cmd"] = cmd
        Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[-1]).write_bytes(b"\xff\xd8\xff" + b"developed")

        class R:
            returncode = 0
            stderr = ""
            stdout = ""

        return R()

    with patch("banger.develop.subprocess.run", side_effect=fake_run):
        result = develop_to_jpeg(
            src, dst, preset_name="bw_moody", presets_dir=presets_dir, darktable_cli=fake_dt
        )

    assert result.success
    assert result.used == "darktable"
    assert dst.exists()
    cmd = captured["cmd"]
    assert str(fake_dt) in cmd
    assert str(presets_dir / "bw_moody.xmp") in cmd
    assert str(src) in cmd
    assert str(dst) in cmd


def test_develop_copy_fallback_when_darktable_fails_on_jpeg(tmp_path, make_jpeg):
    src = make_jpeg()
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    (presets_dir / "bw_moody.xmp").write_text("<xmp/>")
    dst = tmp_path / "out" / "dev.jpg"
    fake_dt = tmp_path / "darktable-cli"
    fake_dt.write_text("")

    def fake_run(*args, **kwargs):
        class R:
            returncode = 1
            stderr = "darktable: invalid frame"
            stdout = ""

        return R()

    with patch("banger.develop.subprocess.run", side_effect=fake_run):
        result = develop_to_jpeg(
            src, dst, preset_name="bw_moody", presets_dir=presets_dir, darktable_cli=fake_dt
        )

    assert result.success
    assert result.used == "copy_fallback"


def test_write_manifest_round_trip(tmp_path):
    entries = [
        {"stem": "DSC1", "score": 4.2, "preset": "bw_moody"},
        {"stem": "DSC2", "score": 3.1, "preset": "crisp_daylight", "fell_back": True},
    ]
    out = tmp_path / "manifest.json"
    write_manifest(out, entries)
    loaded = json.loads(out.read_text())
    assert loaded == entries


def test_develop_result_has_documented_fields():
    r = DevelopResult(success=True, used="darktable", note="ok")
    assert r.success
    assert r.used == "darktable"
    assert r.note == "ok"
