"""Tests for banger.settings: defaults, types, load/save round-trip, coercion.

Pure stdlib (json/threading) under the hood, so these run even without the
cv2 / torch stack. We redirect the on-disk SETTINGS_PATH into tmp_path and
clear the module-level cache so each test sees a clean slate and never
touches the real ~/.local state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from banger import settings


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path: Path, monkeypatch):
    """Point settings at a temp file and reset its in-memory cache per test."""
    path = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "SETTINGS_PATH", path)
    # The module caches the merged dict; force a fresh read for every test.
    monkeypatch.setattr(settings, "_cache", None)
    return path


# --------------------------------------------------------------------------- #
# DEFAULTS
# --------------------------------------------------------------------------- #


def test_defaults_present_and_typed():
    d = settings.DEFAULTS
    # Spot-check the keys other modules depend on, with expected types.
    assert isinstance(d["sharpness_threshold"], float)
    assert isinstance(d["face_sharpness_threshold"], float)
    assert isinstance(d["top_n"], int)
    assert isinstance(d["strategy"], str)
    assert isinstance(d["mmr_diversity"], float)
    assert isinstance(d["tag_min_sim"], float)
    assert isinstance(d["dedup_enabled"], bool)
    assert isinstance(d["dedup_hamming"], int)
    assert isinstance(d["dedup_time_window"], float)
    assert isinstance(d["eye_ear_threshold"], float)


def test_strategy_default_is_a_valid_choice():
    assert settings.DEFAULTS["strategy"] in settings.FIELD_META["strategy"]["choices"]


def test_face_and_eye_gate_default_true():
    """Coordinated with the gate-defaults change: both cull gates ship ON."""
    assert settings.DEFAULTS["face_gate"] is True
    assert settings.DEFAULTS["eye_gate"] is True
    # And load() surfaces the True default when nothing is on disk.
    loaded = settings.load()
    assert loaded["face_gate"] is True
    assert loaded["eye_gate"] is True


def test_every_default_has_field_meta():
    # FIELD_META drives the GUI; a default without meta would be uneditable.
    for key in settings.DEFAULTS:
        assert key in settings.FIELD_META, f"missing FIELD_META for {key}"


# --------------------------------------------------------------------------- #
# load() / get()
# --------------------------------------------------------------------------- #


def test_load_returns_defaults_when_no_file(isolated_settings):
    assert not isolated_settings.exists()
    loaded = settings.load()
    assert loaded == settings.DEFAULTS
    # load() must return a copy, not the live DEFAULTS dict.
    loaded["top_n"] = 999
    assert settings.DEFAULTS["top_n"] != 999


def test_get_falls_back_to_default():
    assert settings.get("top_n") == settings.DEFAULTS["top_n"]
    # Unknown key returns None (DEFAULTS.get miss).
    assert settings.get("does_not_exist") is None


# --------------------------------------------------------------------------- #
# save() round-trip + persistence
# --------------------------------------------------------------------------- #


def test_save_round_trip(isolated_settings):
    settings.save({"top_n": 25, "strategy": "mmr"})
    assert isolated_settings.exists()
    on_disk = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert on_disk["top_n"] == 25
    assert on_disk["strategy"] == "mmr"
    # load() reflects the saved values.
    loaded = settings.load()
    assert loaded["top_n"] == 25
    assert loaded["strategy"] == "mmr"


def test_save_persists_across_cache_reset(isolated_settings, monkeypatch):
    settings.save({"top_n": 42})
    # Simulate a process restart: drop the cache, re-read from disk.
    monkeypatch.setattr(settings, "_cache", None)
    assert settings.load()["top_n"] == 42


def test_save_merges_does_not_drop_other_keys(isolated_settings):
    settings.save({"top_n": 7})
    settings.save({"strategy": "topk"})
    loaded = settings.load()
    assert loaded["top_n"] == 7  # not clobbered by the second save
    assert loaded["strategy"] == "topk"


# --------------------------------------------------------------------------- #
# Unknown-key handling + coercion
# --------------------------------------------------------------------------- #


def test_save_ignores_unknown_keys(isolated_settings):
    result = settings.save({"top_n": 12, "bogus_key": "nope"})
    assert "bogus_key" not in result
    on_disk = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert "bogus_key" not in on_disk
    assert result["top_n"] == 12


def test_save_coerces_numeric_strings(isolated_settings):
    result = settings.save({"top_n": "30", "mmr_diversity": "0.75"})
    assert result["top_n"] == 30 and isinstance(result["top_n"], int)
    assert result["mmr_diversity"] == 0.75 and isinstance(result["mmr_diversity"], float)


def test_save_bad_numeric_falls_back_to_default(isolated_settings):
    result = settings.save({"top_n": "not-a-number"})
    assert result["top_n"] == settings.DEFAULTS["top_n"]


def test_save_invalid_choice_falls_back_to_default(isolated_settings):
    result = settings.save({"strategy": "telepathy"})
    assert result["strategy"] == settings.DEFAULTS["strategy"]


def test_save_coerces_bool(isolated_settings):
    result = settings.save({"face_gate": 0, "eye_gate": 1})
    assert result["face_gate"] is False
    assert result["eye_gate"] is True


def test_unreadable_file_falls_back_to_defaults(isolated_settings, monkeypatch):
    # A corrupt settings.json must not crash load(); it falls back to defaults.
    isolated_settings.write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setattr(settings, "_cache", None)
    loaded = settings.load()
    assert loaded == settings.DEFAULTS


# --------------------------------------------------------------------------- #
# reset()
# --------------------------------------------------------------------------- #


def test_reset_removes_file_and_restores_defaults(isolated_settings):
    settings.save({"top_n": 99})
    assert isolated_settings.exists()
    out = settings.reset()
    assert not isolated_settings.exists()
    assert out == settings.DEFAULTS
    assert settings.load()["top_n"] == settings.DEFAULTS["top_n"]
