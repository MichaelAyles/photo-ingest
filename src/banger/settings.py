"""Runtime-tunable settings for the cull pipeline.

Persisted as JSON in the state dir so the GUI's Settings tab can edit them
and have changes survive a restart. Anything not present in the file falls
back to the DEFAULTS below.
"""
from __future__ import annotations

import json
import threading
from typing import Any

from banger import state

SETTINGS_PATH = state.STATE_DIR / "settings.json"

DEFAULTS: dict[str, Any] = {
    "sharpness_threshold": 150.0,
    "face_sharpness_threshold": 50.0,
    "top_n": 10,
    "strategy": "kmeans",
    "mmr_diversity": 0.5,
    "tag_min_sim": 0.22,
    "face_gate": False,
    "eye_gate": False,
}

FIELD_META: dict[str, dict[str, Any]] = {
    "sharpness_threshold": {
        "label": "Sharpness threshold",
        "min": 0, "max": 500, "step": 5, "type": "number",
        "info": "Laplacian-variance gate. Frames below this are dropped before scoring. Higher = stricter (more rejections); typical range 100-200.",
    },
    "face_sharpness_threshold": {
        "label": "Face sharpness threshold",
        "min": 0, "max": 300, "step": 5, "type": "number",
        "info": "Per-face Laplacian variance for the face-aware gate. Only applies when 'Face gate' is on AND a face is detected. Higher = stricter.",
    },
    "top_n": {
        "label": "Top N",
        "min": 1, "max": 500, "step": 1, "type": "number",
        "info": "How many keepers to surface / export by default. The export modal can still override per-run.",
    },
    "strategy": {
        "label": "Default strategy",
        "type": "choice",
        "choices": ["topk", "mmr", "kmeans", "faces"],
        "info": "Selection algorithm. 'kmeans' = visual variety. 'faces' = one good shot per person. 'mmr' = continuous diversity dial (see below). 'topk' = pure aesthetic ranking.",
    },
    "mmr_diversity": {
        "label": "MMR diversity λ",
        "min": 0, "max": 1, "step": 0.05, "type": "number",
        "info": "Only used by the 'mmr' strategy. 1.0 = pure top-K by aesthetic, 0.0 = pure visual diversity, 0.5 = balanced.",
    },
    "tag_min_sim": {
        "label": "Tag min cosine",
        "min": 0.1, "max": 0.35, "step": 0.01, "type": "number",
        "info": "CLIP cosine threshold for a tag to be applied to a frame. ~0.22 is the empirical 'this clearly applies' line for ViT-B/32. Higher = fewer, more confident tags.",
    },
    "face_gate": {
        "label": "Face gate",
        "type": "bool",
        "info": "Also reject frames whose detected face is softer than the face-sharpness threshold. Catches the classic missed-focus portrait that passes the global sharpness gate.",
    },
    "eye_gate": {
        "label": "Eye gate",
        "type": "bool",
        "info": "Reject frames where someone's eyes appear closed (Eye Aspect Ratio below threshold). Requires mediapipe. No-op if mediapipe isn't installed.",
    },
}

_lock = threading.Lock()
_cache: dict[str, Any] | None = None


def _read_disk() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load() -> dict[str, Any]:
    """Return the current settings dict (defaults overlaid with disk values)."""
    global _cache
    with _lock:
        if _cache is None:
            _cache = {**DEFAULTS, **_read_disk()}
        return dict(_cache)


def get(key: str) -> Any:
    return load().get(key, DEFAULTS.get(key))


def save(updates: dict[str, Any]) -> dict[str, Any]:
    """Merge `updates` into the current settings, persist, return the new dict."""
    global _cache
    with _lock:
        current = {**DEFAULTS, **_read_disk()}
        for k, v in updates.items():
            if k not in DEFAULTS:
                continue
            current[k] = _coerce(k, v)
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
        _cache = current
        return dict(current)


def reset() -> dict[str, Any]:
    global _cache
    with _lock:
        if SETTINGS_PATH.exists():
            SETTINGS_PATH.unlink()
        _cache = dict(DEFAULTS)
        return dict(_cache)


def _coerce(key: str, value: Any) -> Any:
    meta = FIELD_META.get(key, {})
    t = meta.get("type")
    if t == "bool":
        return bool(value)
    if t == "choice":
        choices = meta.get("choices", [])
        return value if value in choices else DEFAULTS[key]
    if isinstance(DEFAULTS[key], int):
        try:
            return int(value)
        except (TypeError, ValueError):
            return DEFAULTS[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return DEFAULTS[key]
