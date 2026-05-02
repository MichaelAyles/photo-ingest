"""Shared test fixtures.

Most fixtures here either build tiny synthetic JPEGs in tmp_path so tests
don't need real photo files committed to the repo, or redirect the global
state directory at `Path.home() / .local / share / banger-pipeline` to a
test-scoped tmp_path so live state isn't touched.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


def _make_image(width: int = 64, height: int = 48, sharpness: str = "medium") -> np.ndarray:
    """Return a BGR uint8 image with a controllable amount of high-frequency content.

    The Laplacian variance — which the pipeline uses for sharpness — depends
    on per-pixel detail. A flat image scores ~0; a checkerboard scores high.
    """
    rng = np.random.default_rng(seed=42)
    if sharpness == "blank":
        # Solid grey: zero gradient, zero variance.
        return np.full((height, width, 3), 128, dtype=np.uint8)
    if sharpness == "low":
        # Smooth gradient, very low Laplacian variance.
        x = np.linspace(0, 255, width, dtype=np.uint8)
        return np.broadcast_to(x[None, :, None], (height, width, 3)).copy()
    if sharpness == "medium":
        # Medium: noise blurred with a Gaussian — some detail, but soft.
        noise = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        return cv2.GaussianBlur(noise, (5, 5), 0)
    if sharpness == "high":
        # High: pure noise + high-contrast checkerboard, lots of edges.
        noise = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        cb = np.indices((height, width)).sum(axis=0) % 2 * 255
        return np.dstack([cb, cb, cb]).astype(np.uint8) // 2 + noise // 2
    raise ValueError(f"unknown sharpness: {sharpness}")


@pytest.fixture
def make_jpeg(tmp_path: Path):
    """Returns a callable that writes a synthetic JPEG into tmp_path and returns the path."""

    def _factory(name: str = "test.JPG", sharpness: str = "medium", subdir: str = "") -> Path:
        target_dir = tmp_path / subdir if subdir else tmp_path
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / name
        img = _make_image(sharpness=sharpness)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok, "cv2.imencode failed"
        path.write_bytes(buf.tobytes())
        return path

    return _factory


@pytest.fixture
def isolated_state(tmp_path: Path, monkeypatch):
    """Redirect banger.state's on-disk paths into tmp_path so tests don't pollute ~/.local/."""
    import banger.state as st

    state_dir = tmp_path / "state"
    monkeypatch.setattr(st, "STATE_DIR", state_dir)
    monkeypatch.setattr(st, "EMBEDDINGS_DIR", state_dir / "embeddings")
    monkeypatch.setattr(st, "THUMBS_DIR", state_dir / "thumbs")
    monkeypatch.setattr(st, "PREVIEWS_DIR", state_dir / "previews")
    monkeypatch.setattr(st, "LABELS_DB", state_dir / "labels.db")
    monkeypatch.setattr(st, "TASTE_HEAD", state_dir / "taste_head.joblib")
    return st
