"""Tests for banger.metrics: the pure-OpenCV quality dimensions.

These need cv2 (so does conftest's make_jpeg fixture). In an environment
without OpenCV the whole module is skipped at import; it runs in CI / the
verification venv where cv2 is installed.

We assert two kinds of property:
  * range/shape -- every 0..10 dim stays in [0, 10], flags are 0/1, etc.
  * ordering    -- deterministic, sensible comparisons (a bright frame reads
    higher exposure than a near-black one; a noisy frame reads noisier than a
    flat one; a colourful frame is not flagged monochrome).

Synthetic frames are built two ways: via conftest's make_jpeg fixture
(round-tripped through JPEG, read back as BGR) for the file-backed cases,
and via small hand-built numpy arrays for the brightness/colour cases that
the fixture doesn't parameterise.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from banger import metrics  # noqa: E402  (after importorskip)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

H, W = 120, 160


def _solid(value: int) -> np.ndarray:
    """Solid-grey BGR frame at the given 0..255 level."""
    return np.full((H, W, 3), value, dtype=np.uint8)


def _gradient() -> np.ndarray:
    """Full-range horizontal luminance ramp: wide contrast + dynamic range."""
    ramp = np.linspace(0, 255, W, dtype=np.uint8)
    return np.broadcast_to(ramp[None, :, None], (H, W, 3)).copy()


def _noisy(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)


def _colourful(seed: int = 3) -> np.ndarray:
    """Random hues so saturation is high -> not monochrome, decent harmony."""
    rng = np.random.default_rng(seed)
    hsv = np.empty((H, W, 3), dtype=np.uint8)
    hsv[:, :, 0] = rng.integers(0, 180, (H, W), dtype=np.uint8)  # hue
    hsv[:, :, 1] = rng.integers(180, 256, (H, W), dtype=np.uint8)  # high sat
    hsv[:, :, 2] = rng.integers(60, 256, (H, W), dtype=np.uint8)  # value
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _read(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None, f"cv2 could not read {path}"
    return img


def _gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _hsv(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def _in_range(x, lo=0.0, hi=10.0):
    return lo <= float(x) <= hi


# --------------------------------------------------------------------------- #
# compute_all: shape, keys, ranges, determinism
# --------------------------------------------------------------------------- #


def test_compute_all_empty_input_returns_empty():
    assert metrics.compute_all(None) == {}
    assert metrics.compute_all(np.empty((0, 0, 3), dtype=np.uint8)) == {}


def test_compute_all_keys_present(make_jpeg):
    out = metrics.compute_all(_read(make_jpeg(name="a.JPG", sharpness="high")))
    for key in (
        "color_harmony", "exposure", "contrast", "noise", "dynamic_range",
        "composition", "leading_lines", "mean_luminance",
        "shadow_clipped", "highlight_clipped", "is_silhouette", "is_monochrome",
    ):
        assert key in out, f"missing metric: {key}"


def test_compute_all_quality_dims_in_range(make_jpeg):
    out = metrics.compute_all(_read(make_jpeg(name="b.JPG", sharpness="medium")))
    for key in metrics.QUALITY_DIMS:
        assert _in_range(out[key]), f"{key}={out[key]} out of 0..10"
    assert 0.0 <= out["mean_luminance"] <= 1.0
    for flag in ("shadow_clipped", "highlight_clipped", "is_silhouette", "is_monochrome"):
        assert out[flag] in (0, 1)
    assert out["noise"] >= 0.0
    assert out["dynamic_range"] >= 0.0


def test_compute_all_is_deterministic():
    frame = _noisy(seed=11)
    a = metrics.compute_all(frame.copy())
    b = metrics.compute_all(frame.copy())
    assert a == b


# --------------------------------------------------------------------------- #
# exposure: brightness ordering + clipping/silhouette flags
# --------------------------------------------------------------------------- #


def test_exposure_mean_luminance_orders_by_brightness():
    dark = metrics._exposure(_gray(_solid(20)))
    mid = metrics._exposure(_gray(_solid(128)))
    bright = metrics._exposure(_gray(_solid(235)))
    assert dark["mean_luminance"] < mid["mean_luminance"] < bright["mean_luminance"]


def test_exposure_midtone_scores_higher_than_extremes():
    # A near-50%-grey frame is "better exposed" than crushed black or blown white.
    mid = metrics._exposure(_gray(_solid(128)))["exposure"]
    dark = metrics._exposure(_gray(_solid(8)))["exposure"]
    bright = metrics._exposure(_gray(_solid(250)))["exposure"]
    assert mid > dark
    assert mid > bright


def test_exposure_dark_frame_flags_shadow_clip():
    out = metrics._exposure(_gray(_solid(5)))
    assert out["shadow_clipped"] == 1
    assert out["highlight_clipped"] == 0


def test_exposure_bright_frame_flags_highlight_clip():
    out = metrics._exposure(_gray(_solid(252)))
    assert out["highlight_clipped"] == 1
    assert out["shadow_clipped"] == 0


def test_exposure_silhouette_detected_on_bimodal_frame():
    # Half crushed shadow, half blown highlight -> intentional backlit silhouette.
    g = np.empty((H, W), dtype=np.uint8)
    g[:, : W // 2] = 5
    g[:, W // 2 :] = 250
    out = metrics._exposure(g)
    assert out["is_silhouette"] == 1


# --------------------------------------------------------------------------- #
# contrast + dynamic range
# --------------------------------------------------------------------------- #


def test_contrast_full_range_beats_flat():
    flat = metrics._contrast(_gray(_solid(128)))["contrast"]
    wide = metrics._contrast(_gray(_gradient()))["contrast"]
    assert wide > flat
    assert flat == pytest.approx(0.0, abs=1e-6)


def test_dynamic_range_full_range_beats_flat():
    flat = metrics._dynamic_range(_gray(_solid(128)))
    wide = metrics._dynamic_range(_gray(_gradient()))
    assert wide > flat


# --------------------------------------------------------------------------- #
# noise
# --------------------------------------------------------------------------- #


def test_noise_random_frame_noisier_than_flat():
    flat = metrics._noise(_gray(_solid(128)))
    noisy = metrics._noise(_gray(_noisy(seed=5)))
    assert noisy > flat
    assert flat == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# colour harmony + monochrome
# --------------------------------------------------------------------------- #


def test_color_harmony_in_range_and_higher_for_varied_hues():
    flat = metrics._color_harmony(_hsv(_solid(128)))["color_harmony"]
    varied = metrics._color_harmony(_hsv(_colourful()))["color_harmony"]
    assert _in_range(flat) and _in_range(varied)
    # A single grey colour has near-zero hue/sat entropy; varied hues read higher.
    assert varied > flat


def test_is_monochrome_grey_vs_colour():
    # Grey frame has ~zero saturation -> monochrome; colourful frame is not.
    assert metrics._is_monochrome(_hsv(_solid(128))) == 1
    assert metrics._is_monochrome(_hsv(_colourful())) == 0


# --------------------------------------------------------------------------- #
# composition + leading lines: range / flag sanity
# --------------------------------------------------------------------------- #


def test_composition_no_subject_returns_neutral():
    # A perfectly flat frame has no edges, so no subject contour is found.
    out = metrics._composition(_solid(128), _gray(_solid(128)))
    assert out["composition_subject_detected"] == 0
    assert _in_range(out["composition"])


def test_composition_with_subject_in_range():
    # A bright block on a dark field gives a detectable subject contour.
    bgr = _solid(10)
    bgr[H // 3 : H // 3 + 25, W // 3 : W // 3 + 25] = 240
    out = metrics._composition(bgr, _gray(bgr))
    assert out["composition_subject_detected"] == 1
    assert _in_range(out["composition"])


def test_leading_lines_in_range_and_zero_on_blank():
    assert metrics._leading_lines(_gray(_solid(128))) == 0.0
    val = metrics._leading_lines(_gray(_gradient()))
    assert _in_range(val)


# --------------------------------------------------------------------------- #
# combined_score
# --------------------------------------------------------------------------- #


def test_combined_score_no_aesthetic_is_cv_average():
    m = {"exposure": 8.0, "contrast": 6.0, "color_harmony": 4.0,
         "composition": 2.0, "leading_lines": 0.0}
    assert metrics.combined_score(m, aesthetic=None) == pytest.approx(4.0)


def test_combined_score_empty_metrics_defaults_to_five():
    assert metrics.combined_score({}, aesthetic=None) == pytest.approx(5.0)


def test_combined_score_blends_aesthetic_and_stays_in_range():
    m = {k: 6.0 for k in metrics.QUALITY_DIMS}
    # aesthetic +5 -> 10/10; blended with cv avg 6 -> 8.0
    assert metrics.combined_score(m, aesthetic=5.0) == pytest.approx(8.0)
    # extreme aesthetic values are clamped into 0..10 before blending.
    hi = metrics.combined_score(m, aesthetic=1000.0)
    lo = metrics.combined_score(m, aesthetic=-1000.0)
    assert _in_range(hi) and _in_range(lo)
    assert hi > lo


def test_combined_score_higher_metrics_score_higher():
    low = metrics.combined_score({k: 2.0 for k in metrics.QUALITY_DIMS})
    high = metrics.combined_score({k: 9.0 for k in metrics.QUALITY_DIMS})
    assert high > low
