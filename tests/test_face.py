"""Face detection: Haar cascade smoke tests with synthetic + real images."""


import numpy as np
import pytest

from banger.face import (
    FACE_SHARPNESS_THRESHOLD,
    best_face_sharpness,
    detect_faces,
    face_sharpness,
)


def test_detect_faces_on_blank_image_returns_empty():
    blank = np.full((300, 400, 3), 128, dtype=np.uint8)
    assert detect_faces(blank) == []


def test_detect_faces_on_random_noise_returns_empty():
    rng = np.random.default_rng(seed=0)
    noise = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    # Haar can occasionally false-positive on noise; allow up to 1.
    assert len(detect_faces(noise)) <= 1


def test_face_sharpness_on_solid_crop_is_zero():
    img = np.full((300, 400, 3), 200, dtype=np.uint8)
    s = face_sharpness(img, (10, 10, 50, 50))
    assert s == 0.0


def test_face_sharpness_on_high_detail_crop_is_nonzero():
    rng = np.random.default_rng(seed=1)
    img = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    s = face_sharpness(img, (10, 10, 100, 100))
    assert s > 100  # noisy crop produces high Laplacian variance


def test_face_sharpness_clamps_out_of_bounds_box():
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    # Box partially outside: still returns a finite value, no crash.
    s = face_sharpness(img, (50, 50, 200, 200))
    assert s == 0.0  # clamped to (50:100, 50:100), all 200 → flat


def test_face_sharpness_zero_for_inverted_box():
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    s = face_sharpness(img, (90, 90, 0, 0))
    assert s == 0.0


def test_best_face_sharpness_no_face():
    blank = np.full((300, 400, 3), 128, dtype=np.uint8)
    count, sharpness = best_face_sharpness(blank)
    assert count == 0
    assert sharpness == 0.0


def test_threshold_documented_in_calibrated_range():
    # Face crops have less detail per pixel; threshold lower than the global
    # 100. Anything wildly outside this band would be a typo.
    assert 10 <= FACE_SHARPNESS_THRESHOLD <= 200


def test_detect_handles_empty_array():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert detect_faces(empty) == []


@pytest.mark.parametrize(
    "shape",
    [
        (10, 10, 3),  # tiny image: faces too small to detect at any scale
        (300, 400, 3),
        (1024, 768, 3),
    ],
)
def test_detect_faces_returns_list_of_int_tuples(shape):
    rng = np.random.default_rng(seed=42)
    img = rng.integers(0, 256, shape, dtype=np.uint8)
    boxes = detect_faces(img)
    for b in boxes:
        assert isinstance(b, tuple) and len(b) == 4
        assert all(isinstance(v, int) for v in b)
