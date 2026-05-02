"""Sharpness gate: Laplacian variance is monotonic in synthetic detail level."""

from banger.preview import load_preview
from banger.sharpness import CONFIG, sharpness, sharpness_from_preview


def test_blank_image_scores_near_zero(make_jpeg):
    p = make_jpeg(sharpness="blank")
    s = sharpness(p)
    assert s < 1.0


def test_high_detail_beats_low_detail(make_jpeg):
    low = sharpness(make_jpeg(name="low.JPG", sharpness="low"))
    high = sharpness(make_jpeg(name="high.JPG", sharpness="high"))
    assert high > low * 10  # several orders of magnitude apart


def test_threshold_default_is_100(make_jpeg):
    # The plan documents threshold=100; if someone bumps it, the docs need updating.
    assert CONFIG["threshold"] == 100.0


def test_sharpness_from_preview_matches_sharpness(make_jpeg):
    p = make_jpeg(sharpness="medium")
    s_path = sharpness(p)
    s_arr = sharpness_from_preview(load_preview(p))
    assert s_path == s_arr
