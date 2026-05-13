"""XMP sidecar writer smoke tests."""

from pathlib import Path

from banger.frames import Frame
from banger.report import Row
from banger.xmp import (
    CLUSTER_COLOURS,
    _sidecar_path,
    _xmp_for,
    stars_from_rank,
    write_sidecar,
    write_for_rows,
)


def test_stars_distribution():
    assert stars_from_rank(0, 10) == 5
    assert stars_from_rank(1, 10) == 4
    assert stars_from_rank(5, 10) == 3
    assert stars_from_rank(8, 10) == 2
    assert stars_from_rank(9, 10) == 1
    assert stars_from_rank(0, 0) == 0
    assert stars_from_rank(-1, 5) == 0


def test_xmp_payload_contains_rating_and_label():
    xml = _xmp_for(rating=5, label="green", subjects=["banger/scene/golden_landscape"])
    assert 'xmp:Rating="5"' in xml
    assert 'xmp:Label="green"' in xml
    assert "banger/scene/golden_landscape" in xml


def test_xmp_payload_skips_label_when_none():
    xml = _xmp_for(rating=3, label=None, subjects=[])
    assert "xmp:Label" not in xml


def test_xmp_payload_escapes_special_chars():
    xml = _xmp_for(rating=2, label="blue", subjects=["banger/scene/<weird&tag>"])
    assert "&lt;weird&amp;tag&gt;" in xml


def test_sidecar_path_preserves_suffix(tmp_path):
    src = tmp_path / "DSC00055.JPG"
    src.write_bytes(b"")
    assert _sidecar_path(src).name == "DSC00055.JPG.xmp"


def _make_row(tmp_path, stem="DSC00055", aest=2.0, scene="golden_landscape"):
    src = tmp_path / f"{stem}.JPG"
    src.write_bytes(b"")
    frame = Frame(stem=stem, subdir="", jpeg=src, raw=None)
    return Row(
        frame=frame,
        sharpness=400.0,
        aesthetic=aest,
        aesthetic_breakdown=None,
        aesthetic_source="head",
        thumb_b64="",
        scene_preset=scene,
        scene_score=0.27,
        scene_fell_back=False,
        metrics={"exposure": 7.5, "contrast": 8.0, "color_harmony": 6.0,
                 "composition": 7.0, "leading_lines": 4.0},
    )


def test_write_sidecar_creates_file_next_to_source(tmp_path):
    row = _make_row(tmp_path)
    out = write_sidecar(row, rank=0, total=10, cluster_id=2)
    assert out is not None and out.exists()
    body = out.read_text(encoding="utf-8")
    assert 'xmp:Rating="5"' in body
    assert f'xmp:Label="{CLUSTER_COLOURS[2]}"' in body
    assert "banger/scene/golden_landscape" in body
    assert "banger/quality/" in body


def test_write_sidecar_with_no_cluster_uses_scene_hash(tmp_path):
    row = _make_row(tmp_path, scene="bw_moody")
    out = write_sidecar(row, rank=0, total=5)
    assert out is not None
    body = out.read_text(encoding="utf-8")
    assert "xmp:Label" in body  # one of the five colours, derived from scene name


def test_write_sidecar_rating_zero_when_aesthetic_missing(tmp_path):
    row = _make_row(tmp_path, aest=None)
    out = write_sidecar(row, rank=0, total=5)
    body = out.read_text(encoding="utf-8")
    assert 'xmp:Rating="0"' in body


def test_write_for_rows_ranks_by_aesthetic(tmp_path):
    rows = [
        _make_row(tmp_path, stem=f"DSC{i:05d}", aest=float(i))
        for i in range(10)
    ]
    n = write_for_rows(rows)
    assert n == 10
    # The frame with the highest aesthetic (DSC00009) should be rank 0 = 5 stars.
    best = (tmp_path / "DSC00009.JPG.xmp").read_text(encoding="utf-8")
    worst = (tmp_path / "DSC00000.JPG.xmp").read_text(encoding="utf-8")
    assert 'xmp:Rating="5"' in best
    assert 'xmp:Rating="1"' in worst
