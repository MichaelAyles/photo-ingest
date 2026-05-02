"""Report HTML: thumbnail bytes encode, write_report produces valid markup."""

import re

from banger.frames import Frame
from banger.preview import load_preview
from banger.report import Row, encode_thumbnail, encode_thumbnail_bytes, write_report


def test_encode_thumbnail_bytes_returns_jpeg(make_jpeg):
    p = make_jpeg()
    preview = load_preview(p)
    data = encode_thumbnail_bytes(preview)
    assert data[:3] == b"\xff\xd8\xff"  # JPEG SOI marker
    assert len(data) > 100


def test_encode_thumbnail_returns_base64_string(make_jpeg):
    p = make_jpeg()
    preview = load_preview(p)
    s = encode_thumbnail(preview)
    assert isinstance(s, str)
    assert len(s) > 0
    assert "/9j/" in s  # base64 of JPEG SOI


def test_write_report_creates_valid_html(tmp_path, make_jpeg):
    p = make_jpeg()
    preview = load_preview(p)
    thumb = encode_thumbnail(preview)
    rows = [
        Row(
            frame=Frame(stem="DSC1", subdir="", jpeg=p, raw=None),
            sharpness=500.0,
            aesthetic=2.5,
            aesthetic_breakdown={"a positive": 0.3, "a negative": 0.1},
            aesthetic_source="prompts",
            thumb_b64=thumb,
        ),
        Row(
            frame=Frame(stem="DSC2", subdir="trip", jpeg=p, raw=None),
            sharpness=50.0,
            aesthetic=None,
            aesthetic_breakdown=None,
            aesthetic_source=None,
            thumb_b64=thumb,
        ),
    ]
    out = tmp_path / "out" / "report.html"
    write_report(out, rows, threshold=100.0)

    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "DSC1" in html
    assert "trip/DSC2" in html  # display_name with subdir
    assert "KEEP" in html
    assert "REJECT" in html
    assert "below sharpness threshold" in html
    # The keep card is sorted before the reject card
    keep_pos = html.index("DSC1")
    reject_pos = html.index("trip/DSC2")
    assert keep_pos < reject_pos
    # No unbalanced template braces
    assert not re.search(r"\{[a-zA-Z_]+\}", html)


def test_write_report_with_burst_cluster(tmp_path, make_jpeg):
    p = make_jpeg()
    preview = load_preview(p)
    thumb = encode_thumbnail(preview)
    rows = [
        Row(
            frame=Frame(stem="best", subdir="", jpeg=p, raw=None),
            sharpness=500.0,
            aesthetic=4.0,
            aesthetic_breakdown=None,
            aesthetic_source="head",
            thumb_b64=thumb,
            cluster_id=1,
            cluster_size=3,
            cluster_best=True,
        ),
        Row(
            frame=Frame(stem="dupe1", subdir="", jpeg=p, raw=None),
            sharpness=500.0,
            aesthetic=3.5,
            aesthetic_breakdown=None,
            aesthetic_source="head",
            thumb_b64=thumb,
            cluster_id=1,
            cluster_size=3,
            cluster_best=False,
        ),
        Row(
            frame=Frame(stem="dupe2", subdir="", jpeg=p, raw=None),
            sharpness=500.0,
            aesthetic=2.0,
            aesthetic_breakdown=None,
            aesthetic_source="head",
            thumb_b64=thumb,
            cluster_id=1,
            cluster_size=3,
            cluster_best=False,
        ),
        Row(
            frame=Frame(stem="solo", subdir="", jpeg=p, raw=None),
            sharpness=500.0,
            aesthetic=1.0,
            aesthetic_breakdown=None,
            aesthetic_source="head",
            thumb_b64=thumb,
        ),
    ]
    out = tmp_path / "out.html"
    write_report(out, rows, threshold=100.0)
    html = out.read_text(encoding="utf-8")

    assert "best of 3 burst" in html
    assert "dupe of cluster 1" in html
    assert "DUPE" in html
    assert "suppressed by burst dedup" in html
    # Best + solo come before any DUPE in the rendered order.
    best_idx = html.index(">best<")
    solo_idx = html.index(">solo<")
    dupe_divider_idx = html.index("suppressed by burst dedup")
    dupe_idx = html.index(">dupe1<")
    assert best_idx < dupe_divider_idx
    assert solo_idx < dupe_divider_idx
    assert dupe_divider_idx < dupe_idx
