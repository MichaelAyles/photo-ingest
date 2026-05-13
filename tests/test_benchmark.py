"""Benchmark harness smoke tests.

Avoids running banger end-to-end (would need a CLIP model load). Stubs the
banger half with a fake pick list and exercises only the comparison +
report-writing logic.
"""

from pathlib import Path

from banger.benchmark import (
    BenchmarkResult,
    Pick,
    _spearman,
    compare,
    write_report,
)


def _picks(stems, scores=None, scene="cluster_00"):
    out = []
    for i, s in enumerate(stems):
        score = scores[i] if scores else 1.0 - i * 0.1
        out.append(
            Pick(rank=i + 1, stem=s, subdir="", aesthetic=score, sharpness=400.0, scene_preset=scene)
        )
    return out


def test_compare_no_facet():
    cmp = compare(_picks(["A", "B", "C"]), facet=None)
    assert cmp["jaccard"] is None
    assert cmp["overlap"] == 0


def test_compare_jaccard_and_overlap():
    a = _picks(["A", "B", "C", "D"])
    b = _picks(["B", "C", "D", "E"])
    cmp = compare(a, b)
    assert cmp["overlap"] == 3
    assert cmp["jaccard"] == round(3 / 5, 3)


def test_spearman_perfect_correlation():
    assert _spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0


def test_spearman_anti_correlation():
    assert _spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_write_report_facet_skipped(tmp_path: Path):
    result = BenchmarkResult(
        input_dir=tmp_path,
        banger_top=_picks(["A", "B", "C"]),
        facet_top=None,
        facet_status="skipped: facet path not found",
        elapsed_banger=1.5,
        elapsed_facet=None,
    )
    out = tmp_path / "report.md"
    write_report(out, result)
    body = out.read_text(encoding="utf-8")
    assert "# banger vs facet benchmark" in body
    assert "Banger top-3" in body
    assert "Facet: skipped:" in body
    assert "## Disagreements" not in body  # only when facet ran


def test_write_report_with_facet(tmp_path: Path):
    result = BenchmarkResult(
        input_dir=tmp_path,
        banger_top=_picks(["A", "B", "C"]),
        facet_top=_picks(["B", "C", "D"]),
        facet_status="ok",
        elapsed_banger=2.0,
        elapsed_facet=8.0,
    )
    out = tmp_path / "report.md"
    write_report(out, result)
    body = out.read_text(encoding="utf-8")
    assert "## Disagreements" in body
    assert "- A" in body  # banger only
    assert "- D" in body  # facet only
    assert "Jaccard:" in body
