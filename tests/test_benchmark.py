"""Benchmark harness smoke tests.

Avoids running banger end-to-end (would need a CLIP model load). Stubs the
banger half with a fake pick list and exercises only the comparison +
report-writing logic.
"""

from pathlib import Path

from banger.benchmark import (
    DETECTORS,
    BenchmarkResult,
    Pick,
    _spearman,
    compare,
    evaluate_against_truth,
    format_accuracy_table,
    precision_recall_f1,
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
    # Title is honest: no "vs facet" claim when facet never ran.
    assert "# banger benchmark" in body
    assert "# banger vs facet benchmark" not in body
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
    # Head-to-head title only appears because facet actually ran.
    assert "# banger vs facet benchmark" in body


def test_write_report_no_facet_is_not_a_comparison(tmp_path: Path):
    """When facet did not run, the report must NOT pretend a comparison happened."""
    result = BenchmarkResult(
        input_dir=tmp_path,
        banger_top=_picks(["A", "B", "C"]),
        facet_top=None,
        facet_status="facet not available: docker CLI not on PATH",
        elapsed_banger=1.0,
        elapsed_facet=None,
    )
    out = tmp_path / "report.md"
    write_report(out, result)
    body = out.read_text(encoding="utf-8")
    # Title drops the "vs facet" framing.
    assert "# banger benchmark" in body
    assert "# banger vs facet benchmark" not in body
    # No fabricated overlap / disagreement sections.
    assert "## Overlap" not in body
    assert "## Disagreements" not in body
    assert "Jaccard:" not in body
    # And it says plainly that facet didn't run.
    assert "facet not available" in body


# --------------------------------------------------------------------------- #
# Measured detector accuracy: precision / recall / F1.
# --------------------------------------------------------------------------- #


def test_precision_recall_f1_basic():
    # Frames: a,b are blurry (positive), c,d are sharp (negative).
    # Detector flags a (TP), c (FP), misses b (FN), correctly skips d (TN).
    preds = {"a": True, "b": False, "c": True, "d": False}
    truth = {"a": True, "b": True, "c": False, "d": False}
    prf = precision_recall_f1(preds, truth)
    assert (prf.tp, prf.fp, prf.fn, prf.tn) == (1, 1, 1, 1)
    assert prf.precision == 0.5  # 1 TP / (1 TP + 1 FP)
    assert prf.recall == 0.5  # 1 TP / (1 TP + 1 FN)
    assert prf.f1 == 0.5
    assert prf.support == 2  # two true positives in ground-truth


def test_precision_recall_perfect():
    preds = {"a": True, "b": True, "c": False}
    truth = {"a": True, "b": True, "c": False}
    prf = precision_recall_f1(preds, truth)
    assert prf.precision == 1.0
    assert prf.recall == 1.0
    assert prf.f1 == 1.0


def test_precision_recall_empty_predictions_no_div_by_zero():
    # No positives predicted at all: precision is defined as 0.0, not a crash.
    preds = {"a": False, "b": False}
    truth = {"a": True, "b": False}
    prf = precision_recall_f1(preds, truth)
    assert prf.precision == 0.0
    assert prf.recall == 0.0  # missed the one true positive
    assert prf.f1 == 0.0


def test_precision_recall_only_scores_shared_keys():
    # Frame "z" is labelled but never predicted; "y" predicted but never labelled.
    # Neither should be counted (otherwise TN gets inflated).
    preds = {"a": True, "y": True}
    truth = {"a": True, "z": False}
    prf = precision_recall_f1(preds, truth)
    assert (prf.tp, prf.fp, prf.fn, prf.tn) == (1, 0, 0, 0)


def test_evaluate_against_truth_tiny_handbuilt_set():
    """A small hand-built truth set exercising all four detectors."""
    predictions = {
        # blur: flags img1 (correct) and img3 (false alarm), misses img2.
        "blur": {"img1": True, "img2": False, "img3": True, "img4": False},
        # blink: perfect on the two it scored.
        "blink": {"img1": True, "img2": False},
        # duplicate: flags img3 & img4 as dupes; img4 actually isn't.
        "duplicate": {"img1": False, "img3": True, "img4": True},
        # keeper: picks img2 & img4 as keepers; both correct, misses img1.
        "keeper": {"img1": False, "img2": True, "img4": True},
    }
    truth = {
        "blur": {"img1": True, "img2": True, "img3": False, "img4": False},
        "blink": {"img1": True, "img2": False},
        "duplicate": {"img1": False, "img3": True, "img4": False},
        "keeper": {"img1": True, "img2": True, "img4": True},
    }
    scores = evaluate_against_truth(predictions, truth)

    # Every detector got scored.
    assert set(scores) == set(DETECTORS)

    blur = scores["blur"]
    assert blur["tp"] == 1 and blur["fp"] == 1 and blur["fn"] == 1
    assert blur["precision"] == 0.5
    assert blur["recall"] == 0.5

    assert scores["blink"]["precision"] == 1.0
    assert scores["blink"]["recall"] == 1.0
    assert scores["blink"]["f1"] == 1.0

    dup = scores["duplicate"]
    assert dup["tp"] == 1 and dup["fp"] == 1  # img3 correct, img4 false alarm
    assert dup["precision"] == 0.5
    assert dup["recall"] == 1.0  # caught the one real dupe

    keep = scores["keeper"]
    assert keep["precision"] == 1.0  # img2,img4 both real keepers
    assert keep["recall"] == round(2 / 3, 4)  # missed img1


def test_evaluate_against_truth_skips_unscored_detectors():
    # Only "blur" present on both sides; "keeper" only in predictions.
    predictions = {"blur": {"a": True}, "keeper": {"a": True}}
    truth = {"blur": {"a": True}}
    scores = evaluate_against_truth(predictions, truth)
    assert set(scores) == {"blur"}


def test_evaluate_against_truth_ignores_unknown_detector_names():
    predictions = {"made_up_detector": {"a": True}, "blur": {"a": True}}
    truth = {"made_up_detector": {"a": True}, "blur": {"a": True}}
    scores = evaluate_against_truth(predictions, truth)
    assert set(scores) == {"blur"}


def test_format_accuracy_table_empty():
    lines = format_accuracy_table({})
    body = "\n".join(lines)
    assert "No ground-truth labels supplied" in body


def test_format_accuracy_table_renders_rows():
    scores = evaluate_against_truth(
        {"blur": {"a": True, "b": False}},
        {"blur": {"a": True, "b": True}},
    )
    lines = format_accuracy_table(scores)
    body = "\n".join(lines)
    assert "Detector accuracy" in body
    assert "| blur |" in body
    assert "precision" in body
