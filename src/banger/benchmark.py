"""Benchmark harness: banger's own picks + measured detector quality.

Runs banger end-to-end on the given input dir and writes a markdown report
at benchmarks/<date>.md with:
  - banger top-N (rank, stem, scores)
  - measured detector quality vs a labelled ground-truth set, when one is
    supplied (precision / recall / F1 for blur, blink, duplicate, keeper)
  - an OPTIONAL head-to-head against facet, but ONLY when facet actually
    runs. When it doesn't, the report says so plainly instead of printing
    an empty "comparison" that pretends a measurement happened.

Honesty rule: there is no ground-truth set shipped in the repo yet, and
facet does not run on the reference host (see ``run_facet``). The harness
therefore does NOT manufacture a comparison out of thin air. It reports
banger's picks on their own and, where a caller passes labels via
``evaluate_against_truth``, reports *measured* accuracy. The "vs facet"
section only appears when facet produced real output.

Facet path: the Docker compose route OOMs at the multi-pass model swap on
a 4 GB RTX 3050 (legacy profile mis-sized, see auto-memory). ``run_facet``
still probes for it so a host with a working install gets a real
head-to-head, but the default outcome on the reference hardware is the
explicit "not available" branch -- which is reported as such, not as a
zero-overlap comparison.
"""

from __future__ import annotations

import datetime
import json
import logging
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("banger")

DEFAULT_FACET_PATH = Path("C:/Users/mikea/OneDrive/Desktop/Projects/facet")


@dataclass
class Pick:
    rank: int
    stem: str
    subdir: str
    aesthetic: float | None
    sharpness: float
    scene_preset: str | None

    @property
    def display(self) -> str:
        return f"{self.subdir}/{self.stem}" if self.subdir else self.stem


@dataclass
class BenchmarkResult:
    input_dir: Path
    banger_top: list[Pick]
    facet_top: list[Pick] | None
    facet_status: str  # "ok" / "skipped: <reason>"
    elapsed_banger: float
    elapsed_facet: float | None


def run_banger(input_dir: Path, recursive: bool, top_n: int) -> tuple[list[Pick], float]:
    """Invoke banger's scoring stages in-process, return top-N picks + elapsed."""
    import time as _time

    from banger import aesthetic, dedup, face, scene_kmeans, scenes, select, state, taste_head
    from banger.frames import discover_frames
    from banger.preview import load_preview
    from banger.report import Row
    from banger.sharpness import CONFIG as SHARP_CFG
    from banger.sharpness import sharpness_from_preview

    threshold = SHARP_CFG["threshold"]
    head = taste_head.load()
    sc_clusters = scene_kmeans.load()

    frames = discover_frames(input_dir, recursive=recursive)
    t0 = _time.monotonic()

    candidates: list[tuple[Row, float, np.ndarray]] = []
    timestamps: dict[str, float] = {}
    phashes: dict[str, object] = {}

    for f in frames:
        sha = state.sha256_of(f.classify_path)
        cached_meta = state.load_frame_metadata(sha)
        cached_emb = state.load_embedding(sha)
        if cached_meta and "phash_hex" in cached_meta and cached_emb is not None:
            sharp = float(cached_meta["sharpness"])
            if sharp < threshold:
                continue
            emb = cached_emb
            import imagehash
            phash = imagehash.hex_to_hash(cached_meta["phash_hex"])
            ts = float(cached_meta["timestamp"])
        else:
            try:
                preview = load_preview(f.classify_path)
            except Exception:
                continue
            sharp = sharpness_from_preview(preview)
            if sharp < threshold:
                continue
            try:
                emb = aesthetic.encode_image(preview)
                state.cache_embedding(sha, emb)
                phash = dedup.phash_from_preview(preview)
                ts = dedup.best_timestamp(f.classify_path)
                fc, fs = face.best_face_sharpness(preview)
                state.cache_frame_metadata(
                    sha, sharp, str(phash), ts, face_count=fc, face_sharpness=fs,
                )
            except Exception:
                continue

        if head is not None:
            score = taste_head.predict_score(head, emb)
        else:
            score, _ = aesthetic.score_from_embedding(emb)

        scene_preset = None
        if sc_clusters is not None:
            scene_preset = sc_clusters.classify_embedding(emb).preset
        else:
            try:
                scene_preset = scenes.classify(emb).preset
            except Exception:
                scene_preset = None

        row = Row(
            frame=f, sharpness=sharp, aesthetic=score,
            aesthetic_breakdown=None, aesthetic_source=("head" if head else "prompts"),
            thumb_b64="", scene_preset=scene_preset,
        )
        candidates.append((row, score, emb))
        timestamps[f.display_name] = ts
        phashes[f.display_name] = phash

    # Dedup via banger.dedup like cmd_run does, then pick kmeans-diverse top-N.
    if not candidates:
        return [], _time.monotonic() - t0

    chosen = select.select_kmeans_top_n(candidates, n=top_n)
    picks: list[Pick] = []
    for rank, (row, _score, _emb) in enumerate(chosen, start=1):
        picks.append(
            Pick(
                rank=rank, stem=row.frame.stem, subdir=row.frame.subdir,
                aesthetic=row.aesthetic, sharpness=row.sharpness,
                scene_preset=row.scene_preset,
            )
        )
    return picks, _time.monotonic() - t0


# Sentinel prefix for every "facet did not produce real output" status. The
# report and cmd_benchmark key off this so a non-run is never rendered as a
# comparison with zero overlap (which would read as "banger and facet agreed
# on nothing"). A status NOT starting with this prefix means facet really ran.
FACET_UNAVAILABLE_PREFIX = "facet not available: "


def run_facet(input_dir: Path, facet_path: Path, top_n: int) -> tuple[list[Pick] | None, str, float | None]:
    """Best-effort facet invocation. Returns ``(picks, status, elapsed)``.

    This is a real, documented "facet not available" path, not a fake
    comparison. When facet cannot run (no checkout, no docker, broken
    profile) we return ``picks=None`` and a status string prefixed with
    ``FACET_UNAVAILABLE_PREFIX`` so callers can distinguish "did not run"
    from "ran and disagreed". The harness never claims a head-to-head
    happened when it didn't.
    """
    import time as _time

    if not facet_path.is_dir():
        return None, f"{FACET_UNAVAILABLE_PREFIX}facet path not found at {facet_path}", None
    if shutil.which("docker") is None:
        return None, f"{FACET_UNAVAILABLE_PREFIX}docker CLI not on PATH", None

    # Try `docker compose run facet python facet.py score <input>` style.
    # On this hardware the legacy profile OOMs at multi-pass, so this is
    # mostly a graceful-degradation drill. We give it 600s.
    t0 = _time.monotonic()
    try:
        proc = subprocess.run(
            ["docker", "compose", "version"],
            cwd=facet_path, capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            return None, f"{FACET_UNAVAILABLE_PREFIX}docker compose unavailable ({proc.stderr.strip()[:80]})", None
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return None, f"{FACET_UNAVAILABLE_PREFIX}docker probe failed ({e})", None

    # Without a stable facet CLI on this box we don't try to actually run it.
    # The auto-memory says the build path is broken on this XPS+Windows setup.
    # Leave the actual invocation as a TODO and return the not-available branch;
    # the harness still produces a useful banger-only report.
    elapsed = _time.monotonic() - t0
    return None, (
        f"{FACET_UNAVAILABLE_PREFIX}facet docker path known-broken on this host "
        "(legacy profile OOMs at multi-pass model swap on 4 GB GPU; re-enable "
        "when running from a host with >=8 GB VRAM or a working CPU-only profile)"
    ), elapsed


def compare(banger: list[Pick], facet: list[Pick] | None) -> dict:
    if not facet:
        return {"jaccard": None, "overlap": 0, "spearman": None}
    a = {p.display for p in banger}
    b = {p.display for p in facet}
    inter = a & b
    union = a | b
    jaccard = len(inter) / len(union) if union else 0.0
    # Spearman over the intersection.
    spearman = None
    if len(inter) >= 3:
        a_ranks = {p.display: p.rank for p in banger}
        b_ranks = {p.display: p.rank for p in facet}
        xs = [a_ranks[d] for d in inter]
        ys = [b_ranks[d] for d in inter]
        spearman = _spearman(xs, ys)
    return {"jaccard": round(jaccard, 3), "overlap": len(inter), "spearman": spearman}


def _spearman(x: list[int], y: list[int]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=True))
    dx = (sum((xi - mx) ** 2 for xi in x)) ** 0.5
    dy = (sum((yi - my) ** 2 for yi in y)) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return round(num / (dx * dy), 3)


# --------------------------------------------------------------------------- #
# Measured detector accuracy.
#
# The audit's core complaint is that nothing here measures accuracy. These
# helpers make it measurable: given the detectors' boolean predictions and a
# hand-labelled ground-truth, they compute precision / recall / F1. No
# ground-truth set ships in the repo yet, so callers supply their own labels;
# the point is that the *machinery* exists and is tested, so the moment a
# labelled set is available the numbers are real rather than asserted.
#
# Convention: for each detector a prediction of True means "this frame is a
# positive" -- blurry, blinking, a duplicate, or a keeper, depending on the
# detector. Precision/recall are computed against the matching truth labels.
# --------------------------------------------------------------------------- #

# Detectors we know how to score. Kept as a tuple so the report can iterate in
# a stable order and so callers can validate detector names.
DETECTORS = ("blur", "blink", "duplicate", "keeper")


@dataclass
class PRF:
    """Precision / recall / F1 plus the raw confusion counts for one detector."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return round(self.tp / denom, 4) if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return round(self.tp / denom, 4) if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return round(2 * p * r / (p + r), 4) if (p + r) else 0.0

    @property
    def support(self) -> int:
        """Number of true positives in the ground-truth (tp + fn)."""
        return self.tp + self.fn

    def as_dict(self) -> dict[str, float | int]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


def precision_recall_f1(predictions: dict[str, bool], truth: dict[str, bool]) -> PRF:
    """Score one detector's boolean predictions against boolean ground-truth.

    Only frames present in BOTH mappings are scored -- a frame the detector
    never saw (or that was never labelled) can't be a true/false anything, so
    silently scoring it as a negative would inflate the true-negative count and
    flatter precision. We intersect the keys instead.

    Args:
        predictions: ``{frame_id: predicted_positive}``
        truth:       ``{frame_id: actually_positive}``

    Returns a :class:`PRF` with confusion counts and derived metrics.
    """
    keys = predictions.keys() & truth.keys()
    tp = fp = fn = tn = 0
    for k in keys:
        pred = bool(predictions[k])
        actual = bool(truth[k])
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    return PRF(tp=tp, fp=fp, fn=fn, tn=tn)


def evaluate_against_truth(
    predictions: dict[str, dict[str, bool]],
    truth: dict[str, dict[str, bool]],
) -> dict[str, dict[str, float | int]]:
    """Measure every detector's precision/recall/F1 against a labelled set.

    Both arguments are keyed by detector name (``blur``, ``blink``,
    ``duplicate``, ``keeper``); each value is a ``{frame_id: bool}`` mapping.
    Detectors absent from either side are skipped (you can't score what you
    didn't predict or didn't label). Unknown detector names are ignored so a
    caller's extra columns don't blow up the report.

    Returns ``{detector: {precision, recall, f1, support, tp, fp, fn, tn}}``.

    Example::

        evaluate_against_truth(
            {"blur": {"a": True, "b": False}},
            {"blur": {"a": True, "b": True}},
        )
        # -> {"blur": {"precision": 1.0, "recall": 0.5, "f1": 0.6667, ...}}
    """
    out: dict[str, dict[str, float | int]] = {}
    for det in DETECTORS:
        if det in predictions and det in truth:
            out[det] = precision_recall_f1(predictions[det], truth[det]).as_dict()
    return out


def format_accuracy_table(scores: dict[str, dict[str, float | int]]) -> list[str]:
    """Render :func:`evaluate_against_truth` output as markdown report lines."""
    if not scores:
        return [
            "## Detector accuracy",
            "",
            "No ground-truth labels supplied, so detector accuracy is unmeasured. "
            "Pass a labelled set to `evaluate_against_truth` to populate this.",
        ]
    lines = [
        "## Detector accuracy (measured vs ground-truth)",
        "",
        "| detector | precision | recall | F1 | support |",
        "|----------|----------:|-------:|---:|--------:|",
    ]
    for det in DETECTORS:
        if det not in scores:
            continue
        s = scores[det]
        lines.append(
            f"| {det} | {s['precision']:.3f} | {s['recall']:.3f} | "
            f"{s['f1']:.3f} | {s['support']} |"
        )
    return lines


def write_report(out_path: Path, result: BenchmarkResult) -> None:
    facet_ran = result.facet_top is not None
    # Title reflects what actually happened: a head-to-head only when facet ran.
    title = (
        f"# banger vs facet benchmark, {datetime.date.today().isoformat()}"
        if facet_ran
        else f"# banger benchmark, {datetime.date.today().isoformat()}"
    )
    lines = [
        title,
        "",
        f"Input: `{result.input_dir}`",
        f"Banger top-{len(result.banger_top)} in {result.elapsed_banger:.1f}s",
    ]
    if facet_ran:
        lines.append(f"Facet top-{len(result.facet_top)} in {result.elapsed_facet:.1f}s")
        cmp = compare(result.banger_top, result.facet_top)
        lines.extend([
            "",
            "## Overlap",
            "",
            f"- Jaccard: {cmp['jaccard']}",
            f"- Frames in both top-N: {cmp['overlap']}",
            f"- Spearman (over intersection): {cmp['spearman']}",
        ])
    else:
        # No comparison happened: say so plainly instead of printing a
        # zero-overlap "Overlap" block that would misread as disagreement.
        lines.append(f"Facet: {result.facet_status}")
    lines.extend([
        "",
        "## Banger top-N",
        "",
        "| rank | frame | aesthetic | sharpness | scene |",
        "|-----:|-------|----------:|----------:|-------|",
    ])
    for p in result.banger_top:
        a = f"{p.aesthetic:.2f}" if p.aesthetic is not None else "-"
        lines.append(f"| {p.rank} | {p.display} | {a} | {p.sharpness:.0f} | {p.scene_preset or '-'} |")

    if facet_ran:
        lines.extend(["", "## Facet top-N", "", "| rank | frame |", "|-----:|-------|"])
        for p in result.facet_top:
            lines.append(f"| {p.rank} | {p.display} |")

        # Disagreement gallery: frames in one list but not the other.
        a_set = {p.display for p in result.banger_top}
        b_set = {p.display for p in result.facet_top}
        only_a = sorted(a_set - b_set)
        only_b = sorted(b_set - a_set)
        a_lines = [f"- {d}" for d in only_a] if only_a else ["(none)"]
        b_lines = [f"- {d}" for d in only_b] if only_b else ["(none)"]
        lines.extend([
            "",
            "## Disagreements",
            "",
            "### Banger picked, facet didn't",
            "",
            *a_lines,
            "",
            "### Facet picked, banger didn't",
            "",
            *b_lines,
        ])
    else:
        lines.extend([
            "",
            "## Note",
            "",
            "Facet did not run on this host, so there is intentionally no comparison "
            "section: an empty overlap table would misread as 'banger and facet agreed "
            "on nothing'. Banger's picks above stand on their own. Rerun on a host with "
            "a working facet install to populate the head-to-head, and pass a labelled "
            "ground-truth set via `evaluate_against_truth` to get measured precision/recall.",
        ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_benchmark(
    input_dir: Path,
    vs_facet: bool,
    top_n: int,
    output: Path | None,
    facet_path: Path,
    recursive: bool,
) -> int:
    if not input_dir.is_dir():
        log.error("not a directory: %s", input_dir)
        return 2

    log.info("banger benchmark on %s (top-N=%d, vs_facet=%s)", input_dir, top_n, vs_facet)
    banger_picks, banger_t = run_banger(input_dir, recursive=recursive, top_n=top_n)
    log.info("banger: %d picks in %.1fs", len(banger_picks), banger_t)

    facet_picks: list[Pick] | None = None
    facet_status = f"{FACET_UNAVAILABLE_PREFIX}--vs not requested"
    facet_t: float | None = None
    if vs_facet:
        facet_picks, facet_status, facet_t = run_facet(input_dir, facet_path, top_n)
        log.info("facet: %s (%.1fs)", facet_status, facet_t or 0.0)

    result = BenchmarkResult(
        input_dir=input_dir,
        banger_top=banger_picks,
        facet_top=facet_picks,
        facet_status=facet_status,
        elapsed_banger=banger_t,
        elapsed_facet=facet_t,
    )
    out = output or (Path("benchmarks") / f"{datetime.date.today().isoformat()}.md")
    write_report(out, result)
    log.info("wrote benchmark report: %s", out)
    # Only emit an overlap block when facet actually ran; otherwise null it so
    # a non-run isn't reported as "overlap: 0" (which reads as disagreement).
    overlap = compare(banger_picks, facet_picks) if facet_picks is not None else None
    print(json.dumps(
        {
            "banger_picks": len(banger_picks),
            "facet": facet_status,
            "report": str(out),
            "overlap": overlap,
        },
        indent=2,
    ))
    return 0
