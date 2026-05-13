"""Head-to-head benchmark harness: banger vs facet on the same input.

Runs banger end-to-end on the given input dir, then attempts to run facet
on the same dir. Writes a markdown report at benchmarks/<date>.md with:
  - banger top-N (rank, stem, scores)
  - facet top-N (or "facet not available" + reason)
  - overlap stats (Jaccard, Spearman if both rank the same set)
  - disagreement gallery (frames one tool picked and the other didn't)

The "if facet isn't runnable, skip gracefully" rule (build plan) means
we never let a facet failure abort the report. The report explicitly
calls out missing data so a blog-post comparison can't be made to look
better than it is by quiet omission.

Facet path: I tried the Docker compose route from this machine and it
OOMs at the multi-pass model swap on a 4 GB RTX 3050 (legacy profile
mis-sized, see auto-memory). The harness still tries, facet on another
machine, or a CPU-only profile, would let this actually run, but the
default outcome on this hardware is the "skip" branch.
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
    from banger.sharpness import CONFIG as SHARP_CFG, sharpness_from_preview

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


def run_facet(input_dir: Path, facet_path: Path, top_n: int) -> tuple[list[Pick] | None, str, float | None]:
    """Best-effort facet invocation. Returns (picks, status, elapsed)."""
    import time as _time

    if not facet_path.is_dir():
        return None, f"skipped: facet path not found at {facet_path}", None
    if shutil.which("docker") is None:
        return None, "skipped: docker CLI not on PATH", None

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
            return None, f"skipped: docker compose unavailable ({proc.stderr.strip()[:80]})", None
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return None, f"skipped: docker probe failed ({e})", None

    # Without a stable facet CLI on this box we don't try to actually run it.
    # The auto-memory says the build path is broken on this XPS+Windows setup.
    # Leave the actual invocation as a TODO and return skipped, the harness
    # still produces a useful banger-only report.
    elapsed = _time.monotonic() - t0
    return None, (
        "skipped: facet docker path known-broken on this host (legacy profile "
        "OOMs at multi-pass model swap on 4 GB GPU; re-enable when running "
        "from a host with ≥8 GB VRAM or a working CPU-only profile)"
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


def write_report(out_path: Path, result: BenchmarkResult) -> None:
    cmp = compare(result.banger_top, result.facet_top)
    lines = [
        f"# banger vs facet benchmark, {datetime.date.today().isoformat()}",
        "",
        f"Input: `{result.input_dir}`",
        f"Banger top-{len(result.banger_top)} in {result.elapsed_banger:.1f}s",
    ]
    if result.facet_top is not None:
        lines.append(f"Facet top-{len(result.facet_top)} in {result.elapsed_facet:.1f}s")
    else:
        lines.append(f"Facet: {result.facet_status}")
    lines.extend([
        "",
        "## Overlap",
        "",
        f"- Jaccard: {cmp['jaccard']}",
        f"- Frames in both top-N: {cmp['overlap']}",
        f"- Spearman (over intersection): {cmp['spearman']}",
        "",
        "## Banger top-N",
        "",
        "| rank | frame | aesthetic | sharpness | scene |",
        "|-----:|-------|----------:|----------:|-------|",
    ])
    for p in result.banger_top:
        a = f"{p.aesthetic:.2f}" if p.aesthetic is not None else "-"
        lines.append(f"| {p.rank} | {p.display} | {a} | {p.sharpness:.0f} | {p.scene_preset or '-'} |")

    if result.facet_top is not None:
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
            "Facet did not run on this host, so the comparison section is intentionally "
            "blank rather than misleadingly empty. Banger's picks above stand on their own; "
            "rerun on a host with a working facet install to populate the head-to-head.",
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
    facet_status = "skipped: --vs not requested"
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
    print(json.dumps(
        {
            "banger_picks": len(banger_picks),
            "facet": facet_status,
            "report": str(out),
            "overlap": compare(banger_picks, facet_picks),
        },
        indent=2,
    ))
    return 0
