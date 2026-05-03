import argparse
import logging
import os
import statistics
import sys
import time
from pathlib import Path

import imagehash
import numpy as np

from banger import aesthetic, dedup, face, scenes, select, server, state, taste_head
from banger import develop as develop_mod
from banger.aesthetic import NEGATIVE_PROMPTS, POSITIVE_PROMPTS
from banger.dedup import ClusterItem
from banger.frames import discover_frames
from banger.preview import load_preview
from banger.report import Row, encode_thumbnail, write_report
from banger.sharpness import CONFIG as SHARPNESS_CONFIG
from banger.sharpness import sharpness_from_preview

DEFAULT_TOP_N = int(os.environ.get("BANGER_TOP_N", "10"))
PRESETS_DIR = Path(__file__).resolve().parent.parent.parent / "presets"


def _default_output_dir() -> Path | None:
    """Resolve the default --output dir from BANGER_OUTPUT_DIR.

    The env var, when set, can be a fully-qualified path or a parent directory
    that gets a YYYY-MM-DD subdir appended. Examples:
      BANGER_OUTPUT_DIR=~/Pictures/bangers          -> ~/Pictures/bangers/2026-05-02
      BANGER_OUTPUT_DIR=~/Pictures/bangers/today    -> ~/Pictures/bangers/today (verbatim)

    The "auto-append today" behaviour fires when the env var ends in "bangers"
    or "/", matching CLAUDE.md's expectation; otherwise we use it verbatim.
    """
    raw = os.environ.get("BANGER_OUTPUT_DIR")
    if not raw:
        return None
    import datetime

    p = Path(raw).expanduser()
    if raw.endswith(("/", "\\")) or p.name == "bangers":
        p = p / datetime.date.today().isoformat()
    return p


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="banger")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the pipeline against a folder of images.")
    run.add_argument("input_dir", type=Path, help="Folder containing JPEGs and/or ARWs.")
    run.add_argument("-r", "--recursive", action="store_true", help="Walk subdirectories.")
    run.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write a self-contained HTML preview report to this path.",
    )
    run.add_argument(
        "--output",
        type=Path,
        default=_default_output_dir(),
        help=(
            "Write top-N developed frames (and manifest.json) into this directory. "
            "Defaults to $BANGER_OUTPUT_DIR (with a YYYY-MM-DD subdir appended when "
            "the env value ends in /, \\ or 'bangers')."
        ),
    )
    run.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of frames to write to --output (default {DEFAULT_TOP_N}, env BANGER_TOP_N).",
    )
    run.add_argument(
        "--face-gate",
        action="store_true",
        help=(
            "Also reject frames whose detected faces are softer than "
            f"face.FACE_SHARPNESS_THRESHOLD (={face.FACE_SHARPNESS_THRESHOLD}). "
            "Frames without a detected face still go through the global gate only."
        ),
    )
    run.add_argument(
        "--diversity",
        type=float,
        default=0.5,
        help=(
            "Diversity lambda for top-N selection in [0, 1]. 1.0 = plain top-K "
            "by aesthetic; 0.5 = balanced MMR; 0.0 = pure visual-diversity. "
            "Lower values spread picks across visually distinct frames."
        ),
    )

    explain = sub.add_parser(
        "explain",
        help="Print the per-prompt cosine similarity breakdown for one or more frames.",
    )
    explain.add_argument("input_dir", type=Path)
    explain.add_argument("-r", "--recursive", action="store_true")
    explain.add_argument("stems", nargs="+", help="Frame stems, e.g. DSC00055 DSC00073.")

    label = sub.add_parser(
        "label",
        help="Record an integer score (-5..+5) for one or more frames by stem.",
    )
    label.add_argument("input_dir", type=Path)
    label.add_argument("-r", "--recursive", action="store_true")
    label.add_argument(
        "score",
        type=int,
        help=f"Integer in [{state.SCORE_MIN}, {state.SCORE_MAX}].",
    )
    label.add_argument("stems", nargs="+", help="Frame stems to label with this score.")

    sub.add_parser(
        "train",
        help="Train a Ridge taste head on labelled CLIP embeddings.",
    )

    sub.add_parser(
        "version",
        help="Print package version + key dep versions + state-dir summary.",
    )

    labels = sub.add_parser("labels", help="Inspect / export / import the label DB.")
    labels_sub = labels.add_subparsers(dest="labels_action", required=True)
    labels_sub.add_parser("list", help="Print all labels as a table.")
    exp = labels_sub.add_parser("export", help="Export labels to CSV.")
    exp.add_argument("path", type=Path)
    imp = labels_sub.add_parser("import", help="Import labels from CSV.")
    imp.add_argument("path", type=Path)

    cache = sub.add_parser("cache", help="Inspect / clear on-disk caches (not labels).")
    cache_sub = cache.add_subparsers(dest="cache_action", required=True)
    cache_sub.add_parser("stats", help="Print counts + sizes for each cache.")
    clear = cache_sub.add_parser(
        "clear", help="Delete cached files. Use --kind repeatedly or --all."
    )
    clear.add_argument(
        "--kind",
        action="append",
        choices=["embeddings", "thumbs", "previews", "metadata"],
        default=[],
    )
    clear.add_argument("--all", action="store_true", help="Clear every cache.")

    ui = sub.add_parser(
        "ui",
        help="Start the local labelling UI (http://127.0.0.1:<port>).",
    )
    ui.add_argument("input_dir", type=Path)
    ui.add_argument("--port", type=int, default=8000)

    return parser


def cmd_run(
    input_dir: Path,
    recursive: bool,
    report_path: Path | None,
    output_dir: Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    face_gate: bool = False,
    diversity: float = 0.5,
) -> int:
    log = logging.getLogger("banger")
    if not input_dir.is_dir():
        log.error("not a directory: %s", input_dir)
        return 2

    frames = discover_frames(input_dir, recursive=recursive)
    if not frames:
        log.warning("no supported images in %s", input_dir)
        return 0

    threshold = SHARPNESS_CONFIG["threshold"]
    head = taste_head.load()
    log.info(
        "processing %d frames (sharpness threshold=%.1f, aesthetic=%s, recursive=%s, report=%s)",
        len(frames),
        threshold,
        "trained head" if head is not None else "prompts",
        recursive,
        report_path or "off",
    )

    rows: list[Row] = []
    # Carry the in-memory embedding alongside dedup metadata so the scene
    # classifier doesn't have to re-read it from disk per cluster best.
    dedup_inputs: list[tuple[Row, ClusterItem, np.ndarray]] = []
    aesthetic_done = 0
    aesthetic_skipped = 0
    cache_hits = 0
    face_gated = 0  # rejected by face-aware gate (in addition to global gate)
    t0 = time.monotonic()
    for f in frames:
        sha = state.sha256_of(f.classify_path)
        cached_meta = state.load_frame_metadata(sha)
        cached_emb = state.load_embedding(sha)

        # If the face-gate is on, the cache also needs face_count + face_sharpness.
        face_data_ok = (not face_gate) or (
            cached_meta is not None
            and "face_count" in cached_meta
            and "face_sharpness" in cached_meta
        )

        # Fast path: every per-frame input we need is on disk → don't decode.
        cache_hit = (
            cached_meta is not None
            and "phash_hex" in cached_meta
            and cached_emb is not None
            and face_data_ok
            and report_path is None  # thumb requires preview
        )

        face_count = (cached_meta or {}).get("face_count", 0)
        face_sharp = (cached_meta or {}).get("face_sharpness", 0.0)

        if cache_hit:
            sharp = float(cached_meta["sharpness"])
            if sharp < threshold:
                rows.append(
                    Row(
                        frame=f,
                        sharpness=sharp,
                        aesthetic=None,
                        aesthetic_breakdown=None,
                        aesthetic_source=None,
                        thumb_b64="",
                    )
                )
                cache_hits += 1
                log.info(
                    "REJECT sharpness=%7.1f aesthetic= ---  %s  [%s] (cached)",
                    sharp, f.display_name, f.kind,
                )
                continue
            phash = imagehash.hex_to_hash(cached_meta["phash_hex"])
            ts = float(cached_meta["timestamp"])
            emb = cached_emb
            try:
                prompt_score, breakdown = aesthetic.score_from_embedding(emb)
                a_score = (
                    taste_head.predict_score(head, emb) if head is not None
                    else prompt_score
                )
                source = "head" if head is not None else "prompts"
                aesthetic_done += 1
                cache_hits += 1
            except Exception as e:
                log.warning("aesthetic skip %s: %s", f.display_name, e)
                aesthetic_skipped += 1
                a_score = None
                breakdown = None
                source = None
        else:
            try:
                preview = load_preview(f.classify_path)
            except Exception as e:
                log.warning("skip %s: %s", f.display_name, e)
                continue

            sharp = sharpness_from_preview(preview)
            a_score = None
            breakdown = None
            source = None
            phash = None
            ts = 0.0
            emb = None
            if sharp >= threshold:
                try:
                    emb = aesthetic.encode_image(preview)
                    state.cache_embedding(sha, emb)
                    prompt_score, breakdown = aesthetic.score_from_embedding(emb)
                    if head is not None:
                        a_score = taste_head.predict_score(head, emb)
                        source = "head"
                    else:
                        a_score = prompt_score
                        source = "prompts"
                    phash = dedup.phash_from_preview(preview)
                    ts = dedup.best_timestamp(f.classify_path)
                    # Always populate face data on cold path so future warm runs
                    # don't need a preview reload even if face-gate is later asked for.
                    face_count, face_sharp = face.best_face_sharpness(preview)
                    state.cache_frame_metadata(
                        sha, sharp, str(phash), ts,
                        face_count=face_count, face_sharpness=face_sharp,
                    )
                    aesthetic_done += 1
                except Exception as e:
                    log.warning("aesthetic skip %s: %s", f.display_name, e)
                    aesthetic_skipped += 1
            thumb = encode_thumbnail(preview) if report_path else ""

        # Face-aware gate: only kicks in when --face-gate is set AND the frame
        # contains a face AND the sharpest face is below the per-face threshold.
        if (
            face_gate
            and a_score is not None
            and face_count > 0
            and face_sharp < face.FACE_SHARPNESS_THRESHOLD
        ):
            face_gated += 1
            # Pretend the frame failed the global gate so it lands in REJECT.
            sharp = min(sharp, threshold - 0.01)
            a_score = None
            breakdown = None
            source = None
            emb = None
            phash = None
            log.info(
                "REJECT (face soft, %.1f < %.1f) %s",
                face_sharp, face.FACE_SHARPNESS_THRESHOLD, f.display_name,
            )

        if not cache_hit:
            row = Row(
                frame=f,
                sharpness=sharp,
                aesthetic=a_score,
                aesthetic_breakdown=breakdown,
                aesthetic_source=source,
                thumb_b64=thumb,
            )
            rows.append(row)
        else:
            row = Row(
                frame=f,
                sharpness=sharp,
                aesthetic=a_score,
                aesthetic_breakdown=breakdown,
                aesthetic_source=source,
                thumb_b64="",
            )
            rows.append(row)

        if phash is not None and a_score is not None and emb is not None:
            dedup_inputs.append(
                (row, ClusterItem(key=f.display_name, phash=phash, timestamp=ts, score=a_score), emb)
            )

        a_str = f"aesthetic={a_score:5.2f}" if a_score is not None else "aesthetic= ---"
        flag = "KEEP  " if sharp >= threshold else "REJECT"
        suffix = " (cached)" if cache_hit else ""
        log.info(
            "%s sharpness=%7.1f %s  %s  [%s]%s",
            flag, sharp, a_str, f.display_name, f.kind, suffix,
        )

    # Burst dedup: cluster by pHash + timestamp, mark non-best siblings.
    rows_by_key = {ci.key: row for row, ci, _ in dedup_inputs}
    clusters = dedup.cluster_bursts([ci for _, ci, _ in dedup_inputs])
    bursts = [c for c in clusters if len(c) > 1]
    suppressed_count = 0
    for cid, cluster in enumerate(bursts, start=1):
        best = cluster.best
        for it in cluster.items:
            row = rows_by_key[it.key]
            row.cluster_id = cid
            row.cluster_size = len(cluster)
            row.cluster_best = it.key == best.key
            if not row.cluster_best:
                suppressed_count += 1

    # Scene classification: only for frames that survive both gates.
    scene_done = 0
    for row, _ci, emb in dedup_inputs:
        if not row.cluster_best:
            continue
        try:
            match = scenes.classify(emb)
        except Exception as e:
            log.warning("scene classify skip %s: %s", row.frame.display_name, e)
            continue
        row.scene_preset = match.preset
        row.scene_score = match.score
        row.scene_fell_back = match.fell_back
        scene_done += 1

    elapsed = time.monotonic() - t0
    sharps = [r.sharpness for r in rows]
    aesthetic_vals = [r.aesthetic for r in rows if r.aesthetic is not None]
    sharp_kept = sum(1 for r in rows if r.sharpness >= threshold)
    final_kept = sum(1 for r in rows if r.sharpness >= threshold and r.cluster_best)

    log.info(
        "summary: %d final kept (= %d sharpness-pass − %d burst dupes), %d rejected of "
        "%d frames in %.1fs (aesthetic done=%d skipped=%d, %d bursts found, "
        "%d scenes classified, %d cache hits, %d face-gated)",
        final_kept,
        sharp_kept,
        suppressed_count,
        len(rows) - sharp_kept,
        len(rows),
        elapsed,
        aesthetic_done,
        aesthetic_skipped,
        len(bursts),
        scene_done,
        cache_hits,
        face_gated,
    )

    if scene_done:
        preset_counts: dict[str, int] = {}
        fell_back = 0
        for r in rows:
            if r.scene_preset:
                preset_counts[r.scene_preset] = preset_counts.get(r.scene_preset, 0) + 1
            if r.scene_fell_back:
                fell_back += 1
        ordered = sorted(preset_counts.items(), key=lambda kv: -kv[1])
        log.info(
            "preset assignments: %s (%d fell back to default)",
            ", ".join(f"{name}={n}" for name, n in ordered),
            fell_back,
        )
    if sharps:
        log.info(
            "sharpness stats: min=%.1f median=%.1f max=%.1f",
            min(sharps),
            statistics.median(sharps),
            max(sharps),
        )
    if aesthetic_vals:
        log.info(
            "aesthetic stats: min=%.2f median=%.2f max=%.2f",
            min(aesthetic_vals),
            statistics.median(aesthetic_vals),
            max(aesthetic_vals),
        )

    if report_path is not None and rows:
        write_report(report_path, rows, threshold)
        log.info("wrote report: %s", report_path)

    if output_dir is not None:
        # Build pool of (Row, score, embedding) for cluster-best survivors.
        embs_by_row_id = {id(row): emb for row, _ci, emb in dedup_inputs}
        candidates: list[tuple[Row, float, "np.ndarray"]] = []
        for r in rows:
            if r.sharpness < threshold or not r.cluster_best:
                continue
            score = r.aesthetic if r.aesthetic is not None else r.sharpness
            emb = embs_by_row_id.get(id(r))
            if emb is None:
                continue
            candidates.append((r, score, emb))

        if not candidates:
            log.warning("no frames qualified for output: %s", output_dir)
        else:
            log.info(
                "selecting top %d from %d candidates (diversity lambda=%.2f)",
                top_n, len(candidates), diversity,
            )
            chosen = select.select_diverse_top_n(
                candidates, n=top_n, diversity_lambda=diversity
            )
            top_rows = [row for row, _s, _e in chosen]
            _write_output(output_dir, top_rows, presets_dir=PRESETS_DIR)
    return 0


def _write_output(output_dir: Path, top_rows: list[Row], presets_dir: Path) -> None:
    log = logging.getLogger("banger")
    output_dir.mkdir(parents=True, exist_ok=True)
    dt_cli = develop_mod.find_darktable()
    if dt_cli is None:
        log.info(
            "darktable-cli not found — copy fallback for top %d to %s",
            len(top_rows),
            output_dir,
        )
    else:
        log.info(
            "developing top %d to %s (darktable-cli: %s, presets: %s)",
            len(top_rows),
            output_dir,
            dt_cli,
            presets_dir if presets_dir.is_dir() else "(none)",
        )

    used_counts: dict[str, int] = {}
    manifest_entries: list[dict] = []
    for rank, r in enumerate(top_rows, start=1):
        src = r.frame.develop_path
        # Output filename: NN_subdir_stem.jpg so the directory listing is sorted
        # by rank and obviously identifies the source.
        name_parts = []
        if r.frame.subdir:
            name_parts.append(r.frame.subdir.replace("/", "_"))
        name_parts.append(r.frame.stem)
        out_name = f"{rank:02d}_{'__'.join(name_parts)}.jpg"
        dst = output_dir / out_name

        result = develop_mod.develop_to_jpeg(
            src=src,
            dst=dst,
            preset_name=r.scene_preset,
            presets_dir=presets_dir,
            darktable_cli=dt_cli,
        )
        used_counts[result.used] = used_counts.get(result.used, 0) + 1
        manifest_entries.append(
            {
                "rank": rank,
                "stem": r.frame.stem,
                "subdir": r.frame.subdir,
                "src_path": str(src),
                "output_path": str(dst.relative_to(output_dir)),
                "kind": r.frame.kind,
                "sharpness": round(r.sharpness, 1),
                "aesthetic": round(r.aesthetic, 3) if r.aesthetic is not None else None,
                "aesthetic_source": r.aesthetic_source,
                "scene_preset": r.scene_preset,
                "scene_score": round(r.scene_score, 4) if r.scene_score is not None else None,
                "scene_fell_back": r.scene_fell_back,
                "cluster_size": r.cluster_size,
                "developed_with": result.used,
                "develop_note": result.note,
            }
        )

    manifest_path = output_dir / "manifest.json"
    develop_mod.write_manifest(manifest_path, manifest_entries)
    log.info(
        "wrote %d frames to %s (%s); manifest at %s",
        len(top_rows),
        output_dir,
        ", ".join(f"{k}={v}" for k, v in used_counts.items()),
        manifest_path,
    )


def cmd_explain(input_dir: Path, recursive: bool, stems: list[str]) -> int:
    log = logging.getLogger("banger")
    if not input_dir.is_dir():
        log.error("not a directory: %s", input_dir)
        return 2

    frames = discover_frames(input_dir, recursive=recursive)
    by_stem: dict[str, list] = {}
    for f in frames:
        by_stem.setdefault(f.stem, []).append(f)

    selected = []
    for s in stems:
        matches = by_stem.get(s, [])
        if not matches:
            log.error("not found: %s", s)
            return 2
        if len(matches) > 1:
            log.error(
                "ambiguous stem %s — appears in: %s",
                s,
                ", ".join(m.display_name for m in matches),
            )
            return 2
        selected.append((s, matches[0]))

    results: list[tuple[str, float, dict[str, float]]] = []
    for stem, f in selected:
        preview = load_preview(f.classify_path)
        score, breakdown = aesthetic.score_from_preview(preview)
        results.append((stem, score, breakdown))

    all_prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
    label_w = 70
    col_w = 11
    print()
    print(f"{'prompt':<{label_w}}", end="")
    for stem, _, _ in results:
        print(f"{stem:>{col_w}}", end="")
    print()
    print("-" * (label_w + col_w * len(results)))
    for prompt in all_prompts:
        sign = "+" if prompt in POSITIVE_PROMPTS else "-"
        text = f"{sign} {prompt}"
        if len(text) > label_w - 2:
            text = text[: label_w - 3] + "..."
        print(f"{text:<{label_w}}", end="")
        for _, _, breakdown in results:
            print(f"{breakdown[prompt]:>{col_w}.4f}", end="")
        print()
    print("-" * (label_w + col_w * len(results)))
    n_pos = len(POSITIVE_PROMPTS)
    print(f"{'mean(positives)':<{label_w}}", end="")
    for _, _, b in results:
        print(f"{sum(list(b.values())[:n_pos]) / n_pos:>{col_w}.4f}", end="")
    print()
    print(f"{'mean(negatives)':<{label_w}}", end="")
    for _, _, b in results:
        print(f"{sum(list(b.values())[n_pos:]) / (len(b) - n_pos):>{col_w}.4f}", end="")
    print()
    print(f"{'score (pos-neg) * 50':<{label_w}}", end="")
    for _, score, _ in results:
        print(f"{score:>{col_w}.4f}", end="")
    print()
    return 0


def cmd_label(input_dir: Path, recursive: bool, score: int, stems: list[str]) -> int:
    log = logging.getLogger("banger")
    if not input_dir.is_dir():
        log.error("not a directory: %s", input_dir)
        return 2
    if not state.SCORE_MIN <= score <= state.SCORE_MAX:
        log.error(
            "score must be in [%d, %d], got %d", state.SCORE_MIN, state.SCORE_MAX, score
        )
        return 2

    frames = discover_frames(input_dir, recursive=recursive)
    by_stem: dict[str, list] = {}
    for f in frames:
        by_stem.setdefault(f.stem, []).append(f)

    for s in stems:
        matches = by_stem.get(s, [])
        if not matches:
            log.error("not found: %s", s)
            return 2
        if len(matches) > 1:
            log.error(
                "ambiguous stem %s — use the UI for recursive labelling. Found: %s",
                s,
                ", ".join(m.display_name for m in matches),
            )
            return 2
        f = matches[0]
        sha = state.sha256_of(f.classify_path)
        state.add_label(sha, score, f.stem, str(f.classify_path))
        log.info("labelled %s as %+d", f.display_name, score)

    return 0


def cmd_train() -> int:
    return 0 if taste_head.train_from_disk() is not None else 2


def cmd_ui(input_dir: Path, port: int) -> int:
    server.serve(input_dir, port=port)
    return 0


def cmd_labels_list() -> int:
    rows = state.all_labels()
    if not rows:
        print("(no labels)")
        return 0
    print(f"{'score':>6}  {'stem':<32}  {'sha':<16}  {'src_path'}")
    print("-" * 80)
    for sha, score, stem, src, _ts in rows:
        sign = "+" if score >= 0 else ""
        print(f"{sign}{score:>5}  {stem[:32]:<32}  {sha[:16]:<16}  {src}")
    print(f"\n{len(rows)} labels.")
    return 0


def cmd_labels_export(path: Path) -> int:
    n = state.export_labels_csv(path)
    log = logging.getLogger("banger")
    log.info("exported %d labels to %s", n, path)
    return 0


def cmd_labels_import(path: Path) -> int:
    log = logging.getLogger("banger")
    if not path.is_file():
        log.error("not a file: %s", path)
        return 2
    imported, skipped = state.import_labels_csv(path)
    log.info("imported %d labels from %s (skipped %d malformed)", imported, path, skipped)
    return 0


def cmd_cache_stats() -> int:
    stats = state.cache_stats()
    print(f"{'cache':<12} {'count':>8} {'size':>10}  path")
    print("-" * 80)
    total_bytes = 0
    total_count = 0
    for name, info in stats.items():
        size_mb = info["bytes"] / (1024 * 1024)
        size_str = f"{size_mb:>9.1f}M" if size_mb >= 1 else f"{info['bytes']:>9}B"
        print(f"{name:<12} {info['count']:>8} {size_str}  {info['path']}")
        total_bytes += info["bytes"]
        total_count += info["count"]
    print("-" * 80)
    print(f"{'TOTAL':<12} {total_count:>8} {total_bytes / (1024 * 1024):>9.1f}M")
    return 0


def cmd_cache_clear(kinds: list[str], all_caches: bool) -> int:
    log = logging.getLogger("banger")
    if all_caches:
        kinds = ["embeddings", "thumbs", "previews", "metadata"]
    if not kinds:
        log.error("specify at least one --kind, or --all")
        return 2
    removed = state.clear_cache(kinds)
    for k, n in removed.items():
        log.info("cleared %s: %d files removed", k, n)
    return 0


def cmd_version() -> int:
    import sys

    from banger import __version__

    print(f"banger {__version__}")
    print(f"python {sys.version.split()[0]} ({sys.executable})")

    deps = []
    for mod_name, label in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("cv2", "opencv-python"),
        ("rawpy", "rawpy"),
        ("imagehash", "imagehash"),
        ("sklearn", "scikit-learn"),
        ("flask", "flask"),
    ):
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "?")
            deps.append(f"  {label}: {ver}")
        except ImportError:
            deps.append(f"  {label}: (not installed)")
    print("dependencies:")
    print("\n".join(deps))

    try:
        import torch

        if torch.cuda.is_available():
            print(f"cuda: {torch.cuda.get_device_name(0)}")
        else:
            print("cuda: (not available)")
    except ImportError:
        pass

    dt = develop_mod.find_darktable()
    print(f"darktable-cli: {dt or '(not found)'}")

    head = taste_head.exists()
    labels = state.labels_dict()
    pos = sum(1 for s in labels.values() if s > 0)
    zero = sum(1 for s in labels.values() if s == 0)
    neg = sum(1 for s in labels.values() if s < 0)
    stats = state.cache_stats()
    print(f"taste head: {'present' if head else 'absent'}")
    print(f"labels: {len(labels)} total ({pos} positive, {zero} zero, {neg} negative)")
    print(
        f"caches: embeddings={stats['embeddings']['count']}, "
        f"thumbs={stats['thumbs']['count']}, "
        f"previews={stats['previews']['count']}, "
        f"metadata={stats['metadata']['count']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(
            args.input_dir,
            args.recursive,
            args.report,
            output_dir=args.output,
            top_n=args.top_n,
            face_gate=args.face_gate,
            diversity=args.diversity,
        )
    if args.command == "explain":
        return cmd_explain(args.input_dir, args.recursive, args.stems)
    if args.command == "label":
        return cmd_label(args.input_dir, args.recursive, args.score, args.stems)
    if args.command == "train":
        return cmd_train()
    if args.command == "ui":
        return cmd_ui(args.input_dir, args.port)
    if args.command == "labels":
        if args.labels_action == "list":
            return cmd_labels_list()
        if args.labels_action == "export":
            return cmd_labels_export(args.path)
        if args.labels_action == "import":
            return cmd_labels_import(args.path)
    if args.command == "cache":
        if args.cache_action == "stats":
            return cmd_cache_stats()
        if args.cache_action == "clear":
            return cmd_cache_clear(args.kind, args.all)
    if args.command == "version":
        return cmd_version()
    return 1


if __name__ == "__main__":
    sys.exit(main())
