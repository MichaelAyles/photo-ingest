import argparse
import logging
import os
import statistics
import sys
import time
from pathlib import Path

import imagehash
import numpy as np

from banger import (
    aesthetic,
    dedup,
    face,
    scene_kmeans,
    scenes,
    select,
    server,
    state,
    taste_head,
)
from banger import (
    eyes as eyes_mod,
)
from banger import (
    face_id as face_id_mod,
)
from banger import (
    metrics as metrics_mod,
)
from banger import (
    settings as settings_mod,
)
from banger.aesthetic import NEGATIVE_PROMPTS, POSITIVE_PROMPTS
from banger.dedup import ClusterItem
from banger.frames import discover_frames
from banger.preview import load_preview
from banger.report import Row, encode_thumbnail, write_report
from banger.sharpness import sharpness_from_preview

DEFAULT_TOP_N = int(os.environ.get("BANGER_TOP_N", "10"))


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

    # Gate defaults follow the persisted settings (both ship ON). They no-op
    # gracefully when mediapipe/insightface are absent, so defaulting them on
    # is safe even on a bare install.
    _cfg = settings_mod.load()
    _face_gate_default = bool(_cfg.get("face_gate", True))
    _eye_gate_default = bool(_cfg.get("eye_gate", True))

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
        action=argparse.BooleanOptionalAction,
        default=_face_gate_default,
        help=(
            "Also reject frames whose detected faces are softer than the "
            "configured face-sharpness threshold, or whose eyes aren't tack-sharp "
            "(tunable via the Settings tab). Frames without a detected face still "
            "go through the global gate only. Use --no-face-gate to disable. "
            f"(default {'on' if _face_gate_default else 'off'} from settings)"
        ),
    )
    run.add_argument(
        "--eye-gate",
        action=argparse.BooleanOptionalAction,
        default=_eye_gate_default,
        help=(
            "Reject frames where any detected face has Eye Aspect Ratio below "
            "the configured eye-aspect-ratio threshold (Settings: eye_ear_threshold, "
            f"default {settings_mod.DEFAULTS['eye_ear_threshold']}). Requires mediapipe; "
            "a no-op (and a warning) if mediapipe isn't installed. Use --no-eye-gate "
            f"to disable. (default {'on' if _eye_gate_default else 'off'} from settings)"
        ),
    )
    run.add_argument(
        "--xmp",
        action="store_true",
        help=(
            "Also write a sidecar .xmp next to each source frame with star rating "
            "(top 10%% = 5★, next 20%% = 4★, etc) and a colour label encoding the "
            "scene cluster. Non-destructive: no copying, no developing. Picked up "
            "automatically by darktable/Lightroom on next library scan."
        ),
    )
    run.add_argument(
        "--strategy",
        choices=["topk", "mmr", "kmeans", "faces"],
        default="kmeans",
        help=(
            "Top-N selection strategy. 'topk' = pure aesthetic ranking, "
            "no diversity. 'mmr' = continuous knob via --diversity. "
            "'kmeans' (default) = cluster the candidates by CLIP embedding "
            "into N visual groups and pick the highest-scoring frame from "
            "each. 'faces' = the Aftershoot trick: cluster by face identity "
            "via insightface, pick one good shot per person, fill remainder "
            "with kmeans-on-CLIP. Best when the dataset has recurring people "
            "and you're tired of seeing the same crew in every pick."
        ),
    )
    run.add_argument(
        "--diversity",
        type=float,
        default=0.5,
        help=(
            "Diversity lambda for the MMR strategy in [0, 1]. 1.0 = plain "
            "top-K by aesthetic; 0.5 = balanced; 0.0 = pure visual-diversity. "
            "Only used when --strategy mmr."
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

    scenes_cmd = sub.add_parser("scenes", help="Manage the KMeans scene clusters.")
    scenes_sub = scenes_cmd.add_subparsers(dest="scenes_action", required=True)
    fit_p = scenes_sub.add_parser("fit", help="Cluster cached CLIP embeddings.")
    fit_p.add_argument("-k", type=int, default=5, help="Number of clusters (default 5).")
    scenes_sub.add_parser("show", help="Print the current cluster summary.")

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

    gui = sub.add_parser(
        "gui",
        help="Open the native desktop GUI (pywebview). The user-friendly entrypoint.",
    )
    gui.add_argument("--port", type=int, default=8765)
    gui.add_argument(
        "--headless",
        action="store_true",
        help="Run the Flask backend only, no webview. Useful for debugging / remote.",
    )

    bench = sub.add_parser(
        "benchmark",
        help="Run banger (and optionally facet) on the same input, write a markdown report.",
    )
    bench.add_argument("input_dir", type=Path)
    bench.add_argument("-r", "--recursive", action="store_true")
    bench.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    bench.add_argument("--vs", choices=["facet"], default=None,
                       help="Comparison target. Currently only 'facet'.")
    bench.add_argument("--facet-path", type=Path,
                       default=None,
                       help="Path to the facet checkout (required when --vs facet).")
    bench.add_argument("--output", type=Path, default=None,
                       help="Where to write the markdown report (default benchmarks/<date>.md).")

    return parser


def _blended_score(head, emb: np.ndarray, prior_score: float, n_labels: int) -> float:
    """Personal taste-head score, cold-start-blended with the aesthetic prior.

    When the head was trained on few labels its predictions are noisy, so we
    fold in the generic prompt-based aesthetic prior, weighting toward the
    personal head as the label count grows (saturating to pure-personal at
    taste_head's full-trust label count). With a strong head (many labels) the
    blend collapses to the personal score, so behaviour is unchanged there.
    """
    personal = taste_head.predict_score(head, emb)
    return taste_head.blend_with_prior(personal, prior_score, n_labels=n_labels)


def cmd_run(
    input_dir: Path,
    recursive: bool,
    report_path: Path | None,
    output_dir: Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    face_gate: bool = False,
    eye_gate: bool = False,
    write_xmp: bool = False,
    diversity: float = 0.5,
    strategy: str = "kmeans",
) -> int:
    log = logging.getLogger("banger")
    if not input_dir.is_dir():
        log.error("not a directory: %s", input_dir)
        return 2

    frames = discover_frames(input_dir, recursive=recursive)
    if not frames:
        log.warning("no supported images in %s", input_dir)
        return 0

    cfg = settings_mod.load()
    threshold = float(cfg["sharpness_threshold"])
    face_sharp_threshold = float(cfg["face_sharpness_threshold"])
    eye_ear_threshold = float(cfg["eye_ear_threshold"])
    dedup_enabled = bool(cfg["dedup_enabled"])
    dedup_hamming = int(cfg["dedup_hamming"])
    dedup_time_window = float(cfg["dedup_time_window"])
    head = taste_head.load()
    # Live label count drives the cold-start blend: a head trained on few
    # labels is noisy, so we fold in the generic aesthetic prior (see below).
    n_labels = len(state.labels_dict())

    if eye_gate and not eyes_mod.mediapipe_available():
        log.warning(
            "--eye-gate requested but mediapipe is not installed; "
            "gate disabled (pip install mediapipe to enable)"
        )
        eye_gate = False
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
    eye_gated = 0  # rejected by eye-aware gate
    metrics_done = 0
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
        eye_data_ok = (not eye_gate) or (cached_meta is not None and "eyes" in cached_meta)
        face_id_data_ok = (strategy != "faces") or (
            cached_meta is not None
            and ("face_detections" in cached_meta or "face_embeddings" in cached_meta)
        )

        # Fast path: every per-frame input we need is on disk → don't decode.
        cache_hit = (
            cached_meta is not None
            and "phash_hex" in cached_meta
            and cached_emb is not None
            and face_data_ok
            and eye_data_ok
            and face_id_data_ok
            and report_path is None  # thumb requires preview
        )

        face_count = (cached_meta or {}).get("face_count", 0)
        face_sharp = (cached_meta or {}).get("face_sharpness", 0.0)
        frame_metrics = (cached_meta or {}).get("metrics")
        frame_eyes = (cached_meta or {}).get("eyes")
        # Eye-region focus of the most-prominent face, stashed inside metrics on
        # the cold path (None when no face / not computed). Used by the face
        # gate's eyes-tack-sharp check.
        face_eye_sharp = (frame_metrics or {}).get("face_eye_sharpness")

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
                    _blended_score(head, emb, prompt_score, n_labels)
                    if head is not None
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
            frame_metrics = None
            frame_eyes = None
            face_eye_sharp = None
            if sharp >= threshold:
                try:
                    emb = aesthetic.encode_image(preview)
                    state.cache_embedding(sha, emb)
                    prompt_score, breakdown = aesthetic.score_from_embedding(emb)
                    if head is not None:
                        a_score = _blended_score(head, emb, prompt_score, n_labels)
                        source = "head"
                    else:
                        a_score = prompt_score
                        source = "prompts"
                    phash = dedup.phash_from_preview(preview)
                    ts = dedup.best_timestamp(f.classify_path)
                    # Always populate face data on cold path so future warm runs
                    # don't need a preview reload even if face-gate is later asked for.
                    face_count, face_sharp = face.best_face_sharpness(preview)
                    try:
                        frame_metrics = metrics_mod.compute_all(preview)
                        metrics_done += 1
                    except Exception as e:
                        log.warning("metrics skip %s: %s", f.display_name, e)
                        frame_metrics = None
                    # Eye-region focus of the most-prominent face — the
                    # "eyes tack-sharp" check. Only worth the cost when the
                    # face gate is active; stashed in metrics so warm runs reuse
                    # it. None means no face detected (gate stays silent).
                    if face_gate:
                        try:
                            face_eye_sharp = face.best_face_eye_sharpness(preview)
                        except Exception as e:
                            log.warning("eye-sharpness skip %s: %s", f.display_name, e)
                            face_eye_sharp = None
                        if face_eye_sharp is not None:
                            if frame_metrics is None:
                                frame_metrics = {}
                            frame_metrics["face_eye_sharpness"] = float(face_eye_sharp)
                    if eye_gate:
                        try:
                            frame_eyes = eyes_mod.analyse_eyes(preview)
                        except Exception as e:
                            log.warning("eyes skip %s: %s", f.display_name, e)
                            frame_eyes = None
                    face_dets_payload = None
                    if strategy == "faces":
                        try:
                            dets = face_id_mod.extract_face_detections(preview)
                            face_dets_payload = face_id_mod.encode_detections_for_cache(dets)
                        except Exception as e:
                            log.warning("face_id skip %s: %s", f.display_name, e)
                    state.cache_frame_metadata(
                        sha, sharp, str(phash), ts,
                        face_count=face_count, face_sharpness=face_sharp,
                        metrics=frame_metrics, eyes=frame_eyes,
                        face_detections=face_dets_payload,
                    )
                    aesthetic_done += 1
                except Exception as e:
                    log.warning("aesthetic skip %s: %s", f.display_name, e)
                    aesthetic_skipped += 1
            thumb = encode_thumbnail(preview) if report_path else ""

        # Face-aware gate: only kicks in when --face-gate is set AND the frame
        # contains a face. The frame is rejected when the sharpest whole face is
        # below the per-face threshold OR — when an eye-region focus measure is
        # available — the subject's eyes aren't tack-sharp. Restricting to the
        # eye region catches the classic miss where focus landed on the cheek/ear
        # but the whole-face Laplacian still squeaks past.
        face_soft = face_count > 0 and face_sharp < face_sharp_threshold
        eyes_soft = face_eye_sharp is not None and face_eye_sharp < face_sharp_threshold
        if face_gate and a_score is not None and (face_soft or eyes_soft):
            face_gated += 1
            # Pretend the frame failed the global gate so it lands in REJECT.
            sharp = min(sharp, threshold - 0.01)
            a_score = None
            breakdown = None
            source = None
            emb = None
            phash = None
            if eyes_soft and not face_soft:
                log.info(
                    "REJECT (eyes soft, %.1f < %.1f) %s",
                    face_eye_sharp, face_sharp_threshold, f.display_name,
                )
            else:
                log.info(
                    "REJECT (face soft, %.1f < %.1f) %s",
                    face_sharp, face_sharp_threshold, f.display_name,
                )

        # Eye-aware gate: same shape as face-gate but on EAR via mediapipe.
        if (
            eye_gate
            and a_score is not None
            and frame_eyes is not None
            and frame_eyes.get("face_count", 0) > 0
            and frame_eyes.get("ear_min", 1.0) < eye_ear_threshold
        ):
            eye_gated += 1
            sharp = min(sharp, threshold - 0.01)
            a_score = None
            breakdown = None
            source = None
            emb = None
            phash = None
            log.info(
                "REJECT (eyes closed, EAR=%.2f < %.2f) %s",
                frame_eyes["ear_min"], eye_ear_threshold, f.display_name,
            )

        row = Row(
            frame=f,
            sharpness=sharp,
            aesthetic=a_score,
            aesthetic_breakdown=breakdown,
            aesthetic_source=source,
            thumb_b64=("" if cache_hit else thumb),
            metrics=frame_metrics,
            eyes=frame_eyes,
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
    clusters = dedup.cluster_bursts(
        [ci for _, ci, _ in dedup_inputs],
        hamming_dist=dedup_hamming,
        time_window=dedup_time_window,
        enabled=dedup_enabled,
    )
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
    # If a KMeans model is on disk, route via cluster id; otherwise fall back
    # to the prompt-based router so behaviour is preserved pre-fit.
    scene_clusters = scene_kmeans.load()
    if scene_clusters is not None:
        log.info(
            "scene routing: KMeans(k=%d) cluster ids → cluster_NN preset slots",
            scene_clusters.k,
        )
    scene_done = 0
    scene_cluster_ids: dict[str, int] = {}
    for row, _ci, emb in dedup_inputs:
        if not row.cluster_best:
            continue
        try:
            if scene_clusters is not None:
                info = scene_clusters.classify_embedding(emb)
                row.scene_preset = info.preset
                row.scene_score = float(np.dot(emb, info.centroid))
                row.scene_fell_back = False
                scene_cluster_ids[row.frame.display_name] = info.cluster_id
            else:
                match = scenes.classify(emb)
                row.scene_preset = match.preset
                row.scene_score = match.score
                row.scene_fell_back = match.fell_back
            scene_done += 1
        except Exception as e:
            log.warning("scene classify skip %s: %s", row.frame.display_name, e)
            continue

    elapsed = time.monotonic() - t0
    sharps = [r.sharpness for r in rows]
    aesthetic_vals = [r.aesthetic for r in rows if r.aesthetic is not None]
    sharp_kept = sum(1 for r in rows if r.sharpness >= threshold)
    final_kept = sum(1 for r in rows if r.sharpness >= threshold and r.cluster_best)

    log.info(
        "summary: %d final kept (= %d sharpness-pass − %d burst dupes), %d rejected of "
        "%d frames in %.1fs (aesthetic done=%d skipped=%d, %d bursts found, "
        "%d scenes classified, %d cache hits, %d face-gated, %d eye-gated, %d metrics computed)",
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
        eye_gated,
        metrics_done,
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

    if write_xmp and rows:
        from banger import xmp as xmp_mod

        n = xmp_mod.write_for_rows(rows, cluster_ids=scene_cluster_ids or None)
        log.info("wrote %d XMP sidecars (rating from rank, label from scene)", n)

    if output_dir is not None:
        # Build pool of (Row, score, embedding) for cluster-best survivors.
        embs_by_row_id = {id(row): emb for row, _ci, emb in dedup_inputs}
        candidates: list[tuple[Row, float, np.ndarray]] = []
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
            if strategy == "topk":
                log.info("selecting top %d (plain top-K by aesthetic)", top_n)
                chosen = select.select_top_k(candidates, n=top_n)
            elif strategy == "mmr":
                log.info(
                    "selecting top %d (MMR, lambda=%.2f) from %d candidates",
                    top_n, diversity, len(candidates),
                )
                chosen = select.select_diverse_top_n(
                    candidates, n=top_n, diversity_lambda=diversity
                )
            elif strategy == "faces":
                face_embs_per_item: list[list[np.ndarray]] = []
                for r, _s, _e in candidates:
                    sha2 = state.sha256_of(r.frame.classify_path)
                    meta2 = state.load_frame_metadata(sha2) or {}
                    payload = meta2.get("face_detections") or meta2.get("face_embeddings")
                    face_embs_per_item.append(face_id_mod.decode_from_cache(payload))
                n_people = sum(1 for embs in face_embs_per_item if embs)
                log.info(
                    "selecting top %d (face-diverse) from %d candidates "
                    "(%d carry face embeddings)",
                    top_n, len(candidates), n_people,
                )
                chosen = select.select_faces_top_n(
                    candidates, face_embs_per_item=face_embs_per_item, n=top_n,
                )
            else:  # kmeans
                log.info(
                    "selecting top %d (k-means, one per cluster) from %d candidates",
                    top_n, len(candidates),
                )
                chosen = select.select_kmeans_top_n(candidates, n=top_n)
            top_rows = [row for row, _s, _e in chosen]
            _write_output(output_dir, top_rows)
    return 0


def _write_output(output_dir: Path, top_rows: list[Row]) -> None:
    """Copy the top-N originals (RAW + JPEG sidecars) into output_dir, flat.

    Copies are atomic (temp .part -> fsync -> os.replace, then re-hash) via
    fsutil.atomic_copy, so a kill/full-disk mid-copy never leaves a truncated
    file at the final name. We preflight free space with fsutil.has_free_space
    and surface any per-file copy failures in the summary instead of letting
    them pass silently.
    """
    import json as _json

    from banger import fsutil

    log = logging.getLogger("banger")
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("copying top %d originals to %s", len(top_rows), output_dir)

    # Preflight: sum the bytes we're about to copy (skip ones already present)
    # and fail fast with a clear error if the destination volume can't hold them.
    needed_bytes = 0
    for r in top_rows:
        for src in (p for p in (r.frame.raw, r.frame.jpeg) if p is not None):
            dst = output_dir / src.name
            if dst.exists():
                continue
            try:
                needed_bytes += src.stat().st_size
            except OSError:
                pass
    if not fsutil.has_free_space(output_dir, needed_bytes):
        log.error(
            "not enough free space to export ~%.1f MiB to %s; aborting copy",
            needed_bytes / (1024 * 1024),
            output_dir,
        )
        return

    manifest_entries: list[dict] = []
    copied = 0
    failed: list[str] = []
    for rank, r in enumerate(top_rows, start=1):
        sources = [p for p in (r.frame.raw, r.frame.jpeg) if p is not None]
        copied_names: list[str] = []
        for src in sources:
            dst = output_dir / src.name
            if dst.exists():
                copied_names.append(src.name)
                continue
            # We only know a trusted sha for the frame's classify_path; for
            # the matching source pass it so atomic_copy can verify the copy.
            expected_sha = None
            if src == r.frame.classify_path:
                try:
                    expected_sha = state.sha256_of(src)
                except OSError:
                    expected_sha = None
            try:
                fsutil.atomic_copy(src, dst, expected_sha=expected_sha)
                copied += 1
                copied_names.append(src.name)
            except OSError as e:
                log.warning("copy %s -> %s failed: %s", src, dst, e)
                failed.append(src.name)
        manifest_entries.append({
            "rank": rank,
            "stem": r.frame.stem,
            "subdir": r.frame.subdir,
            "kind": r.frame.kind,
            "files": copied_names,
            "sharpness": round(r.sharpness, 1),
            "aesthetic": round(r.aesthetic, 3) if r.aesthetic is not None else None,
            "aesthetic_source": r.aesthetic_source,
        })

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(_json.dumps(manifest_entries, indent=2), encoding="utf-8")
    if failed:
        log.warning(
            "copied %d files to %s (%d FAILED: %s); manifest at %s",
            copied, output_dir, len(failed), ", ".join(failed), manifest_path,
        )
    else:
        log.info("copied %d files to %s; manifest at %s", copied, output_dir, manifest_path)


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


def _install_file_logging() -> None:
    """Tee log output to a rotating file under STATE_DIR/logs/banger.log.

    Best-effort: if the logs dir can't be created (read-only home, etc.) we
    just skip the file handler and keep console logging. Idempotent — re-runs
    in the same process won't stack duplicate handlers.
    """
    from logging.handlers import RotatingFileHandler

    root = logging.getLogger()
    log_dir = state.STATE_DIR / "logs"
    log_path = log_dir / "banger.log"
    if any(
        isinstance(h, RotatingFileHandler)
        and getattr(h, "baseFilename", None) == str(log_path)
        for h in root.handlers
    ):
        return
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        root.addHandler(handler)
    except OSError:
        logging.getLogger("banger").debug("file logging unavailable", exc_info=True)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _install_file_logging()
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(
            args.input_dir,
            args.recursive,
            args.report,
            output_dir=args.output,
            top_n=args.top_n,
            face_gate=args.face_gate,
            eye_gate=args.eye_gate,
            write_xmp=args.xmp,
            diversity=args.diversity,
            strategy=args.strategy,
        )
    if args.command == "explain":
        return cmd_explain(args.input_dir, args.recursive, args.stems)
    if args.command == "label":
        return cmd_label(args.input_dir, args.recursive, args.score, args.stems)
    if args.command == "train":
        return cmd_train()
    if args.command == "benchmark":
        if args.vs == "facet" and args.facet_path is None:
            logging.getLogger("banger").error(
                "--vs facet requires --facet-path <path to facet checkout>"
            )
            return 2
        from banger import benchmark as bench_mod
        return bench_mod.cmd_benchmark(
            input_dir=args.input_dir,
            vs_facet=(args.vs == "facet"),
            top_n=args.top_n,
            output=args.output,
            facet_path=args.facet_path,
            recursive=args.recursive,
        )
    if args.command == "scenes":
        if args.scenes_action == "fit":
            return 0 if scene_kmeans.fit(k=args.k) is not None else 2
        if args.scenes_action == "show":
            return scene_kmeans.print_summary()
    if args.command == "ui":
        return cmd_ui(args.input_dir, args.port)
    if args.command == "gui":
        from banger import gui as gui_mod
        gui_mod.serve(port=args.port, open_window=not args.headless)
        return 0
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
