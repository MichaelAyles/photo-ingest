import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

from banger import aesthetic, dedup, server, state, taste_head
from banger.aesthetic import NEGATIVE_PROMPTS, POSITIVE_PROMPTS
from banger.dedup import ClusterItem
from banger.frames import discover_frames
from banger.preview import load_preview
from banger.report import Row, encode_thumbnail, write_report
from banger.sharpness import CONFIG as SHARPNESS_CONFIG
from banger.sharpness import sharpness_from_preview


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

    ui = sub.add_parser(
        "ui",
        help="Start the local labelling UI (http://127.0.0.1:<port>).",
    )
    ui.add_argument("input_dir", type=Path)
    ui.add_argument("--port", type=int, default=8000)

    return parser


def cmd_run(input_dir: Path, recursive: bool, report_path: Path | None) -> int:
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
    dedup_inputs: list[tuple[Row, ClusterItem]] = []
    aesthetic_done = 0
    aesthetic_skipped = 0
    t0 = time.monotonic()
    for f in frames:
        try:
            preview = load_preview(f.classify_path)
        except Exception as e:
            log.warning("skip %s: %s", f.display_name, e)
            continue

        sharp = sharpness_from_preview(preview)
        a_score: float | None = None
        breakdown: dict[str, float] | None = None
        source: str | None = None
        phash = None
        if sharp >= threshold:
            try:
                emb = aesthetic.encode_image(preview)
                sha = state.sha256_of(f.classify_path)
                state.cache_embedding(sha, emb)
                _, breakdown = aesthetic.score_from_embedding(emb)
                if head is not None:
                    a_score = taste_head.predict_score(head, emb)
                    source = "head"
                else:
                    a_score, _ = aesthetic.score_from_embedding(emb)
                    source = "prompts"
                phash = dedup.phash_from_preview(preview)
                aesthetic_done += 1
            except Exception as e:
                log.warning("aesthetic skip %s: %s", f.display_name, e)
                aesthetic_skipped += 1

        thumb = encode_thumbnail(preview) if report_path else ""
        row = Row(
            frame=f,
            sharpness=sharp,
            aesthetic=a_score,
            aesthetic_breakdown=breakdown,
            aesthetic_source=source,
            thumb_b64=thumb,
        )
        rows.append(row)
        if phash is not None and a_score is not None:
            ts = dedup.best_timestamp(f.classify_path)
            dedup_inputs.append(
                (row, ClusterItem(key=f.display_name, phash=phash, timestamp=ts, score=a_score))
            )

        a_str = f"aesthetic={a_score:5.2f}" if a_score is not None else "aesthetic= ---"
        flag = "KEEP  " if sharp >= threshold else "REJECT"
        log.info("%s sharpness=%7.1f %s  %s  [%s]", flag, sharp, a_str, f.display_name, f.kind)

    # Burst dedup: cluster by pHash + timestamp, mark non-best siblings.
    rows_by_key = {ci.key: row for row, ci in dedup_inputs}
    clusters = dedup.cluster_bursts([ci for _, ci in dedup_inputs])
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

    elapsed = time.monotonic() - t0
    sharps = [r.sharpness for r in rows]
    aesthetic_vals = [r.aesthetic for r in rows if r.aesthetic is not None]
    sharp_kept = sum(1 for r in rows if r.sharpness >= threshold)
    final_kept = sum(1 for r in rows if r.sharpness >= threshold and r.cluster_best)

    log.info(
        "summary: %d final kept (= %d sharpness-pass − %d burst dupes), %d rejected of "
        "%d frames in %.1fs (aesthetic done=%d skipped=%d, %d bursts found)",
        final_kept,
        sharp_kept,
        suppressed_count,
        len(rows) - sharp_kept,
        len(rows),
        elapsed,
        aesthetic_done,
        aesthetic_skipped,
        len(bursts),
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
    return 0


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


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args.input_dir, args.recursive, args.report)
    if args.command == "explain":
        return cmd_explain(args.input_dir, args.recursive, args.stems)
    if args.command == "label":
        return cmd_label(args.input_dir, args.recursive, args.score, args.stems)
    if args.command == "train":
        return cmd_train()
    if args.command == "ui":
        return cmd_ui(args.input_dir, args.port)
    return 1


if __name__ == "__main__":
    sys.exit(main())
