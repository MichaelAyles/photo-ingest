import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

from banger import aesthetic
from banger.aesthetic import NEGATIVE_PROMPTS, POSITIVE_PROMPTS
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
    explain.add_argument("stems", nargs="+", help="Frame stems, e.g. DSC00055 DSC00073.")

    return parser


def cmd_run(input_dir: Path, report_path: Path | None) -> int:
    log = logging.getLogger("banger")
    if not input_dir.is_dir():
        log.error("not a directory: %s", input_dir)
        return 2

    frames = discover_frames(input_dir)
    if not frames:
        log.warning("no supported images in %s", input_dir)
        return 0

    threshold = SHARPNESS_CONFIG["threshold"]
    log.info(
        "processing %d frames (sharpness threshold=%.1f, report=%s)",
        len(frames),
        threshold,
        report_path or "off",
    )

    rows: list[Row] = []
    aesthetic_done = 0
    aesthetic_skipped = 0
    t0 = time.monotonic()
    for f in frames:
        try:
            preview = load_preview(f.classify_path)
        except Exception as e:
            log.warning("skip %s: %s", f.stem, e)
            continue

        sharp = sharpness_from_preview(preview)
        a_score: float | None = None
        breakdown: dict[str, float] | None = None
        if sharp >= threshold:
            try:
                a_score, breakdown = aesthetic.score_from_preview(preview)
                aesthetic_done += 1
            except Exception as e:
                log.warning("aesthetic skip %s: %s", f.stem, e)
                aesthetic_skipped += 1

        thumb = encode_thumbnail(preview) if report_path else ""
        rows.append(
            Row(
                frame=f,
                sharpness=sharp,
                aesthetic=a_score,
                aesthetic_breakdown=breakdown,
                thumb_b64=thumb,
            )
        )

        a_str = f"aesthetic={a_score:5.2f}" if a_score is not None else "aesthetic= ---"
        flag = "KEEP  " if sharp >= threshold else "REJECT"
        log.info("%s sharpness=%7.1f %s  %s  [%s]", flag, sharp, a_str, f.stem, f.kind)

    elapsed = time.monotonic() - t0
    sharps = [r.sharpness for r in rows]
    aesthetic_vals = [r.aesthetic for r in rows if r.aesthetic is not None]
    kept = sum(1 for r in rows if r.sharpness >= threshold)

    log.info(
        "summary: %d kept, %d rejected of %d frames in %.1fs (aesthetic done=%d skipped=%d)",
        kept,
        len(rows) - kept,
        len(rows),
        elapsed,
        aesthetic_done,
        aesthetic_skipped,
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


def cmd_explain(input_dir: Path, stems: list[str]) -> int:
    log = logging.getLogger("banger")
    if not input_dir.is_dir():
        log.error("not a directory: %s", input_dir)
        return 2

    frames = {f.stem: f for f in discover_frames(input_dir)}
    missing = [s for s in stems if s not in frames]
    if missing:
        log.error("not found in %s: %s", input_dir, ", ".join(missing))
        return 2

    results: list[tuple[str, float, dict[str, float]]] = []
    for stem in stems:
        f = frames[stem]
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
    print(f"{'mean(positives)':<{label_w}}", end="")
    n_pos = len(POSITIVE_PROMPTS)
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


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args.input_dir, args.report)
    if args.command == "explain":
        return cmd_explain(args.input_dir, args.stems)
    return 1


if __name__ == "__main__":
    sys.exit(main())
