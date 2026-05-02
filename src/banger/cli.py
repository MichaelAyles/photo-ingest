import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

from banger import aesthetic
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
        if sharp >= threshold:
            try:
                a_score = aesthetic.score_from_preview(preview)
                aesthetic_done += 1
            except Exception as e:
                log.warning("aesthetic skip %s: %s", f.stem, e)
                aesthetic_skipped += 1

        thumb = encode_thumbnail(preview) if report_path else ""
        rows.append(Row(frame=f, sharpness=sharp, aesthetic=a_score, thumb_b64=thumb))

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


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args.input_dir, args.report)
    return 1


if __name__ == "__main__":
    sys.exit(main())
