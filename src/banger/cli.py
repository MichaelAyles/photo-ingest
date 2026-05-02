import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

from banger.frames import Frame, discover_frames
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
        "scoring %d frames for sharpness (threshold=%.1f, report=%s)",
        len(frames),
        threshold,
        report_path or "off",
    )

    t0 = time.monotonic()
    scored: list[tuple[Frame, float]] = []
    rows: list[Row] = []
    for f in frames:
        try:
            preview = load_preview(f.classify_path)
        except Exception as e:
            log.warning("skip %s: %s", f.stem, e)
            continue
        score = sharpness_from_preview(preview)
        scored.append((f, score))
        if report_path is not None:
            rows.append(Row(frame=f, score=score, thumb_b64=encode_thumbnail(preview)))
    elapsed = time.monotonic() - t0

    scored.sort(key=lambda x: x[1], reverse=True)
    for f, s in scored:
        flag = "KEEP  " if s >= threshold else "REJECT"
        log.info("%s sharpness=%8.1f  %s  [%s]", flag, s, f.stem, f.kind)

    if scored:
        scores = [s for _, s in scored]
        kept = sum(1 for s in scores if s >= threshold)
        log.info(
            "stats: min=%.1f median=%.1f max=%.1f",
            min(scores),
            statistics.median(scores),
            max(scores),
        )
        log.info(
            "summary: %d kept, %d rejected of %d frames in %.1fs",
            kept,
            len(scored) - kept,
            len(scored),
            elapsed,
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
