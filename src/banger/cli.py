import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

from banger.frames import Frame, discover_frames
from banger.sharpness import CONFIG as SHARPNESS_CONFIG
from banger.sharpness import sharpness


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="banger")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the pipeline against a folder of images.")
    run.add_argument("input_dir", type=Path, help="Folder containing JPEGs and/or ARWs.")

    return parser


def cmd_run(input_dir: Path) -> int:
    log = logging.getLogger("banger")
    if not input_dir.is_dir():
        log.error("not a directory: %s", input_dir)
        return 2

    frames = discover_frames(input_dir)
    if not frames:
        log.warning("no supported images in %s", input_dir)
        return 0

    threshold = SHARPNESS_CONFIG["threshold"]
    log.info("scoring %d frames for sharpness (threshold=%.1f)", len(frames), threshold)

    t0 = time.monotonic()
    scored: list[tuple[Frame, float]] = []
    for f in frames:
        try:
            scored.append((f, sharpness(f.classify_path)))
        except Exception as e:
            log.warning("skip %s: %s", f.stem, e)
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
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args.input_dir)
    return 1


if __name__ == "__main__":
    sys.exit(main())
