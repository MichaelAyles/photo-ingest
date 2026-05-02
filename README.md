# banger pipeline

Plug a Sony a6600 into a Linux laptop, walk away, come back to ten finished JPEGs.

A local image triage and develop daemon. Pulls files via gphoto2, culls with a sharpness gate plus CLIP-based aesthetic scoring, picks a darktable preset per frame by matching the scene against a small set of treatment prompts, and writes developed JPEGs into a dated folder. No phone, no cloud, no GUI.

## Status

Skeleton only. CLI entry point exists and parses args; no pipeline stages wired up yet. See `plan.md` for the build order and `CLAUDE.md` for design rationale and non-goals.

## Install (development)

```sh
python -m venv .venv
. .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# torch needs the CUDA wheel index; pick the one matching your CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# aesthetic-predictor-v2.5 is GitHub-only:
pip install git+https://github.com/discus0434/aesthetic-predictor-v2-5
```

## Usage

```sh
python -m banger run ./tests/fixtures/
```

Currently a stub — logs the input path and exits.

## Why this exists

The Sony Creators' App does not support the a6600. Lightroom Mobile locks originals into Adobe's cloud. Google Photos refuses to back up from USB OTG. The gap is "JPEGs straight to my own destination, with some processing applied, no phone in the loop." This fills it for one user (me) on one camera (a6600) on one machine (a Linux-booted XPS 15).

## Stack

Python 3.11+, gphoto2, rawpy, PyTorch + CUDA, open_clip, aesthetic-predictor-v2.5, imagehash, darktable-cli, SQLite, systemd + udev. Single Python process, no servers.

## Layout

- `CLAUDE.md` — design doc, scope, non-goals, risks.
- `plan.md` — execution plan, ordered.
- `presets/` — hand-tuned darktable `.xmp` sidecars, one per treatment prompt. Empty until step 8.
- `src/banger/` — pipeline code (forthcoming).
- `tests/fixtures/` — sample JPEGs for running the pipeline without a camera.
