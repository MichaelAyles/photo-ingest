# banger pipeline

Plug a Sony a6600 into a Linux laptop, walk away, come back to ten finished JPEGs.

A local image triage and develop daemon. Pulls files via gphoto2, culls with a sharpness gate plus CLIP-based aesthetic scoring, picks a darktable preset per frame by matching the scene against a small set of treatment prompts, and writes developed JPEGs into a dated folder. No phone, no cloud, no GUI.

## Status

Skeleton only. CLI entry point exists and parses args; no pipeline stages wired up yet. See `plan.md` for the build order and `CLAUDE.md` for design rationale and non-goals.

## Install (development)

```sh
python -m venv .venv
. .venv/bin/activate          # on Windows: .venv\Scripts\activate

# torch needs the CUDA wheel index; install it FIRST so pip doesn't pull
# the CPU wheel as a transitive dep. Pick the index matching your CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Then everything else:
pip install -e ".[dev]"
pip install "transformers<5"  # transformers 5.x has a Windows-segfaulting parallel weight loader
```

## Usage

```sh
# Score a folder (sharpness + aesthetic) and write an HTML report:
python -m banger run ./test_photos --report reports/today.html

# See why CLIP ranked two specific frames the way it did:
python -m banger explain ./test_photos DSC00055 DSC00073

# Build a personal taste head from your own thumbs-up/down:
python -m banger run ./test_photos --report reports/today.html  # caches CLIP embeddings
python -m banger label ./test_photos up   DSC00055 DSC00088 DSC00091
python -m banger label ./test_photos down DSC00073 DSC00077
python -m banger train                                          # fits logistic regression
python -m banger run ./test_photos --report reports/today.html  # auto-uses the head
```

State (cached CLIP embeddings, label DB, taste head) lives at
`~/.local/share/banger-pipeline/`. Delete that dir to start fresh.

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
