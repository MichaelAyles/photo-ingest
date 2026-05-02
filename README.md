# banger pipeline

Plug a Sony a6600 into a Linux laptop, walk away, come back to ten finished JPEGs.

A local image triage and develop daemon. Pulls files via gphoto2, culls with a sharpness gate plus CLIP-based aesthetic scoring, picks a darktable preset per frame by matching the scene against a small set of treatment prompts, and writes developed JPEGs into a dated folder. No phone, no cloud, no GUI.

## Status

Not built yet. v0 is a one-weekend scope. See `plan.md` for the build order and `CLAUDE.md` for design rationale and non-goals.

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
