# banger pipeline

Plug a Sony a6600 into a Linux laptop, walk away, come back to ten finished JPEGs.

A local image triage and develop daemon. Pulls files via gphoto2, culls with a sharpness gate plus CLIP-based aesthetic scoring, dedups bursts via pHash, picks a darktable preset per frame by matching the scene against a small set of treatment prompts, and writes developed JPEGs into a dated folder. No phone, no cloud, no GUI required to use it (a labelling UI is bundled when you want to teach the model your taste).

## Status

v0 pipeline complete end-to-end on the algorithm side: sharpness, CLIP aesthetic (prompts or trained head), burst dedup, scene/preset selection, top-N output, manifest. 62 unit tests across the suite. Camera ingest (gphoto2) and the udev/systemd auto-trigger remain Linux-only and aren't wired up yet — run manually for now.

See `plan.md` for what's done vs deferred and `CLAUDE.md` for design rationale and non-goals.

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

darktable-cli is optional. If absent, the pipeline writes camera JPEGs verbatim and notes the would-have-been preset in the output manifest. Install darktable separately if you want it to actually develop.

## Usage

### Score a folder and write the diagnostic HTML report

```sh
python -m banger run ./test_photos -r --report reports/today.html
```

`-r` walks subdirectories. The HTML is one self-contained file with thumbnails and per-frame metadata; safe to email or move.

### Compare two specific frames CLIP-by-CLIP

```sh
python -m banger explain ./test_photos DSC00055 DSC00073
```

Prints a side-by-side cosine similarity table for every aesthetic prompt and the final score. Useful for figuring out *why* the model ranked one over the other.

### Teach the model your taste

```sh
python -m banger ui ./test_photos
# open http://127.0.0.1:8000 — score frames -5 to +5 with number keys
python -m banger train                                                      # fits a Ridge regressor
python -m banger run ./test_photos -r --report reports/with_head.html       # auto-uses the head
```

The UI is iOS-photo-roll style: hero image fills the viewport, score buttons under it, filmstrip of thumbnails along the bottom, all keyboard-driven. Numbers `1`-`5` map to scores -5..-1, `6`-`9`+`0` map to +1..+5; arrows navigate; space skips; backspace clears. Live histogram in the header shows your label distribution so you can spot bias as you work. Aim for 100+ labels spread across the full range; the head can train from any number with both classes but is unreliable below ~30-50.

### Final v0 output: top N developed frames in a folder

```sh
python -m banger run ./test_photos -r --output ~/Pictures/bangers/2026-05-02 --top-n 10
```

Writes top-N (default 10, env `BANGER_TOP_N`) developed JPEGs into the directory along with a `manifest.json` recording rank, source path, scores, picked preset, and which develop method was used. Filenames are `NN_subdir__stem.jpg` so the directory listing IS the ranked output.

`--output` defaults to `$BANGER_OUTPUT_DIR`; if the env value ends in `/`, `\` or `bangers`, a `YYYY-MM-DD/` subdir is appended so the canonical setup `BANGER_OUTPUT_DIR=~/Pictures/bangers` produces a fresh dated folder per run.

### Face-aware sharpness gate (opt-in)

```sh
python -m banger run ./test_photos -r --face-gate
```

Adds a second sharpness check: for any frame where a face is detected, the *sharpest face crop* must also pass `face.FACE_SHARPNESS_THRESHOLD` (default 50). Catches the classic missed-focus portrait that the global Laplacian gate lets through because the background is sharp.

### Re-run speed

The first `banger run` over a folder loads previews, computes Laplacian variance, embeds via CLIP, computes pHash + EXIF timestamp + (optionally) face data, and caches everything keyed by sha256. Subsequent runs over the same content skip preview decoding entirely and run in ~1/7th the wall-clock time. Delete `~/.local/share/banger-pipeline/metadata/` to force re-compute.

### CLI alternative for labelling

If you'd rather label from the terminal:

```sh
python -m banger label ./test_photos +5 DSC00055
python -m banger label ./test_photos -3 DSC00073
```

State (cached CLIP embeddings, thumbnails, hero previews, label DB, taste head) lives at `~/.local/share/banger-pipeline/`. Delete that dir to start fresh.

## Why this exists

The Sony Creators' App does not support the a6600. Lightroom Mobile locks originals into Adobe's cloud. Google Photos refuses to back up from USB OTG. The gap is "JPEGs straight to my own destination, with some processing applied, no phone in the loop." This fills it for one user (me) on one camera (a6600) on one machine (a Linux-booted XPS 15).

## Stack

Python 3.11+, gphoto2 (Linux), rawpy, PyTorch + CUDA, transformers (CLIP ViT-B/32), imagehash, scikit-learn, darktable-cli (optional), SQLite, systemd + udev (Linux). Single Python process. The labelling UI is Flask + a single HTML page.

## Layout

- `CLAUDE.md` — design doc, scope, non-goals, risks.
- `plan.md` — execution plan, ordered. ✓ marks done, ✗ deferred.
- `presets/` — hand-tuned darktable `.xmp` sidecars, one per treatment prompt. Build these in darktable and drop them in here.
- `src/banger/` — pipeline modules: `preview`, `frames`, `sharpness`, `aesthetic`, `dedup`, `scenes`, `face`, `develop`, `taste_head`, `state`, `report`, `server`, `cli`.
- `tests/` — pytest suite (89 tests, ~6 s).

## Running tests

```sh
.venv/bin/pytest               # on Windows: .venv\Scripts\pytest.exe
```
