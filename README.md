# banger

Local photo culler. Point it at a folder, get back the top picks with stars, tags, scene info, EXIF, and face names. No cloud, no upload, no phone in the loop. Built around a Sony a6600 + Linux workflow but works fine on any folder of JPEGs.

## What it does

For a folder of photos, banger:

1. **Filters** soft-focus frames via Laplacian sharpness, with optional face-sharpness and closed-eye gates.
2. **Scores** each surviving frame with CLIP plus a personal taste head trained on your -5..+5 labels.
3. **Dedups bursts** via pHash + EXIF timestamps so 9 near-identical shots become 1.
4. **Routes** each frame to a scene cluster (KMeans on CLIP embeddings) for treatment selection.
5. **Picks** a top-N with one of four strategies: pure score, MMR diversity, KMeans portfolio, or face-identity diversity (the Aftershoot trick).
6. **Surfaces** picks in a native desktop GUI with click-for-detail panels showing tags, scores, EXIF, and face names.

## Quickstart

```sh
python -m venv .venv
.venv\Scripts\activate                # Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or cpu wheels
pip install -e ".[dev,gui]"
pip install "transformers<5"          # transformers 5.x has a Windows-segfaulting weight loader

python -m banger gui                  # opens the native desktop window
```

First launch auto-loads CLIP, fits scene clusters on any cached embeddings, and installs mediapipe in the background if missing. A splash screen shows progress.

## GUI

Single window, four screens:

- **Welcome** — drag-drop a folder or click to browse. Pick strategy (kmeans / faces / mmr / topk), top-N count, and optional gates (face-sharpness, closed-eyes, XMP sidecar output).
- **Running** — live progress bar, current stage, log tail.
- **Results** — grid of top picks with rank badges, star ratings (top 10% = 5★), scene tag, score. Click any photo for detail.
- **Detail overlay** — hero image plus a stats panel:
  - **Tags**: 5-10 CLIP zero-shot chips ("hiking", "couple", "wide-angle landscape").
  - **Why it was picked**: aesthetic score, sharpness, scene preset.
  - **Quality dims**: bars for exposure, contrast, colour harmony, composition, leading lines. Flags for clipped shadows/highlights, silhouette, monochrome.
  - **Faces**: thumbnails of every detected face with click-to-name. Names persist across runs via a centroid DB so the same person gets auto-tagged on future runs.
  - **EXIF**: camera, lens, aperture, shutter, ISO, focal length (+ 35mm equiv), date, metering, flash, dimensions.
- **Settings** — pipeline state badges, face library (rename, forget, sample count).

## CLI

The GUI uses the same pipeline, but the CLI exposes more knobs:

```sh
python -m banger run ./photos -r --report reports/today.html
python -m banger run ./photos -r --output ~/Pictures/bangers/2026-05-13 --top-n 10
python -m banger run ./photos -r --strategy faces --face-gate --eye-gate --xmp

python -m banger label ./photos +5 DSC00055
python -m banger train                              # fits Ridge on labelled embeddings
python -m banger scenes fit -k 5                    # KMeans on cached CLIP embeddings
python -m banger scenes show                        # cluster summaries with nearest prompts

python -m banger explain ./photos DSC00055 DSC00073 # side-by-side prompt cosines
python -m banger benchmark ./photos --vs facet      # writes benchmarks/<date>.md
python -m banger version                            # deps + cache + state summary
```

`banger ui <folder>` opens just the labelling page (the GUI uses this internally for its Label view).

## Selection strategies

- **kmeans** (default): cluster candidates into N visual groups by CLIP embedding, take the best of each. Best for "portfolio variety from a single trip."
- **faces**: cluster by face identity via insightface ArcFace embeddings, take one good shot per person, fill remainder with kmeans-on-CLIP. Best when the same crew recurs across photos and pure aesthetic ranking keeps picking the well-lit folder.
- **mmr**: continuous diversity knob via `--diversity` (0 = pure visual diversity, 1 = pure score).
- **topk**: pure aesthetic ranking, no diversity. Best when you already know the folder is varied.

## State

Everything banger learns lives under `~/.local/share/banger-pipeline/` (yes, even on Windows):

- `labels.db` — your -5..+5 ratings, keyed by sha256 of the source file.
- `taste_head.joblib` — Ridge regressor trained on `labels.db`.
- `embeddings/<sha>.npy` — CLIP image embeddings, the most expensive cache to rebuild.
- `metadata/<sha>.json` — per-frame sharpness, phash, EXIF timestamp, quality metrics, face detections, tags.
- `thumbs/<sha>.jpg`, `previews/<sha>.jpg` — UI assets.
- `scene_kmeans.joblib` + `scene_kmeans.json` — scene cluster model.
- `faces.db` — named-face centroid DB with thumbnails.

Warm runs (everything cached) are ~7x faster than cold. Clear specific caches with `banger cache clear --kind embeddings` or `--all`.

## Stack

Python 3.11+, PyTorch + CUDA, transformers (CLIP ViT-B/32), insightface (ArcFace), mediapipe (FaceMesh / EAR), rawpy, imagehash, scikit-learn, Flask, pywebview. Optional: darktable-cli, gphoto2 (Linux).

## Limitations / scope

- This is a triage tool, not a library manager or photo editor.
- darktable preset authoring per scene cluster is manual and needs the darktable GUI on Linux.
- gphoto2 camera ingest is sketched but deferred until the user is on the Linux side of the laptop.
- See `TODO.md` for the active open list.

## Status

138 tests, ~10 s. GUI is the primary entrypoint; the CLI is fully functional and exercises the same pipeline. Built and dogfooded on Windows; designed to run on Linux without code changes.
