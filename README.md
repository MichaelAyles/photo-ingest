# banger

A local photo library, culler, and editor. Open it, point it at your photo folders, and it indexes everything, tags it via CLIP, geotags it, finds the faces, scores each frame against your trained taste, and lets you edit and export. No cloud, no upload, no phone in the loop.

Started as a triage script for a Sony a6600. Now a full Lightroom-shaped workbench that runs entirely on your machine.

## What it does

For every photo in every folder you watch:

- **Indexes** filename, EXIF (camera, lens, aperture, shutter, ISO, focal length, date), GPS coords reverse-geocoded to city / region / country, sharpness, pHash.
- **Tags** with ~180 CLIP zero-shot labels across subjects, scenes, activities, lighting, aesthetic, weather. "hiking, couple, valley, smiling, drone shot" type results.
- **Faces**: insightface (ArcFace) embeddings, cross-frame clustering, persistent names. Name a face once, the system finds them everywhere.
- **Scores** with a personal taste head trained on your -5..+5 labels.
- **Edits**: numpy + OpenCV develop pipeline (exposure / contrast / highlights / shadows / whites / blacks / saturation / vibrance / temp / tint / rotation / crop), real-time slider preview, full-res JPEG export.
- **Exports gallery-ready bangers**: one click → top-N picked, smart-edited per frame with eight named looks (auto-tone / B&W moody / B&W classic / vivid / warm / cool / golden / cinematic), copies to `/raw`, renders to `/edits`, drops a manifest, opens the folder.

Everything is searchable, filterable, and lives on your disk.

## Quickstart

```sh
python -m venv .venv
.venv\Scripts\activate                # Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev,gui]"
pip install "transformers<5"

python -m banger gui                  # native desktop window
```

First launch auto-loads CLIP, fits scene clusters on any cached embeddings, backfills GPS / geocoding for any pre-indexed frames, and installs mediapipe in the background. A splash screen shows progress.

## The GUI

Six views, all native, no Chromium bundled:

### Library (home)

Sidebar: watched root folders with frame counts, subfolder tree, text search across tags / names / cameras / places, dropdown filters for camera / face / place / minimum star rating.

Grid: 4:3 thumbnails of every photo in the selected scope. Lazy-loaded. Click for the detail overlay, double-click straight to the editor.

Toolbar:
- **Import…** copies from any folder (SD card, camera mount point) into a destination, organised by date / camera / flat, with RAW+JPEG pairs kept together. Auto-adds destination to watched roots and scans.
- **Rescan** re-walks the selected root for new / changed / removed files.
- **Index all** runs the heavy pre-compute on every frame in every root: CLIP embedding, tags, insightface face detections, scene cluster assignment, taste head scoring. After this, culling and filtering are instant SQL/JSON reads.
- **Score this view** kicks the cull pipeline (sharpness gate, dedup, selection by strategy) for the current scope. Runs in the background; bottom strip shows progress.
- **Export bangers ↗** culls the current scope and exports the top N as a gallery of finished JPEGs. See the Export section below.

Empty state shows big "Open folder" and "Import from device" buttons.

### Detail overlay (click a photo)

Two-column. Hero image on the left, scrollable stats panel on the right showing:
- **Tags**: 5-10 CLIP zero-shot chips ("hiking", "couple", "wide-angle landscape")
- **Why it was picked**: aesthetic score + source, sharpness, scene preset, file kind
- **Quality dims**: bars for exposure, contrast, colour harmony, composition, leading lines. Flags for clipped shadows/highlights, silhouette, monochrome
- **Numbers**: noise σ, dynamic range in stops, mean luminance
- **Faces**: thumbnails with click-to-name. Names persist via a centroid DB; the same person across folders auto-matches once named
- **EXIF**: camera, lens, aperture, shutter, ISO, focal length (+ 35mm equiv), exposure comp, date, metering, flash, dimensions, **location** (reverse-geocoded city / region / country plus GPS coords)
- **Source**: path + sha prefix

Edit button jumps to the editor with this frame loaded.

### Editor

Full-screen. Hero canvas left, sliders right (Light / Colour / Geometry groups). Drag any slider for a real-time render (~180 ms server-side). Hold **Before/After** to peek the original. **Reset** zeroes everything. **Export…** writes a full-res JPEG (`<stem>_edit.jpg`) next to the source. **← Library** auto-saves edits to the develop sidecar and goes back.

### Export bangers (closing the loop)

The "I want to share these" workflow in one click. Available from both Results and Library toolbars.

Modal asks for:
- **Folder name** — empty defaults to a timestamp, type to append (e.g. `caminito-trip` → `2026-05-13_1718_caminito-trip/`)
- **How many bangers**
- **Variety** checkbox + look chips — when on, each frame gets the look that fits it best, with diversity bias so the gallery isn't ten identical auto-tones

Output structure:

```
~/Pictures/bangers/<YYYY-MM-DD_HHMM>[_label]/
  raw/                    originals copied verbatim
    DSC00012.JPG
    DSC00013.JPG
    ...
  edits/                  rendered JPEGs with the chosen look
    01_DSC00012_golden.jpg
    02_DSC00013_bw_moody.jpg
    ...
  manifest.json           what was applied per frame + look_counts summary
```

When done, the folder opens in OS Explorer. Filenames carry the look suffix so sorting groups treatments. Frames you've manually edited in the editor keep your saved params (look=null in the manifest).

The eight looks:

- **auto** — neutral bump (+contrast, +vibrance). Fallback when nothing else fits strongly.
- **bw_moody** — high-contrast B&W. Loves silhouettes + monochrome content.
- **bw_classic** — lighter B&W. Suits portraits and low-colour frames.
- **vivid** — colour-rich subjects (vibrance-only, not deep-fried).
- **warm** — gentle yellow nudge for sunny / warm cast frames.
- **cool** — water / winter / overcast / fog.
- **golden** — sunset / sunrise / golden hour. Lifts shadows.
- **cinematic** — portraits, low-light, raised blacks for the teal-and-orange feel.

Each look has a fit-score function that reads the frame's quality metrics (mean luminance, contrast, monochrome, silhouette, clipping) and CLIP tags. Greedy assignment with diversity bias picks one look per frame, prioritising best-fit but penalising looks already used heavily so the gallery comes out varied.

### Settings

Pipeline state badges (CLIP / scene clusters / mediapipe / insightface readiness, label count, cache counts).

**Face library** lists every named or auto-discovered person with a thumbnail and sample count. Click name to rename, × to forget. "Discover people" button DBSCANs every face embedding in the library and auto-creates `Person 1`, `Person 2`, ... entries for unnamed clusters — rename them and the system propagates the new name to every matching frame.

### Quick run (formerly Welcome)

The original one-shot path: pick a folder, configure top-N + strategy + gates, run. Mostly useful for benchmarks and the labelling-UI link. The Library tab is the primary entrypoint now.

### Running

Detailed live progress for an active scoring job. Optional view — accessible via the bottom strip's "expand ↗" button. Most users just watch the bottom strip instead.

## Background work

A thin progress strip at the bottom of the screen shows tagger + scoring-job progress when either is running. Click "expand ↗" to jump to the Running view for the active job. After a library scan, banger auto-kicks the tagger so new folders get embeddings + tags without a click. The "Index all" button does the heavier full pass.

## Camera support

Single Sony body was the original target. Supported formats now (via libraw / rawpy):

- **Sony** (ARW)
- **Canon** (CR2, CR3)
- **Nikon** (NEF, NRW)
- **Fuji** (RAF)
- **Pentax** (PEF)
- **Olympus** (ORF)
- **Panasonic** (RW2)
- **Apple / Adobe / Pixel** (DNG)
- **Samsung** (SRW), **Sigma** (X3F), **Hasselblad** (3FR), **Phase One** (IIQ), **Leica** (RWL), **GoPro** (GPR)
- **HEIC / HEIF** (via optional `pip install pillow-heif`)
- Standard JPEG

Phone JPEGs (iPhone, Pixel, Android) work without RAW conversion.

## CLI

The GUI uses the same pipeline. The CLI exposes everything with more knobs:

```sh
python -m banger run ./photos -r --strategy faces --face-gate --eye-gate --xmp
python -m banger label ./photos +5 DSC00055
python -m banger train                              # Ridge head on labelled embeddings
python -m banger scenes fit -k 5                    # KMeans on cached embeddings
python -m banger scenes show                        # cluster summaries
python -m banger explain ./photos DSC00055 DSC00073 # side-by-side prompt cosines
python -m banger benchmark ./photos --vs facet      # writes benchmarks/<date>.md
python -m banger version                            # state + deps summary
python -m banger ui <folder>                        # labelling UI only (used by Label view internally)
```

## Selection strategies (Score this view + CLI --strategy)

- **kmeans** (default): visual clusters via KMeans on CLIP embeddings, one best per cluster. Portfolio variety.
- **faces**: cluster by face identity, one good shot per person, fill with kmeans for face-less frames. The Aftershoot trick. Best for groups / events.
- **mmr**: continuous diversity knob via `--diversity` (0 = max diversity, 1 = pure score).
- **topk**: pure aesthetic ranking, no diversity.

## State

Everything banger learns lives under `~/.local/share/banger-pipeline/` (yes, on Windows too):

- `library.db` — SQLite frame index. Roots + frames tables with EXIF / camera / GPS / place columns.
- `labels.db` — your -5..+5 ratings, keyed by sha256.
- `faces.db` — named-face centroid DB with thumbnails.
- `taste_head.joblib` — Ridge regressor.
- `scene_kmeans.joblib` + `scene_kmeans.json` — scene cluster model.
- `embeddings/<sha>.npy` — CLIP image embeddings (the expensive cache).
- `metadata/<sha>.json` — sharpness, phash, EXIF timestamp, quality metrics, face detections, tags, develop params, scene cluster, taste score.
- `thumbs/<sha>.jpg`, `previews/<sha>.jpg` — UI assets.

Warm runs (everything cached) are ~7× faster than cold. `banger cache clear --kind embeddings` (or `--all`) to reset.

## Stack

Python 3.11+, PyTorch + CUDA, transformers (CLIP ViT-B/32), insightface (ArcFace), mediapipe (FaceMesh / EAR), rawpy, imagehash, scikit-learn, reverse_geocoder, Flask, pywebview. Optional: pillow-heif, darktable-cli (not required since the built-in editor replaced it), gphoto2 (Linux).

138 tests, ~10 s.

## Limitations / scope

- The Linux camera daemon (gphoto2 + udev + systemd) is sketched but deferred until the user is on a Linux box.
- Video files (.mp4 etc.) aren't indexed — only photo formats. Pixel and DJI bodies mix videos with photo folders; for now they're invisible to the library.
- No cloud sync, by design. Files-only is the value proposition.
- Crop is wired in the develop pipeline but the editor has no draggable rectangle UI yet — set crop in the manifest by hand or via API for now.
- Multi-monitor edge cases in pywebview occasionally render slowly on first paint; reopen the window if you see it.

See `TODO.md` for the active open list.
