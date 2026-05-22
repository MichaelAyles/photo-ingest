# banger — what it is

A local, all-in-one photo culler, library, and gallery-export tool. You point it at the folders you already keep your photos in, it understands what's in them, and it gives you the keepers ready to share. Nothing leaves your machine.

It started life as a triage script for one Sony a6600 — "plug the camera in, walk away, come back to ten finished JPEGs." Over a couple of weekend sessions it grew into something the size and shape of Lightroom, minus the cloud lock-in and the per-month subscription.

## The mental model

Three stages, each adds intelligence over the same set of files:

1. **Index** — make the library know everything it can about every file.
2. **Cull** — for any scope, pick the keepers from the noise.
3. **Ship** — render the keepers with sensible edits to a dated folder, ready for Instagram, a photo book, or sharing.

Everything else is supporting infrastructure for those three stages.

## What it knows about each photo

Once a folder is indexed, every frame has:

- **Filename, sha256, path, mtime, EXIF** (camera body, lens, aperture, shutter, ISO, focal length + 35 mm equivalent, exposure comp, date taken, metering mode, flash)
- **GPS** parsed from EXIF, then **reverse-geocoded offline** to city / region / country (so "swansea" or "keswick" works as a search term)
- **CLIP embedding** (ViT-B/32), used as the universal feature vector for tags, scenes, and similarity
- **5-10 tags** from a curated 180-entry vocabulary across subjects, scenes, activities, lighting, aesthetic, weather. "hiking, couple, valley, smiling, drone shot" type results.
- **Quality metrics**: sharpness via Laplacian variance, exposure with shadow/highlight clipping + silhouette detection, contrast, colour harmony entropy, composition score via rule-of-thirds detection, leading lines via Hough, noise via Immerkaer, dynamic range in stops
- **Faces**: insightface ArcFace embeddings + bounding boxes. Cross-frame DBSCAN clustering. Persistent name DB — name Beth once, every Beth photo in every folder gets tagged Beth forever.
- **Scene cluster** (KMeans on CLIP embeddings — your library's actual visual categories rather than guessed prompts)
- **Taste score** from a Ridge regressor trained on your -5..+5 hand-labels

All of it lives in `~/.local/share/banger-pipeline/` — SQLite indices plus a per-frame JSON sidecar. Indexing 1000 frames takes ~10 minutes the first time; after that everything is a database read.

## How the user actually drives it

**Library tab is home.** Left sidebar: watched folders (with frame counts), subfolder tree, text search (matches across tags, names, cameras, places), filter dropdowns for camera, face, place, minimum star rating. Right pane: a grid of 4:3 thumbnails. Toolbar: Import / Rescan / Index all / Score this view / Export bangers.

**Click a photo → detail overlay.** Hero image left, stats panel right showing tags, why-it-was-picked numbers, quality dim bars, exposure flags, faces (click to name), full EXIF including reverse-geocoded location.

**Edit a photo → full-screen editor** with sliders for exposure, contrast, highlights, shadows, whites, blacks, saturation, vibrance, temperature, tint, rotation. Drag any slider, the hero image re-renders in ~180 ms. Hold a button for before/after. Export writes a full-res JPEG next to the source.

**Settings → Face library** lists every named or auto-discovered person with a thumbnail and sample count. Rename, merge, forget.

**Bottom of the screen** has a subtle progress strip that shows the background tagger and any active scoring job. Click "expand ↗" to see the live log for the active job.

## Selection: four strategies for "give me the keepers"

- **kmeans** (default) — visual diversity. Cluster candidates by CLIP embedding into N visual groups, take the best of each. Portfolio variety.
- **faces** — Aftershoot's killer trick. Cluster by face identity, give you one good shot per person across the batch. Fixes "well-lit folder always wins."
- **mmr** — continuous diversity dial.
- **topk** — pure aesthetic ranking, no diversity. For when you already know the batch is varied.

Every strategy reads pre-indexed data and runs in seconds.

## The export workflow

This is the closer. One button: **Export bangers ↗**. Modal asks how many, a folder name (defaults to a timestamp), and which looks to use.

You get an output folder like:

```
~/Pictures/bangers/2026-05-13_1718_caminito-trip/
  raw/           originals copied verbatim
  edits/         rendered JPEGs with the look that fits each frame
                 (01_DSC00012_golden.jpg, 02_DSC00013_bw_moody.jpg, ...)
  manifest.json  what was applied per frame + look_counts summary
```

Eight named looks, each is a fixed DevelopParams plus a fit-score function reading the frame's metrics and CLIP tags:

- **auto** — neutral bump, fallback
- **bw_moody** — high-contrast B&W, loves silhouettes
- **bw_classic** — lighter B&W, suits portraits and low-colour
- **vivid** — colour-rich subjects
- **warm** — gentle yellow nudge
- **cool** — water, winter, overcast
- **golden** — sunset / sunrise, lifts shadows
- **cinematic** — portraits, low-light, teal-and-orange feel

A greedy diversity-biased assignment picks one look per frame. Each frame gets its best fit, but every subsequent assignment of the same look incurs a penalty, so the gallery comes out varied rather than ten identical auto-tones. Filenames carry the look suffix so sorting groups treatments. Frames you've manually edited keep your saved params.

The folder opens in OS Explorer when done.

## Multi-camera by design

Library scan supports the format library that libraw covers: Sony ARW, Canon CR2/CR3, Nikon NEF/NRW, Fuji RAF, Pentax PEF, Olympus ORF, Panasonic RW2, DNG (Apple, Adobe, Pixel), plus Samsung SRW, Sigma X3F, Hasselblad 3FR, Phase One IIQ, Leica RWL, GoPro GPR. HEIC/HEIF via optional pillow-heif. Standard JPEG everywhere.

Phone JPEGs (iPhone, Pixel, Android) work without RAW conversion. The same `test_photos` folder I dogfood with mixes Sony a6600 ARW shots with six different Pixel bodies and they all index, tag, and cull side-by-side.

## What it deliberately doesn't do

- **No cloud sync, no web service, no phone app.** Files-only is the value proposition. Every byte stays on the machine that has the files.
- **Not trying to replace Lightroom for high-end retouching.** No local-adjustment masks, healing brush, or Adobe's lens-correction DB. The editor is for "give every photo a sensible global look in bulk."
- **No video.** Library currently skips .mp4 etc. Pixel and DJI mix videos with photos so it should index them eventually, but that's a TODO not a v0 goal.
- **Not multi-tenant.** Single-user app running on a single laptop.

## The stack

Python 3.11+, PyTorch + CUDA, transformers (CLIP ViT-B/32), insightface (ArcFace), mediapipe (FaceMesh / EAR), rawpy / libraw, imagehash, scikit-learn, reverse_geocoder, Flask, pywebview. Native window via WebView2 on Windows, WKWebView on Mac. No bundled Chromium. 138 unit tests, ~10 s.

## Why this exists

Aftershoot is $15-48/month and cloud-tied. Lightroom Mobile locks originals into Adobe's cloud. The Sony Creators app doesn't support the a6600. Google Photos refuses to back up from USB OTG. The gap was "let me cull and finish a thousand photos locally without paying rent on my own files." That's banger.
