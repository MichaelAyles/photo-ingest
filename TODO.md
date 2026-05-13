# TODO

Forward-looking only. Done work lives in `git log`. Ranked by what would actually improve daily use.

## High value

### 1. Crop UI in the editor
Backend supports crop already (`editor.DevelopParams.crop` accepts `{x, y, w, h}` normalised), but the editor view has no draggable rectangle. Need:
- Crop tool toggle in the editor toolbar
- Drag rectangle on the hero canvas (corners + edges)
- Aspect-ratio lock dropdown (1:1, 4:5, 3:2, 16:9, freeform)
- Reset crop button

### 2. Video file support
Library scan currently ignores .mp4/.mov/etc. Pixel and DJI bodies sprinkle videos through photo folders so they should be at least indexed (filename, duration, thumbnail of first frame, EXIF if any). No editing for now; just "show up in the library so I can see what's there."
- Add .mp4/.mov/.avi/.mkv to `preview.SUPPORTED_SUFFIXES`
- `load_preview` for video = ffmpeg first-frame extract
- Library cell shows a play badge

### 3. Tests for library / editor / tagger
138 tests, but library.py, editor.py, tagger.py, gui.py have ~0 coverage. The geocoding regression that took two debug cycles to spot would have been caught by a 5-line test. Priority targets:
- library.scan_root idempotency + GPS column write
- editor.apply_develop math for each adjustment (delta vs identity)
- tagger.\_tag\_one full vs tags mode covers the right keys
- face_names.discover_clusters round-trip

### 4. Multi-cluster face merging
Face library has rename + forget but no "these two clusters are actually the same person, merge them." Common after discover_clusters produces e.g. Person 3 and Person 7 that the user knows are the same face seen from different angles. Need:
- Multi-select on the face library cards
- Merge button that averages centroids + transfers all face_detections refs
- Optionally: a "review pairs" surface that shows likely duplicates ranked by inter-centroid cosine sim

## Medium value

### 5. Smarter selection: stratified-by-folder + stratified-by-face
`faces` strategy still suffers when the same crew appears in well-lit folders more than mountain folders. Stratified picking ("at most K per subdir AND at most M per person") would force the crew shots from mountains into the top picks. Cheap to wire on top of existing strategies.

### 6. Tag vocab expansion + per-domain vocabs
180 tags covers general subjects fine but lacks domain specifics: climbing gear, dog breeds, food types, art styles, music venues. The CLIP cosines top out around 0.27 so even good matches sit close to the threshold. Either expand the vocab or split into per-domain banks the user selects from.

### 7. Maps view
Library has GPS + reverse-geocoded place names. A clickable map view (Leaflet + OpenStreetMap tiles, offline-cached) showing every frame's location as a pin would be ~2 days of work and instantly useful for "show me everything from that one trip."

### 8. Linux camera daemon
gphoto2 ingest, udev rule, systemd user unit. Blocked on dual-boot, sketched in v0.5 of the original plan. Real darktable preset authoring per scene cluster also lives here (needs the darktable GUI).

## Lower value / speculative

### 9. Packaging
PyInstaller bundle (~3 GB with torch + CLIP) or Tauri shell + PyInstaller sidecar (smaller installer, native window). Deferred until the algorithm + UI feel stable enough to be worth installing for someone else.

### 10. Better captioning
Tags are punchy and searchable but don't tell a story. BLIP-2 or Florence-2 generates real sentences for photos worth the extra ~1.5 GB of model weights on disk. Worth adding as opt-in if a user wants the detail panel to read like a description.

### 11. Multi-cluster face merging UI polish
Adjacent to (4): "who's in this batch" overview pane that shows every person across the current library scope at once, with click to filter.

## Known rough edges

- **WebView2 CSS quirks**: the library grid uses fixed 200 px row heights because both `aspect-ratio` and the `padding-bottom` % trick render as ~20 px collapsed cells inside a CSS grid with auto rows. The fix works but isn't responsive to window width.
- **OneDrive folders + git**: working in `OneDrive\Desktop\...` triggers sporadic `.git/index.lock` failures when OneDrive's sync agent locks files mid-commit. Run from a non-synced path if you can.
- **Insightface CPU only**: ~500 ms per face-bearing frame. Full Index of a 10k library is ~30 min. GPU path is wired but disabled to avoid contention with CLIP on a 4 GB card. Worth re-enabling on hosts with bigger GPUs.
- **mediapipe + Python 3.12 + Windows**: install can fail when the GUI process holds `cv2.pyd` open. Auto-install uses `--no-deps` to work around this; manual `pip install mediapipe` is the fallback.

## Explicit non-goals

- Cloud sync, web service, or phone app. Local-first is the point.
- Replacing Lightroom for high-end retouching (local adjustments, healing brush, lens corrections from Adobe's profile DB). The editor covers the bulk-edit use case, not the per-pixel one.
- An ML training UI. `banger train` from the CLI on labels.db is enough; nobody needs a GUI for that one-shot operation.
