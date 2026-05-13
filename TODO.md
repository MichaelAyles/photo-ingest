# TODO

Forward-looking only. Done work lives in `git log`. Old planning docs are gone.

## Open, ranked by what would actually change my use of the tool

### 1. Better captions / tags
Tag vocabulary is hand-curated (~180 entries) and CLIP zero-shot tops out around 0.27 cosine on real photos. Two paths if the chips start to feel limiting:
- Expand vocabulary toward domain-specific terms (climbing gear, dog breeds, food types, etc).
- Swap CLIP zero-shot for a captioning model (BLIP-2 or Florence-2) and post-process into chips. Worth ~1.5 GB of model weights on disk; build pulls down on first use.

### 2. Folder-stratified selection
Even with `faces` strategy the well-lit folder still dominates picks that don't contain faces. Easy lever: enforce "at most K per subdir" on top of any selection strategy. Wire as a `--max-per-folder` knob on cmd_run + a checkbox in the GUI welcome screen.

### 3. Face library improvements
The library shows thumbnails + sample count + rename/forget. Missing:
- Showing other frames the centroid matches against, so the user can see whether a centroid has drifted.
- Merging two names (Alice + Alice-bad-spelling) into one centroid average.
- "Forget this sample" surgery: remove a single frame's contribution from a name without nuking the whole centroid.

### 4. Linux story (still blocked on dual-boot)
- `banger ingest` via gphoto2 (skeleton was sketched in v0.5 plan).
- udev rule + systemd user unit so plugging the a6600 in triggers a run.
- `notify-send` summary at end of run.
- Real darktable `.xmp` presets per scene cluster (manual authoring step, needs darktable GUI).

### 5. Packaging
GUI is `python -m banger gui`. For a non-technical user, the install dance (python, venv, pip install) is too much. Two paths considered earlier:
- PyInstaller bundle (~3 GB with torch + CLIP, but no install).
- Tauri shell + PyInstaller sidecar (smaller installer, native window).
Both deferred until the algorithm and UI feel stable enough to be worth installing.

## Known rough edges

- **mediapipe install on Python 3.12 + Windows** can fail in pip due to cv2.pyd file lock if the GUI is already running. Auto-setup uses `--no-deps` to work around it, but on a truly fresh env it may need a manual `pip install mediapipe` once.
- **Insightface CPU extraction** is ~1s per face-bearing frame. First `--strategy faces` run on a thousand-frame folder takes ~15 min. Subsequent runs hit the cache. GPU path is wired but disabled (insightface fights CLIP for the 4 GB GPU).
- **Caminito skew on the test set** when the taste head has heavy caminito labelling. Either label more non-caminito frames via the UI's `order: uncertain` toggle, or use `faces` strategy to break the visual-cluster tie. See item 2 above for a third lever.

## Things I've decided not to do

- A general-purpose photo browser. This tool is a culler / triage helper, not a library manager.
- Cloud upload / sync. Files-only is the value proposition.
- Replacing Lightroom for real edits. Develop step is "apply one preset" tops.
- Multi-camera support. A6600 vendor/product ID is hard-coded.
- Backwards-compat shims for the older `face_embeddings` cache format beyond what's already in `face_id.decode_from_cache`.
