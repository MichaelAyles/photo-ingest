# Build plan

Execution plan for the banger pipeline. Design rationale lives in `CLAUDE.md`. This file is the ordered list of things to actually do.

Working assumption: development happens on the Windows side of the XPS for editing convenience, but the pipeline only needs to run on Linux (gphoto2, udev, darktable-cli all want Linux). v0 should be runnable from a Linux shell against a folder of sample JPEGs without any camera attached, so the algorithmic stages can be built and tuned before touching hardware.

## v0 — one weekend, prove the treatment-selection idea works

Goal: a script that takes a folder of JPEGs, picks the top 10, applies a CLIP-matched darktable preset to each, drops them in a dated output folder. No camera, no daemon, no notifications.

### Step 1 — project skeleton
- `pyproject.toml` (or `requirements.txt` if simpler) pinning Python 3.11, torch + CUDA 12, open_clip, rawpy, imagehash, Pillow, numpy.
- `src/banger/` package with empty `__init__.py`.
- `src/banger/cli.py` entry point: `python -m banger run <input_dir>`.
- `presets/` folder, empty for now, README inside saying "drop .xmp files here named to match prompts".
- `tests/fixtures/` for a handful of sample JPEGs to run the pipeline against.

### Step 2 — sharpness gate
- Function `sharpness(path) -> float` using cv2 Laplacian variance on a 1024px-long-edge resize.
- Threshold lives in a config dict at the top of the module, default 100.
- Return all paths with their score so the caller decides what to bin.
- The unit of work for the rest of the pipeline is a `Frame` (stem, jpeg, raw — one or both present), not a raw path. Classification stages always read `frame.classify_path` (prefer JPEG, fall back to RAW preview); develop reads `frame.develop_path` (prefer RAW, fall back to JPEG). This kills duplicate scoring of RAW+JPEG pairs and folds most of step 15 into the loader.

### Step 3 — aesthetic scoring
- Original plan: wrap aesthetic-predictor-v2.5 (SigLIP-so400m) in a `score(path) -> float`.
- Pivoted: use CLIP ViT-B/32 with prompt-based scoring (mean sim to "good photo" prompts minus mean sim to "bad photo" prompts). Reason: SigLIP's 3.3 GB safetensors triggers a torch+Windows mmap-slice bug (`torch.UntypedStorage.__getitem__` access violation on files above ~2 GB). The plan explicitly permits the CLIP-prompt fallback.
- Output range is roughly [-2, 3] rather than [1, 10]; only the ranking matters for top-N selection.
- Module-level `lru_cache` loads CLIP once and pre-encodes prompts; per-frame scoring is one image forward + one matmul.
- Replace with a trained taste head in v1 (step 17). The trained head can run on Linux where the SigLIP path also works if we want it later.

### Step 4 — top-N selection
- Filter by sharpness threshold, sort surviving frames by aesthetic score, take top N (default 10, env var `BANGER_TOP_N`).
- If zero survive, log and exit 0.

### Step 5 — scene matching
- Load CLIP once, encode the seven prompts from `CLAUDE.md` § Stage 7 once.
- For each survivor, encode the image, take cosine sim against each prompt, return `(best_prompt, score)`.
- If best score < 0.25, return the safe default ("crisp daylight outdoor portrait").
- Map prompt → preset filename via static dict in code.

### Step 6 — darktable develop
- Shell out to `darktable-cli input output --style preset_name`. Capture stderr, log on non-zero exit.
- Fallback path: if darktable-cli fails, copy the source JPEG to output and log a warning. PIL tone-curve approximation can wait.
- Parallelise with `concurrent.futures.ProcessPoolExecutor`, worker count = `os.cpu_count() - 1`.

### Step 7 — output and logging
- Output folder: `~/Pictures/bangers/YYYY-MM-DD/` (configurable via `BANGER_OUTPUT_DIR`).
- Stdout logging via stdlib `logging` at INFO. Per-frame: path, sharpness, aesthetic score, chosen preset.
- Print a summary line at the end: "N kept, M rejected, T seconds".

### Step 8 — hand-tune presets
- Open darktable on the actual machine, build one preset per prompt against a representative photo, export each as `.xmp` into `presets/`.
- This is the unskippable manual step. Without good presets the rest is theatre.

### Step 9 — manual smoke test
- Point the CLI at a folder of ~100 mixed JPEGs from a real shoot. Eyeball the output.
- Decide: does the prompt → preset pairing produce sensible-looking results? If no, re-tune prompts before any further engineering.

## v0.5 — make it run on plug-in

Once v0 is proven on a fixtures folder, wire it to the camera.

### Step 10 — gphoto2 ingest
- `banger ingest` subcommand: shell out to `gphoto2 --camera "Sony Alpha-A6600" --get-all-files --filename "%f"` into a staging dir.
- Detect PTP-vs-MTP mismatch by checking gphoto2's exit code and stderr; print a clear "set USB Connection → PTP in the camera menu" message.

### Step 11 — SQLite state
- `~/.local/share/banger-pipeline/state.db`, single table `processed(filename TEXT PRIMARY KEY, sha256 TEXT, processed_at INTEGER)`.
- Skip files already in the table at the start of ingest.
- Atomic move from staging to processed only after the row is committed.

### Step 12 — burst dedup
- pHash via `imagehash` on each preview. Cluster by Hamming distance ≤ 6 within a 3-second EXIF timestamp window.
- Keep the highest aesthetic score per cluster. Log cluster sizes so I can tune the thresholds.

### Step 13 — udev + systemd
- udev rule matching the a6600's USB vendor/product IDs, runs `systemctl --user start banger-pipeline.service`.
- systemd user unit invokes `python -m banger run-from-camera`. `Type=oneshot`, no restart.
- Document the install steps in the README.

### Step 14 — notify-send
- After the run finishes, emit a notification with kept/rejected counts.
- Click action opens the dated output folder. `gio open` is the simplest cross-DE way.

### Step 15 — RAW preview extraction
- Already done as part of step 2: `Frame.classify_path` falls back to the RAW for RAW-only frames, and `preview.load_preview` extracts the embedded JPEG via rawpy. Nothing extra to do unless a real shoot turns up an ARW with no embedded thumb (libraw raises `LibRawNoThumbnailError`, currently logged-and-skipped).

## v1 — make the cull actually mine

### Step 16 — thumbs-up/down log (PULLED FORWARD)
- Done early: generic CLIP-prompt scoring failed empirically on real photos (it ranked a clean shot of a dog swimming away above the user's actual favourite, because CLIP's negatives punished the favourite's intentional shallow depth-of-field as if it were accidental blur). The plan called for the trained head only at v1, but with the prompt scoring this misranked, we needed the labelling tool now.
- Implemented: `banger label <input_dir> up|down <stem>...` writes a SQLite row keyed by sha256 of the source file. CLIP embeddings cached during `banger run` at `~/.local/share/banger-pipeline/embeddings/{sha}.npy`.

### Step 17 — trained taste head (PULLED FORWARD)
- Done early: `banger train` reads the labels DB, looks up cached embeddings by sha, fits a sklearn LogisticRegression, saves to `~/.local/share/banger-pipeline/taste_head.joblib`. Leave-one-out CV accuracy printed for sanity. `cmd_run` auto-loads the head if it exists and uses it for scoring; otherwise falls back to prompts. Plan said 200 labels; the head can be trained from any number with both classes present, but is unreliable until at least 30-50.

### Step 18 — face-aware sharpness
- Run a face detector (mediapipe is fine) on survivors. If a face is present, override the global sharpness threshold with an "eyes-in-focus" check on the face crop.

### Step 19 — per-preset confidence thresholds
- Replace the single 0.25 cutoff with per-prompt thresholds tuned from logged matches.
- Anything below its prompt's threshold falls back to the safe default.

## v2 — speculative, not committed

- Local VLM pass on survivors for caption/crop suggestions.
- Field box (Orin Nano in the camper).
- Immich upload over Tailscale.

## Definition of done per phase

- **v0**: I can run `python -m banger run ./tests/fixtures/` and get a folder of developed JPEGs whose treatment looks defensible. Bad pairings are a v0 finding, not a v0 failure.
- **v0.5**: I plug the camera in, walk away, come back to a notification and a folder of JPEGs. Re-plugging the same card does not redo work.
- **v1**: After three real trips of labelling, the trained head beats the generic predictor on a held-out set of my own labels. Subjective check: top 10 contains my actual favourites at least 70% of the time.

## Things I am explicitly deferring

- Multi-camera support. a6600 vendor/product ID hard-coded in the udev rule until a second body exists.
- A GUI. Console logging only.
- Cloud anything. Output is files.
- Replacing Lightroom for serious edits. This is trip triage.
- Contact sheet PDF — open question in `CLAUDE.md`, decide after using v0 for one trip.
- Reject retention policy — same, decide after one trip.
