# Build plan

Execution plan for the banger pipeline. Design rationale lives in `CLAUDE.md`. This file is the ordered list of things to actually do.

Working assumption: development happens on the Windows side of the XPS for editing convenience, but the pipeline only needs to run on Linux (gphoto2, udev, darktable-cli all want Linux). v0 should be runnable from a Linux shell against a folder of sample JPEGs without any camera attached, so the algorithmic stages can be built and tuned before touching hardware.

## v0 — one weekend, prove the treatment-selection idea works

Goal: a script that takes a folder of JPEGs, picks the top 10, applies a CLIP-matched darktable preset to each, drops them in a dated output folder. No camera, no daemon, no notifications.

### Step 1 — project skeleton ✓
- `pyproject.toml` (or `requirements.txt` if simpler) pinning Python 3.11, torch + CUDA 12, open_clip, rawpy, imagehash, Pillow, numpy.
- `src/banger/` package with empty `__init__.py`.
- `src/banger/cli.py` entry point: `python -m banger run <input_dir>`.
- `presets/` folder, empty for now, README inside saying "drop .xmp files here named to match prompts".
- `tests/fixtures/` for a handful of sample JPEGs to run the pipeline against.

### Step 2 — sharpness gate ✓
- Function `sharpness(path) -> float` using cv2 Laplacian variance on a 1024px-long-edge resize.
- Threshold lives in a config dict at the top of the module, default 100.
- Return all paths with their score so the caller decides what to bin.
- The unit of work for the rest of the pipeline is a `Frame` (stem, jpeg, raw — one or both present), not a raw path. Classification stages always read `frame.classify_path` (prefer JPEG, fall back to RAW preview); develop reads `frame.develop_path` (prefer RAW, fall back to JPEG). This kills duplicate scoring of RAW+JPEG pairs and folds most of step 15 into the loader.

### Step 3 — aesthetic scoring ✓
- Original plan: wrap aesthetic-predictor-v2.5 (SigLIP-so400m) in a `score(path) -> float`.
- Pivoted: use CLIP ViT-B/32 with prompt-based scoring (mean sim to "good photo" prompts minus mean sim to "bad photo" prompts). Reason: SigLIP's 3.3 GB safetensors triggers a torch+Windows mmap-slice bug (`torch.UntypedStorage.__getitem__` access violation on files above ~2 GB). The plan explicitly permits the CLIP-prompt fallback.
- Output range is roughly [-2, 3] rather than [1, 10]; only the ranking matters for top-N selection.
- Module-level `lru_cache` loads CLIP once and pre-encodes prompts; per-frame scoring is one image forward + one matmul.
- Replace with a trained taste head in v1 (step 17). The trained head can run on Linux where the SigLIP path also works if we want it later.

### Step 4 — top-N selection ✓
- Filter by sharpness threshold, sort surviving frames by aesthetic score, take top N (default 10, env var `BANGER_TOP_N`).
- If zero survive, log and exit 0.
- Done as the `--output` arm of cmd_run: cluster-best frames passing sharpness, sorted by aesthetic descending, top N. Cluster-best filter (from step 12) means burst dupes don't crowd the top.

### Step 5 — scene matching ✓
- Load CLIP once, encode the seven prompts from `CLAUDE.md` § Stage 7 once.
- For each survivor, encode the image, take cosine sim against each prompt, return `(best_prompt, score)`.
- If best score < 0.25, return the safe default ("crisp daylight outdoor portrait").
- Map prompt → preset filename via static dict in code.
- Done in `scenes.py`. Threshold lowered from 0.25 → 0.20 because empirically 97% of frames fell back at 0.25 — CLIP image-text sims for these prompts cluster tightly between 0.20 and 0.27 on real photos. Calibration to be revisited.

### Step 6 — darktable develop ✓ (sequential; parallelisation deferred)
- Shell out to `darktable-cli input output --style preset_name`. Capture stderr, log on non-zero exit.
- Fallback path: if darktable-cli fails, copy the source JPEG to output and log a warning. PIL tone-curve approximation can wait.
- Parallelise with `concurrent.futures.ProcessPoolExecutor`, worker count = `os.cpu_count() - 1`.
- Done in `develop.py`. Detects darktable-cli on PATH or in standard Windows install dirs; on the user's Windows it isn't present so all top-N go through `copy_fallback`. ProcessPoolExecutor parallelisation is deferred — top-10 sequential is ~50 ms (just file copies); will matter once darktable is in the loop.

### Step 7 — output and logging ✓
- Output folder: `~/Pictures/bangers/YYYY-MM-DD/` (configurable via `BANGER_OUTPUT_DIR`).
- Stdout logging via stdlib `logging` at INFO. Per-frame: path, sharpness, aesthetic score, chosen preset.
- Print a summary line at the end: "N kept, M rejected, T seconds".
- Done in `cmd_run` `--output` flag. Default top-N from env `BANGER_TOP_N`. The dated-folder default (`~/Pictures/bangers/YYYY-MM-DD/`) isn't auto-substituted — caller passes a path explicitly. `BANGER_OUTPUT_DIR` shorthand can be added if it turns out to be the typical invocation.

### Step 8 — hand-tune presets ✗ (manual, requires darktable on a Linux machine)
- Open darktable on the actual machine, build one preset per prompt against a representative photo, export each as `.xmp` into `presets/`.
- This is the unskippable manual step. Without good presets the rest is theatre.
- Pipeline plumbing is in place (`PROMPT_TO_PRESET` map, `develop.develop_to_jpeg` looks for `presets/<preset_name>.xmp`). The folder is empty until a darktable session produces real `.xmp` files.

### Step 9 — manual smoke test ✓ (preset pairing tuning ongoing)
- Point the CLI at a folder of ~100 mixed JPEGs from a real shoot. Eyeball the output.
- Decide: does the prompt → preset pairing produce sensible-looking results? If no, re-tune prompts before any further engineering.
- Smoke-tested on the user's 1033-frame test_photos with the trained taste head. Aesthetic ranking subjectively useful (top-N is from caminito hike — the user's labelled subset there). Preset pairing has very tight CLIP score margins so the bias toward `documentary_flat` is somewhat noisy; treat the prompt set as a v0 starting point that will need rewording once real presets exist.

## v0.5 — make it run on plug-in

Once v0 is proven on a fixtures folder, wire it to the camera.

### Step 10 — gphoto2 ingest ✗ (Linux only, camera not connected)
- `banger ingest` subcommand: shell out to `gphoto2 --camera "Sony Alpha-A6600" --get-all-files --filename "%f"` into a staging dir.
- Detect PTP-vs-MTP mismatch by checking gphoto2's exit code and stderr; print a clear "set USB Connection → PTP in the camera menu" message.
- Deferred until on Linux with the camera attached. Manual file copies into the input dir work fine for now.

### Step 11 — SQLite state ✗ (partial: labels DB done, processed-file tracking still open)
- `~/.local/share/banger-pipeline/state.db`, single table `processed(filename TEXT PRIMARY KEY, sha256 TEXT, processed_at INTEGER)`.
- Skip files already in the table at the start of ingest.
- Atomic move from staging to processed only after the row is committed.
- The label DB exists at `~/.local/share/banger-pipeline/labels.db` (sha-keyed scores). Embeddings, thumbnails, and hero previews are also cached on disk by sha. The "skip already-processed files on re-plug" idempotency path is what's still missing — coupled to step 10 (camera ingest).

### Step 12 — burst dedup ✓
- pHash via `imagehash` on each preview. Cluster by Hamming distance ≤ 6 within a 3-second EXIF timestamp window.
- Keep the highest aesthetic score per cluster. Log cluster sizes so I can tune the thresholds.
- Done in `dedup.py`; greedy time-ordered clustering, EXIF DateTimeOriginal with mtime fallback. Test_photos run found 55 bursts suppressing 60 dupes out of 1003 sharpness-pass frames (~6% reduction).

### Step 13 — udev + systemd ✗ (Linux only)
- udev rule matching the a6600's USB vendor/product IDs, runs `systemctl --user start banger-pipeline.service`.
- systemd user unit invokes `python -m banger run-from-camera`. `Type=oneshot`, no restart.
- Document the install steps in the README.
- Deferred — needs the Linux dual-boot.

### Step 14 — notify-send ✗ (Linux only)
- After the run finishes, emit a notification with kept/rejected counts.
- Click action opens the dated output folder. `gio open` is the simplest cross-DE way.
- Deferred — coupled to step 13.

### Step 15 — RAW preview extraction ✓
- Already done as part of step 2: `Frame.classify_path` falls back to the RAW for RAW-only frames, and `preview.load_preview` extracts the embedded JPEG via rawpy. Nothing extra to do unless a real shoot turns up an ARW with no embedded thumb (libraw raises `LibRawNoThumbnailError`, currently logged-and-skipped).

## v1 — make the cull actually mine

### Step 16 — thumbs-up/down log (PULLED FORWARD) ✓
- Done early: generic CLIP-prompt scoring failed empirically on real photos (it ranked a clean shot of a dog swimming away above the user's actual favourite, because CLIP's negatives punished the favourite's intentional shallow depth-of-field as if it were accidental blur). The plan called for the trained head only at v1, but with the prompt scoring this misranked, we needed the labelling tool now.
- Implemented: `banger label <input_dir> up|down <stem>...` writes a SQLite row keyed by sha256 of the source file. CLIP embeddings cached during `banger run` at `~/.local/share/banger-pipeline/embeddings/{sha}.npy`.

### Step 17 — trained taste head (PULLED FORWARD) ✓
- Done early: `banger train` reads the labels DB, looks up cached embeddings by sha, fits a sklearn Ridge regression (changed from LogisticRegression after the schema flipped from binary up/down to integer -5..+5 scores), saves to `~/.local/share/banger-pipeline/taste_head.joblib`. Leave-one-out MAE printed for sanity. `cmd_run` auto-loads the head if it exists and uses it for scoring; otherwise falls back to prompts. Plan said 200 labels; the head can be trained from any number with at least two examples, but is unreliable until at least 30-50.
- Bonus: a labelling UI lives in `server.py` (Flask app, iOS-photo-roll layout, -5..+5 keyboard shortcuts, live histogram). Goes well beyond the original step 16 spec but the user asked for it.

### Step 18 — face-aware sharpness ✓ (opt-in)
- Run a face detector (mediapipe is fine) on survivors. If a face is present, override the global sharpness threshold with an "eyes-in-focus" check on the face crop.
- Done with `face.py` using OpenCV's bundled Haar cascade (no extra dep). `--face-gate` flag on `run` rejects any frame whose sharpest face is below `FACE_SHARPNESS_THRESHOLD` (default 50). Face data is cached in metadata/<sha>.json so warm runs don't reload previews.

### Step 19 — per-preset confidence thresholds ✓ (plumbing in place; calibration TBD)
- Replace the single 0.25 cutoff with per-prompt thresholds tuned from logged matches.
- Anything below its prompt's threshold falls back to the safe default.
- `scenes.PER_PROMPT_THRESHOLDS` is the override dict; empty by default. `classify_with_emb(per_prompt_thresholds=...)` honours it. Populating sensible per-prompt values requires logged matches from real use, so the dict starts empty and the global threshold (0.20) governs everything until tuning evidence arrives.

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
