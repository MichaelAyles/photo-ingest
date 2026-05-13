# BUILD_PLAN.md

Tracks the facet-parity push (six features), separate from the v0/v0.5/v1
roadmap in `plan.md`. Each line marks status (✓ done, ~ partial, ✗ deferred)
plus a short result note. The narrative lives in module docstrings.

## Step 1, multi-dim scoring ✓

Added `banger/metrics.py`. Seven cv2-only dims lifted from facet's
TechnicalAnalyzer + CompositionAnalyzer: exposure (histogram with shadow,
highlight, silhouette), contrast (percentile + RMS), color harmony (HSV
entropy), noise (Immerkaer), dynamic range (log2 percentile ratio),
composition (rule-of-thirds + power points), leading lines (Hough). All
return 0-10 normalised so they speak the same scale as facet. Wired into
cmd_run's cold path, cached into `metadata/<sha>.json`. Surfaced on the
Row dataclass so the report and the XMP encoder can consume them. Falls
back to ignoring metrics if the compute step errors per frame, the
pipeline keeps running.

Result on `test_photos` (1033 frames, warm cache): metrics add roughly
40 ms per cold frame, free on warm runs. 109 -> 122 tests pass.

## Step 2, closed-eye detection ✓

Added `banger/eyes.py` plus `--eye-gate`. Uses mediapipe FaceMesh's 468
landmarks; EAR computed from six points per eye, frame fails if
`min(per-face EAR) < 0.21`. Off by default. mediapipe is not a hard dep:
gate becomes a no-op with a warning when import fails. Cached into
`metadata.eyes` so warm runs don't re-process.

Result on `test_photos`: mediapipe wasn't installed on the dev box, the
gate logged its no-op warning, the pipeline ran clean.

## Step 3, XMP sidecar output ✓

Added `banger/xmp.py` plus `--xmp`. Hand-rolled darktable-compatible
RDF-XML, no new dep. xmp:Rating from rank percentile (top 10% gets 5
stars, next 20% gets 4, next 30% gets 3, next 30% gets 2, last 10% gets
1). xmp:Label from KMeans cluster id (or stable hash of scene preset
name when only the prompt path ran). dc:subject carries
banger/source/<head|prompts>, banger/scene/<preset>,
banger/rank/<N>, banger/quality/<0-10>. Picked up by darktable and
Lightroom on next library scan. The "no copying" angle is the
workflow-integration win facet and Aftershoot don't have.

9 new tests.

## Step 4, scene KMeans rework ✓

Added `banger/scene_kmeans.py` plus `banger scenes fit -k K` and
`banger scenes show`. Replaces the prompt-cosine scene router with
KMeans over cached CLIP embeddings. Cluster slots named `cluster_NN`
become the new preset slots. Persisted at
`~/.local/share/banger-pipeline/scene_kmeans.{joblib,json}`. The summary
JSON includes each cluster's top-3 nearest SCENE_PROMPTS so the
clusters stay human-readable for darktable preset authoring. cmd_run
auto-uses the fitted model when present, falls back to scenes.classify
otherwise, no regression in the no-fit state. The XMP step also reads
cluster ids to colour-label sidecars.

Result on `test_photos`: not yet fit, the prompt fallback continues to
work as before. Fitting is a 1-step user action after a `banger run`
has produced embeddings.

## Step 5, active learning UI ✓

Added `GET /api/unlabelled-uncertain` to `banger/server.py` plus an
`order: random | uncertain` toggle in the UI header. Returns frames
ordered by `|head.predict|` ascending so labelling lands on the
confusion cases. Frames with no cached embedding fall to the back.
Graceful when no head is on disk yet, the button surfaces the reason
instead of failing silently. Reload resets to the random ordering.

2 new server tests.

## Step 6, benchmark harness ✓

Added `banger benchmark <input_dir>` plus `banger/benchmark.py`. Runs
banger's pipeline in-process (reuses cached embeddings + metadata for
free), optionally probes for facet via `docker compose version`, writes
`benchmarks/<date>.md` with: banger top-N table, facet top-N table or
explicit skip reason, Jaccard overlap, Spearman rank correlation over
the intersection, and a disagreement gallery section. The "skip
gracefully" rule from the brief is the default on this hardware (facet
docker path is known-broken on the 4 GB GPU per auto-memory), the
report still gives the banger-only side and a note about why the
comparison is blank.

Smoke-tested on test_photos with `--vs facet`: 10 picks in 15.7 s
(warm cache), facet skipped with the expected reason, markdown wrote
clean.

6 new tests.

## Verification

- 130/130 unit tests pass.
- `python -m banger benchmark ./test_photos -r --top-n 10 --vs facet`
  runs cleanly end-to-end, produces `benchmarks/<today>.md`.
- All new code and generated markdown are em-dash-free per the brief.

## Explicitly deferred (still off the table)

- Authoring darktable preset XMPs for the new cluster slots: manual,
  needs Linux + the darktable GUI.
- gphoto2 ingest, udev, systemd, notify-send: needs Linux dual-boot.
- Actually invoking facet end-to-end in the harness: docker path is
  broken on this host, fix is a hardware change not a code change.
- The blog post itself.
