# TODO — when you're back

The /ralph-loop autopilot worked through the v0 plan + selected v1 items. Snapshot at commit `071d3b0`:

## What got done in this run

Iteration started: 49 tests, v0 algorithmic stages just finished. Iteration ended:
- **89 tests passing** in ~6 s, including a CLI integration test that exercises `cmd_run` end-to-end with mocked CLIP.
- **Burst dedup** (step 12) — pHash + EXIF time, found 55 bursts (60 dupes) in test_photos.
- **Scene matching** (step 5) — 7-prompt CLIP zero-shot, threshold tuned 0.25 → 0.20.
- **Per-prompt thresholds** (step 19) — plumbing in `scenes.PER_PROMPT_THRESHOLDS`; empty until tuning evidence accumulates.
- **Top-N output + manifest** (steps 6, 7, 9) — `--output DIR --top-n N` writes ranked JPEGs + manifest.json. Default output dir from `BANGER_OUTPUT_DIR` env, with auto-appended date subdir.
- **Develop fallback** — copies camera JPEG when darktable-cli isn't on PATH; calls darktable-cli when present.
- **Metadata cache** — sha-keyed JSON files in `~/.local/share/banger-pipeline/metadata/`. Warm runs are 7× faster (test_photos: 188 s → 27 s).
- **Face-aware sharpness gate** (step 18) — `--face-gate` flag, OpenCV Haar cascade, face data cached alongside other metadata.
- **CSV labels export/import** — `banger labels list / export PATH / import PATH` for moving labels between machines.

Git log between `1a5ad8b` (start of this run) and `071d3b0` is the full breadcrumb trail; each commit is a self-contained change with its own diff and reasoning.

## Things still genuinely open

1. **Open `reports/final.html`** — the latest full diagnostic grid (1033 frames, dedup-aware, scene-classified). Top-10 ranked JPEGs are at `reports/final_top10/` with `manifest.json` next to them. If the rank-1 frames look like keepers, the loop is closing. If they look wrong, label another batch via the UI and retrain.
2. **Tune scene prompts.** ~60% of frames classified as `documentary_flat` — that prompt is too generic. Real fix needs your taste, not mine; current prompts are a v0 placeholder.
3. **Build real `presets/*.xmp`** in darktable when on Linux. Until then the develop stage just copies the camera JPEG and notes the would-have-been preset in the manifest.
4. **Linux story** — gphoto2 ingest (step 10), udev/systemd (step 13), notify-send (step 14). All blocked on the dual-boot.

## Architecture left undone

- **CLI is ~530 lines.** `cli.py:cmd_run` is the heaviest function (~180 lines). Worth a refactor when the next feature lands. Held back here because the loop wanted features over refactors.
- **Develop is sequential.** Designed to be ProcessPoolExecutor-parallel; not material until darktable is in the loop and copy stops being instant.
- **Per-prompt thresholds aren't populated.** The dict is empty by design — needs logged real-use data to know which prompts deserve which cut-off.
- **No `aesthetic.encode_image` test.** It calls real CLIP. Indirectly exercised through the integration test (which mocks it out).

## Once this looks right, delete this file.
