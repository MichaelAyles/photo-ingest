# TODO — pick up here after reboot

Last good commit: `e80fe97`. The whole stack (sharpness + CLIP aesthetic + label DB + trained head) is wired up but the final smoke tests were blocked by Windows commit-charge pressure (`ServiceShell` was at 13 GB RAM; OpenCV couldn't allocate 72 MB to read a JPEG).

## Run these in order

```sh
cd /c/Users/mikea/OneDrive/Desktop/Projects/photo-ingest
PY=.venv/Scripts/python.exe
```

### 1. Verify the rewritten prompts moved the needle on their own

Earlier we found CLIP was punishing DSC00055 (your favourite, shallow DOF) for "blurry, poorly framed photograph" — high similarity because the bokeh background reads as blur to CLIP. Rewrote the prompts to drop blur/focus negatives (sharpness gate already filters real blur) and add shallow-DOF positives. Expected impact: 55 should rise relative to 73, even before any labels.

```sh
$PY -m banger explain test_photos DSC00055 DSC00073
```

Compare to the pre-rewrite breakdown (in commit `349bebc`): mean(positives) was 0.2355 vs 0.2347 (tied), mean(negatives) was 0.2240 vs 0.1874 (75 winning here). If the rewrite worked, expect 55's mean(negatives) to drop because we removed the blur term that was firing on its bokeh.

### 2. End-to-end run with the new caching + head fallback path

Coded blind — not yet executed. Should:
- Score every frame's sharpness as before
- For survivors: encode CLIP, save embedding to `~/.local/share/banger-pipeline/embeddings/{sha}.npy`
- Fall back to prompt scoring (no head trained yet)
- Write HTML report with `prompts` source tag on each card

```sh
$PY -m banger run test_photos --report reports/full.html
ls ~/.local/share/banger-pipeline/embeddings/ | wc -l   # should be 110-ish
```

### 3. Build a label set and train

DSC00055 is already labelled `up` from earlier (test of `banger label`). Add maybe 10-20 more on each side from a quick scroll through `reports/full.html`:

```sh
$PY -m banger label test_photos up   DSC00055 <stem> <stem> ...
$PY -m banger label test_photos down DSC00073 <stem> <stem> ...
$PY -m banger train
```

Expected output: leave-one-out CV accuracy. Below ~30-50 labels it'll be noisy; just sanity-check it's above chance.

### 4. Re-run with the head and eyeball

```sh
$PY -m banger run test_photos --report reports/with_head.html
```

Report should now show `head` source tag (not `prompts`). Open and check whether 55 rises above 73, and whether the new top-10 looks more like your taste.

## If something went wrong

- **Memory pressure recurring**: see Get-Process check earlier; ServiceShell was the worst offender. May want to figure out what spawns it (Dell SupportAssist? a gaming launcher?) and disable auto-start.
- **`banger train` says "no cached embedding"**: step 2 didn't run; do step 2 first.
- **CLIP still segfaults**: that was a separate bug for SigLIP only (>2 GB safetensors), not CLIP-B/32 (~600 MB). If it bites here, see memory note in commit `e80fe97`.

## Once this works, delete this file.

It's session ephemera, not a project artefact.
