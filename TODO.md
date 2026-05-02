# TODO — pick up here

Memory pressure eased before reboot, so steps 1 and 2 are done. Last good commit at the time of writing this update: caching + new prompts validated end-to-end, 110 embeddings cached at `~/.local/share/banger-pipeline/embeddings/`.

## Run these in order

```sh
cd /c/Users/mikea/OneDrive/Desktop/Projects/photo-ingest
PY=.venv/Scripts/python.exe
```

### ~~1. Verify the rewritten prompts moved the needle on their own~~ DONE

Result: DSC00055 went from 0.57 to 1.47, DSC00073 dropped from 2.37 to 1.11. The user's favourite is now ranked above the swimming-away dog with the new prompts alone.

### ~~2. End-to-end run with the new caching + head fallback path~~ DONE

110 embeddings cached. Full run took 35 s. Aesthetic range -1.83 to 2.04, median 0.70 (shifted positive because the rewritten negatives are softer than the old "blurry/cluttered/amateur" set).

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
