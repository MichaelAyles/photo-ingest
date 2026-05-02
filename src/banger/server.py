"""Local labelling UI: Flask app at localhost:<port>.

Walks the input directory recursively on startup and parallel-hashes every
frame's classify-path so labels can be keyed by sha256 (content-addressed,
survives moves and renames). Thumbnails and CLIP embeddings are encoded
LAZILY — thumbs on the first /api/thumb/<sha> request, embeddings on the
first /api/label POST for a frame. That keeps startup fast (~10 s for
~1000 frames; only sha hashing) and skips CLIP work entirely for frames
you never label.

Differs from CLAUDE.md ("No web server. No frontend.") — the user
explicitly asked for a labelling frontend.
"""

import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_file

from banger import aesthetic, state
from banger.frames import discover_frames
from banger.preview import load_preview
from banger.report import encode_thumbnail_bytes

log = logging.getLogger("banger")


def _hash_frames(frames):
    """Sha256 every frame's classify_path in parallel. Disk-bound, so threads work."""

    def hash_one(f):
        return f, state.sha256_of(f.classify_path)

    with ThreadPoolExecutor(max_workers=8) as exe:
        return list(exe.map(hash_one, frames))


def serve(input_dir: Path, port: int = 8000) -> None:
    if not input_dir.is_dir():
        raise SystemExit(f"not a directory: {input_dir}")

    log.info("discovering frames in %s", input_dir)
    frames = discover_frames(input_dir, recursive=True)
    if not frames:
        raise SystemExit(f"no supported images in {input_dir} (recursive)")

    log.info("hashing %d frames (parallel)…", len(frames))
    pairs = _hash_frames(frames)
    sha_to_frame = {sha: f for f, sha in pairs}

    frames_view = [
        {
            "sha": sha,
            "stem": f.stem,
            "subdir": f.subdir,
            "display": f.display_name,
            "kind": f.kind,
        }
        for f, sha in pairs
    ]
    # Shuffle so labelling samples the dataset evenly and the user doesn't get
    # a long run of near-duplicate consecutive frames. Stable across one
    # server session.
    random.shuffle(frames_view)
    log.info("ready: %d frames (shuffled)", len(frames_view))

    # Encoders can race when multiple cards' thumbnails load concurrently —
    # serialise so we don't double-encode the same sha.
    thumb_locks: dict[str, threading.Lock] = {}
    thumb_locks_guard = threading.Lock()

    def _thumb_lock(sha: str) -> threading.Lock:
        with thumb_locks_guard:
            if sha not in thumb_locks:
                thumb_locks[sha] = threading.Lock()
            return thumb_locks[sha]

    app = Flask(__name__)

    @app.route("/")
    def index():
        labels = state.labels_dict()
        # Put unlabelled frames first so the action surface is at the top of
        # the grid; labelled frames stay visible below for review.
        unlabelled = [f for f in frames_view if f["sha"] not in labels]
        labelled = [f for f in frames_view if f["sha"] in labels]
        return render_template_string(
            _TEMPLATE,
            frames=unlabelled + labelled,
            labels=labels,
            input_dir=str(input_dir),
        )

    @app.route("/api/thumb/<sha>")
    def thumb(sha):
        f = sha_to_frame.get(sha)
        if f is None:
            return ("not found", 404)
        path = state.thumbnail_path(sha)
        if not path.exists():
            with _thumb_lock(sha):
                if not path.exists():
                    try:
                        preview = load_preview(f.classify_path)
                    except Exception as e:
                        log.warning("thumb fail %s: %s", f.display_name, e)
                        return ("preview failed", 500)
                    state.cache_thumbnail(sha, encode_thumbnail_bytes(preview))
        return send_file(path, mimetype="image/jpeg")

    @app.route("/api/label", methods=["POST"])
    def set_label():
        data = request.get_json(silent=True) or {}
        sha = data.get("sha")
        score = data.get("score")
        f = sha_to_frame.get(sha)
        if f is None:
            return ("unknown sha", 404)
        if score is None:
            return ("missing score", 400)
        try:
            score = int(score)
        except (TypeError, ValueError):
            return ("score must be an integer", 400)
        if not state.SCORE_MIN <= score <= state.SCORE_MAX:
            return (f"score out of [{state.SCORE_MIN}, {state.SCORE_MAX}]", 400)

        # Encode CLIP embedding lazily on first label so train can use it later.
        if state.load_embedding(sha) is None:
            try:
                preview = load_preview(f.classify_path)
                state.cache_embedding(sha, aesthetic.encode_image(preview))
            except Exception as e:
                log.warning("embedding fail %s: %s", f.display_name, e)
                # Label anyway; train will warn about missing embeddings.

        state.add_label(sha, score, f.stem, str(f.classify_path))
        return jsonify({"sha": sha, "score": score})

    @app.route("/api/label/<sha>", methods=["DELETE"])
    def clear_label(sha):
        if sha not in sha_to_frame:
            return ("unknown sha", 404)
        import sqlite3

        with sqlite3.connect(state.LABELS_DB) as conn:
            conn.execute("DELETE FROM labels WHERE sha256=?", (sha,))
        return jsonify({"sha": sha, "score": None})

    @app.route("/api/stats")
    def stats():
        labels = state.labels_dict()
        in_view = {sha: s for sha, s in labels.items() if sha in sha_to_frame}
        return jsonify(
            {
                "total": len(frames_view),
                "labelled": len(in_view),
                "global_labelled": len(labels),
            }
        )

    log.info("serving %s on http://127.0.0.1:%d", input_dir, port)
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False, threaded=True)


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>banger label — {{ input_dir }}</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 1rem; background: #111; color: #eee; }
  header { position: sticky; top: 0; background: #111; padding: .25rem 0 .75rem; border-bottom: 1px solid #2a2a2a; margin-bottom: .75rem; z-index: 10; }
  h1 { margin: 0 0 .25rem; font-size: 1rem; }
  .summary { font-size: .8rem; color: #aaa; }
  .summary code { background: #222; padding: 1px 5px; border-radius: 3px; color: #ccc; }
  .help { font-size: .7rem; color: #888; margin-top: .25rem; }
  .help kbd { background: #222; border: 1px solid #333; border-radius: 3px; padding: 0 4px; font-family: ui-monospace, monospace; color: #bbb; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: .75rem; }
  .card { margin: 0; background: #1a1a1a; border-radius: 6px; overflow: hidden; border: 1px solid #2a2a2a; scroll-margin-top: 6rem; }
  .card.labelled { border-color: #3a4a3a; }
  .card.focused { outline: 3px solid #ffaa55; outline-offset: -3px; }
  .card img { width: 100%; display: block; aspect-ratio: 3/2; object-fit: cover; cursor: pointer; }
  figcaption { padding: .4rem .55rem; font-size: .75rem; }
  .head { display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; }
  .stem { font-family: ui-monospace, monospace; }
  .kind { font-size: .65rem; color: #888; }
  .subdir { font-size: .65rem; color: #888; }
  .scores { display: flex; gap: 2px; margin-top: .35rem; flex-wrap: wrap; }
  .scores button { flex: 1 1 0; min-width: 24px; padding: .25rem 0; background: #222; color: #888; border: 1px solid transparent; border-radius: 3px; cursor: pointer; font: inherit; font-size: .7rem; font-variant-numeric: tabular-nums; }
  .scores button:hover { background: #2c2c2c; color: #ddd; }
  .scores button.up { color: #5fa05f; }
  .scores button.down { color: #a05f5f; }
  .scores button.zero { color: #888; }
  .scores button.active { background: #2e3a2e; color: #d8eed8; border-color: #4a6a4a; }
  .scores button.down.active { background: #3a2e2e; color: #eed8d8; border-color: #6a4a4a; }
  .scores button.zero.active { background: #333; color: #eee; border-color: #666; }
  .clear { margin-top: .25rem; font-size: .65rem; color: #555; background: none; border: none; cursor: pointer; padding: 0; }
  .clear:hover { color: #888; }
</style>
</head>
<body>
<header>
  <h1>banger label</h1>
  <div class="summary">
    <code>{{ input_dir }}</code> &middot;
    <span id="stats"></span>
  </div>
  <div class="help">
    <kbd>1</kbd>-<kbd>5</kbd> = -5 to -1 &nbsp; <kbd>6</kbd>-<kbd>9</kbd> + <kbd>0</kbd> = +1 to +5 &nbsp;·&nbsp;
    <kbd>←</kbd> <kbd>→</kbd> move &nbsp;·&nbsp; <kbd>Space</kbd> skip &nbsp;·&nbsp; <kbd>Backspace</kbd> clear
  </div>
</header>
<main class="grid" id="grid"></main>
<script>
const FRAMES = {{ frames | tojson }};
const LABELS = {{ labels | tojson }};
const SCORES = [-5,-4,-3,-2,-1,0,1,2,3,4,5];
const KEY_TO_SCORE = {
  '1': -5, '2': -4, '3': -3, '4': -2, '5': -1,
  '6': 1, '7': 2, '8': 3, '9': 4, '0': 5,
};

let focusIdx = 0;

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function cardHtml(f, idx) {
  const score = LABELS[f.sha];
  const labelled = score !== undefined ? "labelled" : "";
  const buttons = SCORES.map(s => {
    const cls = (s > 0 ? "up" : s < 0 ? "down" : "zero") + (s === score ? " active" : "");
    const label = s > 0 ? "+" + s : s;
    return `<button data-score="${s}" class="${cls}" tabindex="-1">${label}</button>`;
  }).join("");
  const sub = f.subdir ? `<span class="subdir">${escapeHtml(f.subdir)}</span>` : "";
  return `
    <figure class="card ${labelled}" data-sha="${f.sha}" data-idx="${idx}">
      <img loading="lazy" src="/api/thumb/${f.sha}" alt="${escapeHtml(f.display)}">
      <figcaption>
        <div class="head">
          <span class="stem">${escapeHtml(f.stem)}</span>
          <span class="kind">${escapeHtml(f.kind)}</span>
        </div>
        ${sub}
        <div class="scores">${buttons}</div>
        <button class="clear" tabindex="-1">clear label</button>
      </figcaption>
    </figure>
  `;
}

function render() {
  document.getElementById("grid").innerHTML = FRAMES.map(cardHtml).join("");
  setFocus(0);
  updateStats();
}

function cards() {
  return document.querySelectorAll(".card");
}

function setFocus(idx) {
  const all = cards();
  if (all.length === 0) return;
  idx = Math.max(0, Math.min(idx, all.length - 1));
  document.querySelectorAll(".card.focused").forEach(c => c.classList.remove("focused"));
  focusIdx = idx;
  all[idx].classList.add("focused");
  all[idx].scrollIntoView({block: "nearest", behavior: "smooth"});
}

async function setScore(sha, score) {
  const res = await fetch("/api/label", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({sha, score})
  });
  if (!res.ok) {
    console.error("label failed:", await res.text());
    return;
  }
  LABELS[sha] = score;
  refreshCard(sha);
}

async function clearScore(sha) {
  const res = await fetch("/api/label/" + sha, {method: "DELETE"});
  if (!res.ok) return;
  delete LABELS[sha];
  refreshCard(sha);
}

function refreshCard(sha) {
  const card = document.querySelector(`.card[data-sha="${sha}"]`);
  if (!card) return;
  const score = LABELS[sha];
  card.classList.toggle("labelled", score !== undefined);
  card.querySelectorAll(".scores button").forEach(b => {
    b.classList.toggle("active", parseInt(b.dataset.score) === score);
  });
  updateStats();
}

function updateStats() {
  const labelled = FRAMES.filter(f => LABELS[f.sha] !== undefined).length;
  document.getElementById("stats").textContent = `${FRAMES.length} frames · ${labelled} labelled`;
}

document.addEventListener("click", e => {
  const card = e.target.closest(".card");
  if (!card) return;
  const idx = parseInt(card.dataset.idx);
  if (!Number.isNaN(idx)) setFocus(idx);
  const sha = card.dataset.sha;
  if (e.target.matches(".scores button")) {
    setScore(sha, parseInt(e.target.dataset.score));
  } else if (e.target.matches(".clear")) {
    clearScore(sha);
  }
});

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;

  const all = cards();
  if (all.length === 0) return;
  const sha = all[focusIdx]?.dataset.sha;

  if (e.key in KEY_TO_SCORE) {
    e.preventDefault();
    if (sha) setScore(sha, KEY_TO_SCORE[e.key]);
    setFocus(focusIdx + 1);
  } else if (e.key === "ArrowRight" || e.key === "l") {
    e.preventDefault();
    setFocus(focusIdx + 1);
  } else if (e.key === "ArrowLeft" || e.key === "h") {
    e.preventDefault();
    setFocus(focusIdx - 1);
  } else if (e.key === "ArrowDown" || e.key === "j") {
    e.preventDefault();
    // Approximate column count by comparing card top offsets.
    const cur = all[focusIdx];
    const nextRow = Array.from(all).findIndex((c, i) => i > focusIdx && c.offsetTop > cur.offsetTop);
    if (nextRow >= 0) {
      const cols = nextRow - all.findIndex((c, i) => c.offsetTop === cur.offsetTop);
      setFocus(focusIdx + cols);
    }
  } else if (e.key === "ArrowUp" || e.key === "k") {
    e.preventDefault();
    const cur = all[focusIdx];
    const colsAbove = Array.from(all).slice(0, focusIdx).reverse().findIndex(c => c.offsetTop < cur.offsetTop);
    if (colsAbove >= 0) {
      setFocus(focusIdx - colsAbove - 1);
    } else {
      setFocus(0);
    }
  } else if (e.key === "Backspace") {
    e.preventDefault();
    if (sha) clearScore(sha);
  } else if (e.key === " ") {
    e.preventDefault();
    setFocus(focusIdx + 1);
  }
});

render();
</script>
</body>
</html>
"""
