"""Local labelling UI: Flask app at localhost:<port>.

iOS photo-roll layout: one big hero image plus a horizontal filmstrip of
thumbnails along the bottom. Eye stays in the same place; left/right
arrows or number keys advance, the filmstrip auto-centres on the current
frame.

Walks the input directory recursively on startup and parallel-hashes
every frame's classify-path so labels can be keyed by sha256
(content-addressed, survives moves and renames). Thumbnails, hero
previews, and CLIP embeddings are encoded LAZILY:

  - 480 px thumbs on first /api/thumb/<sha>
  - 1024 px hero JPEGs on first /api/preview/<sha>
  - CLIP embeddings on first /api/label POST

That keeps startup fast (~1 s/1k frames; only sha hashing) and skips
encoder work entirely for frames that never enter view.

Differs from CLAUDE.md ("No web server. No frontend.") — the user
explicitly asked for a labelling frontend.
"""

import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template_string, request, send_file

from banger import aesthetic, state, taste_head
from banger.frames import discover_frames
from banger.preview import load_preview
from banger.report import encode_thumbnail_bytes

log = logging.getLogger("banger")

PREVIEW_JPEG_QUALITY = 88


def _hash_frames(frames):
    """Sha256 every frame's classify_path in parallel. Disk-bound, so threads work."""

    def hash_one(f):
        return f, state.sha256_of(f.classify_path)

    with ThreadPoolExecutor(max_workers=8) as exe:
        return list(exe.map(hash_one, frames))


def _encode_preview_jpeg(preview_bgr) -> bytes:
    ok, buf = cv2.imencode(".jpg", preview_bgr, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


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
    random.shuffle(frames_view)
    log.info("ready: %d frames (shuffled)", len(frames_view))

    locks_guard = threading.Lock()
    locks: dict[tuple[str, str], threading.Lock] = {}

    def _lock(kind: str, sha: str) -> threading.Lock:
        key = (kind, sha)
        with locks_guard:
            if key not in locks:
                locks[key] = threading.Lock()
            return locks[key]

    app = Flask(__name__)

    @app.route("/")
    def index():
        labels = state.labels_dict()
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
            with _lock("thumb", sha):
                if not path.exists():
                    try:
                        preview = load_preview(f.classify_path)
                    except Exception as e:
                        log.warning("thumb fail %s: %s", f.display_name, e)
                        return ("preview failed", 500)
                    state.cache_thumbnail(sha, encode_thumbnail_bytes(preview))
        return send_file(path, mimetype="image/jpeg")

    @app.route("/api/preview/<sha>")
    def preview(sha):
        f = sha_to_frame.get(sha)
        if f is None:
            return ("not found", 404)
        path = state.preview_jpeg_path(sha)
        if not path.exists():
            with _lock("preview", sha):
                if not path.exists():
                    try:
                        preview_arr = load_preview(f.classify_path)
                    except Exception as e:
                        log.warning("preview fail %s: %s", f.display_name, e)
                        return ("preview failed", 500)
                    state.cache_preview_jpeg(sha, _encode_preview_jpeg(preview_arr))
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

        if state.load_embedding(sha) is None:
            try:
                preview_arr = load_preview(f.classify_path)
                state.cache_embedding(sha, aesthetic.encode_image(preview_arr))
            except Exception as e:
                log.warning("embedding fail %s: %s", f.display_name, e)

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

    @app.route("/api/unlabelled-uncertain")
    def unlabelled_uncertain():
        """Return unlabelled frames ordered by |predicted score| ascending.

        Active-learning play: the head is least confident about frames whose
        predicted taste score sits near zero. Labelling those grows the training
        set on confusion cases instead of uniform sampling, which is where a
        personalised head gets its leverage with small N. Frames with no cached
        embedding fall to the back (we can't score them yet without preview load).
        """
        head = taste_head.load()
        labels = state.labels_dict()
        scored: list[tuple[float, dict, bool]] = []  # (|score|, view, scored?)
        for f_view in frames_view:
            sha = f_view["sha"]
            if sha in labels:
                continue
            if head is not None:
                emb = state.load_embedding(sha)
                if emb is not None:
                    pred = taste_head.predict_score(head, emb)
                    scored.append((abs(pred), {**f_view, "predicted": round(pred, 3)}, True))
                    continue
            scored.append((float("inf"), f_view, False))
        scored.sort(key=lambda t: t[0])
        return jsonify(
            {
                "head_loaded": head is not None,
                "ordered": [v for _, v, _ in scored],
                "scored_count": sum(1 for _, _, s in scored if s),
                "total_unlabelled": len(scored),
            }
        )

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
  html, body { height: 100%; }
  body { margin: 0; background: #0c0c0c; color: #eee; font-family: ui-sans-serif, system-ui, sans-serif; display: flex; flex-direction: column; overflow: hidden; }
  header { flex: 0 0 auto; padding: .5rem .75rem; border-bottom: 1px solid #2a2a2a; background: #111; display: flex; gap: .75rem; align-items: baseline; flex-wrap: wrap; }
  h1 { margin: 0; font-size: .9rem; }
  .summary { font-size: .75rem; color: #aaa; }
  .summary code { background: #222; padding: 1px 5px; border-radius: 3px; color: #ccc; }
  .help { font-size: .7rem; color: #888; margin-left: auto; }
  .help kbd { background: #222; border: 1px solid #333; border-radius: 3px; padding: 0 4px; font-family: ui-monospace, monospace; color: #bbb; }

  .hist-wrap { display: inline-flex; align-items: center; gap: .5rem; font-size: .7rem; color: #888; }
  .hist { display: inline-flex; align-items: flex-end; height: 32px; gap: 2px; padding: 0 4px; background: #161616; border: 1px solid #232323; border-radius: 3px; }
  .hist .bar { width: 9px; min-height: 2px; border-radius: 1px 1px 0 0; background: #444; transition: height .15s ease; }
  .hist .bar.up { background: #5fa05f; }
  .hist .bar.down { background: #a05f5f; }
  .hist .bar.zero { background: #888; }
  .hist .bar.empty { background: #2a2a2a; }
  .hist-axis { display: flex; gap: 2px; padding: 0 4px; font-size: .55rem; color: #666; font-variant-numeric: tabular-nums; }
  .hist-axis span { width: 9px; text-align: center; }

  #hero { flex: 1 1 auto; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 1rem; min-height: 0; gap: .75rem; }
  #hero-img-wrap { flex: 1 1 auto; min-height: 0; display: flex; align-items: center; justify-content: center; width: 100%; }
  #hero-img { max-width: 100%; max-height: 100%; object-fit: contain; box-shadow: 0 4px 32px rgba(0,0,0,.5); border-radius: 4px; background: #000; }
  #hero-info { flex: 0 0 auto; display: flex; flex-direction: column; align-items: center; gap: .35rem; }
  #hero-meta { font-size: .8rem; color: #ccc; font-family: ui-monospace, monospace; }
  #hero-meta .sub { color: #888; }
  #hero-meta .kind { color: #888; font-size: .7rem; margin-left: .35rem; }

  .scores { display: flex; gap: 3px; }
  .scores button { min-width: 38px; padding: .35rem 0; background: #1f1f1f; color: #888; border: 1px solid transparent; border-radius: 4px; cursor: pointer; font: inherit; font-size: .85rem; font-variant-numeric: tabular-nums; }
  .scores button:hover { background: #2c2c2c; color: #ddd; }
  .scores button.up { color: #5fa05f; }
  .scores button.down { color: #a05f5f; }
  .scores button.zero { color: #888; }
  .scores button.active { background: #2e3a2e; color: #d8eed8; border-color: #4a6a4a; }
  .scores button.down.active { background: #3a2e2e; color: #eed8d8; border-color: #6a4a4a; }
  .scores button.zero.active { background: #333; color: #eee; border-color: #666; }
  .clear { font-size: .65rem; color: #555; background: none; border: none; cursor: pointer; padding: 0; }
  .clear:hover { color: #888; }

  #filmstrip { flex: 0 0 96px; display: flex; gap: 4px; overflow-x: auto; padding: 8px 12px; background: #0a0a0a; border-top: 1px solid #2a2a2a; scrollbar-width: thin; }
  #filmstrip::-webkit-scrollbar { height: 6px; }
  #filmstrip::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
  .thumb { flex: 0 0 auto; height: 80px; aspect-ratio: 3/2; cursor: pointer; opacity: .55; transition: opacity .12s, transform .12s; border-radius: 3px; object-fit: cover; background: #000; border: 2px solid transparent; box-sizing: border-box; }
  .thumb:hover { opacity: .85; }
  .thumb.current { opacity: 1; outline: 2px solid #ffaa55; outline-offset: -2px; transform: scale(1.06); }
  .thumb.labelled { border-bottom-color: #5fa05f; }
  .thumb.labelled.down { border-bottom-color: #a05f5f; }
  .thumb.labelled.zero { border-bottom-color: #888; }
</style>
</head>
<body>
<header>
  <h1>banger label</h1>
  <div class="summary">
    <code>{{ input_dir }}</code> &middot; <span id="stats"></span>
  </div>
  <div class="hist-wrap" title="distribution of your scores">
    <div>
      <div id="hist" class="hist"></div>
      <div class="hist-axis">
        <span>-5</span><span></span><span></span><span></span><span></span>
        <span>0</span>
        <span></span><span></span><span></span><span></span><span>+5</span>
      </div>
    </div>
    <span id="hist-summary"></span>
  </div>
  <button id="mode-toggle" title="Switch ordering to uncertain frames (active learning)"
          style="margin-left: .5rem; background: #1f1f1f; color: #ccc; border: 1px solid #333; border-radius: 3px; padding: 2px 8px; font-size: .7rem; cursor: pointer;">
    order: random
  </button>
  <div class="help">
    <kbd>1</kbd>-<kbd>5</kbd> = -5..-1 &nbsp; <kbd>6</kbd>-<kbd>9</kbd>+<kbd>0</kbd> = +1..+5 &nbsp;·&nbsp;
    <kbd>←</kbd><kbd>→</kbd> move &nbsp;·&nbsp; <kbd>Space</kbd> skip &nbsp;·&nbsp; <kbd>Bksp</kbd> clear
  </div>
</header>

<section id="hero">
  <div id="hero-img-wrap">
    <img id="hero-img" alt="">
  </div>
  <div id="hero-info">
    <div id="hero-meta"></div>
    <div class="scores" id="hero-scores"></div>
    <button class="clear" id="hero-clear">clear label</button>
  </div>
</section>

<div id="filmstrip"></div>

<script>
let FRAMES = {{ frames | tojson }};
const LABELS = {{ labels | tojson }};
const SCORES = [-5,-4,-3,-2,-1,0,1,2,3,4,5];
const KEY_TO_SCORE = {
  '1': -5, '2': -4, '3': -3, '4': -2, '5': -1,
  '6': 1, '7': 2, '8': 3, '9': 4, '0': 5,
};
const PRELOAD_RANGE = 3;

let focusIdx = 0;

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function buildFilmstrip() {
  const strip = document.getElementById("filmstrip");
  strip.innerHTML = FRAMES.map((f, i) => {
    const score = LABELS[f.sha];
    const labelClass = score === undefined ? "" :
      "labelled " + (score > 0 ? "up" : score < 0 ? "down" : "zero");
    return `<img class="thumb ${labelClass}" loading="lazy"
                 data-idx="${i}" data-sha="${f.sha}"
                 src="/api/thumb/${f.sha}"
                 alt="${escapeHtml(f.display)}"
                 title="${escapeHtml(f.display)}${score !== undefined ? ' (' + (score>0?'+':'') + score + ')' : ''}">`;
  }).join("");
}

function buildScoreButtons() {
  const wrap = document.getElementById("hero-scores");
  wrap.innerHTML = SCORES.map(s => {
    const cls = s > 0 ? "up" : s < 0 ? "down" : "zero";
    const label = s > 0 ? "+" + s : s;
    return `<button data-score="${s}" class="${cls}" tabindex="-1">${label}</button>`;
  }).join("");
}

function refreshFilmstripThumb(sha) {
  const node = document.querySelector(`#filmstrip .thumb[data-sha="${sha}"]`);
  if (!node) return;
  const score = LABELS[sha];
  node.classList.remove("labelled", "up", "down", "zero");
  if (score !== undefined) {
    node.classList.add("labelled", score > 0 ? "up" : score < 0 ? "down" : "zero");
  }
  let title = node.dataset.display || node.alt;
  if (score !== undefined) title += " (" + (score > 0 ? "+" : "") + score + ")";
  node.title = title;
}

function refreshHero() {
  const f = FRAMES[focusIdx];
  if (!f) return;
  const img = document.getElementById("hero-img");
  img.src = "/api/preview/" + f.sha;
  img.alt = f.display;

  const meta = document.getElementById("hero-meta");
  const sub = f.subdir ? `<span class="sub">${escapeHtml(f.subdir)}/</span>` : "";
  meta.innerHTML = `${sub}${escapeHtml(f.stem)}<span class="kind">${escapeHtml(f.kind)}</span>`;

  const score = LABELS[f.sha];
  document.querySelectorAll("#hero-scores button").forEach(b => {
    b.classList.toggle("active", parseInt(b.dataset.score) === score);
  });

  document.querySelectorAll("#filmstrip .thumb.current").forEach(n => n.classList.remove("current"));
  const cur = document.querySelector(`#filmstrip .thumb[data-idx="${focusIdx}"]`);
  if (cur) {
    cur.classList.add("current");
    centreThumb(cur);
  }

  preloadAhead();
  updateStats();
}

function centreThumb(node) {
  const strip = document.getElementById("filmstrip");
  const target = node.offsetLeft - strip.clientWidth / 2 + node.offsetWidth / 2;
  strip.scrollTo({left: target, behavior: "smooth"});
}

function preloadAhead() {
  for (let off = -PRELOAD_RANGE; off <= PRELOAD_RANGE; off++) {
    if (off === 0) continue;
    const i = focusIdx + off;
    if (i < 0 || i >= FRAMES.length) continue;
    const img = new Image();
    img.src = "/api/preview/" + FRAMES[i].sha;
  }
}

function setFocus(idx) {
  if (idx < 0) idx = 0;
  if (idx >= FRAMES.length) idx = FRAMES.length - 1;
  focusIdx = idx;
  refreshHero();
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
  refreshFilmstripThumb(sha);
  if (FRAMES[focusIdx].sha === sha) {
    document.querySelectorAll("#hero-scores button").forEach(b => {
      b.classList.toggle("active", parseInt(b.dataset.score) === score);
    });
  }
  updateStats();
}

async function clearScore(sha) {
  const res = await fetch("/api/label/" + sha, {method: "DELETE"});
  if (!res.ok) return;
  delete LABELS[sha];
  refreshFilmstripThumb(sha);
  if (FRAMES[focusIdx].sha === sha) {
    document.querySelectorAll("#hero-scores button.active").forEach(b => b.classList.remove("active"));
  }
  updateStats();
}

function updateStats() {
  const labelled = FRAMES.filter(f => LABELS[f.sha] !== undefined).length;
  document.getElementById("stats").textContent =
    `${focusIdx + 1} / ${FRAMES.length} · ${labelled} labelled`;
  updateHistogram();
}

function updateHistogram() {
  const counts = {};
  SCORES.forEach(s => counts[s] = 0);
  Object.values(LABELS).forEach(s => { if (s in counts) counts[s]++; });
  const max = Math.max(1, ...Object.values(counts));

  const hist = document.getElementById("hist");
  hist.innerHTML = SCORES.map(s => {
    const c = counts[s];
    const cls = c === 0 ? "empty" : (s > 0 ? "up" : s < 0 ? "down" : "zero");
    const h = c === 0 ? 2 : Math.max(3, Math.round(c / max * 28));
    const label = `${s > 0 ? "+" : ""}${s}: ${c}`;
    return `<div class="bar ${cls}" style="height:${h}px" title="${label}"></div>`;
  }).join("");

  const summary = document.getElementById("hist-summary");
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  if (total === 0) {
    summary.textContent = "no labels yet";
  } else {
    let sum = 0;
    SCORES.forEach(s => sum += s * counts[s]);
    const mean = sum / total;
    summary.textContent = `mean ${mean >= 0 ? "+" : ""}${mean.toFixed(1)}`;
  }
}

document.getElementById("filmstrip").addEventListener("click", e => {
  const t = e.target.closest(".thumb");
  if (!t) return;
  setFocus(parseInt(t.dataset.idx));
});

document.getElementById("hero-scores").addEventListener("click", e => {
  if (!e.target.matches("button")) return;
  const sha = FRAMES[focusIdx]?.sha;
  if (sha) setScore(sha, parseInt(e.target.dataset.score));
});

document.getElementById("hero-clear").addEventListener("click", () => {
  const sha = FRAMES[focusIdx]?.sha;
  if (sha) clearScore(sha);
});

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const sha = FRAMES[focusIdx]?.sha;

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
  } else if (e.key === "Backspace") {
    e.preventDefault();
    if (sha) clearScore(sha);
  } else if (e.key === " ") {
    e.preventDefault();
    setFocus(focusIdx + 1);
  }
});

let mode = "random";
const modeToggle = document.getElementById("mode-toggle");
modeToggle.addEventListener("click", async () => {
  if (mode === "random") {
    modeToggle.textContent = "order: loading…";
    modeToggle.disabled = true;
    try {
      const res = await fetch("/api/unlabelled-uncertain");
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (!data.head_loaded) {
        modeToggle.textContent = "order: random (train head first)";
        modeToggle.disabled = false;
        return;
      }
      // Replace the in-memory frames list with the uncertainty-ordered one,
      // then append the labelled set at the end so nothing falls off the strip.
      const labelledSet = new Set(Object.keys(LABELS));
      const labelledFrames = FRAMES.filter(f => labelledSet.has(f.sha));
      FRAMES = [...data.ordered, ...labelledFrames];
      mode = "uncertain";
      modeToggle.textContent = `order: uncertain (n=${data.scored_count} scored)`;
      buildFilmstrip();
      setFocus(0);
    } catch (e) {
      console.error("uncertain fetch failed:", e);
      modeToggle.textContent = "order: random (fetch failed)";
    } finally {
      modeToggle.disabled = false;
    }
  } else {
    // We don't keep the original random ordering on the client; reloading
    // is the simplest way to reset and it's exactly what the user expects.
    location.reload();
  }
});

buildFilmstrip();
buildScoreButtons();
setFocus(0);
</script>
</body>
</html>
"""
