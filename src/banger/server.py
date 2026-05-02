"""Local labelling UI: Flask app at localhost:<port>.

Walks the input directory recursively on startup, hashes each frame's
classify-path, and ensures both a CLIP embedding and a thumbnail JPEG
are cached for it. Serves a grid of thumbnails with -5..+5 score buttons
per card; clicks POST to /api/label and persist via banger.state.

Differs from CLAUDE.md ("No web server. No frontend."), but the user
explicitly asked for a labelling frontend.
"""

import logging
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_file

from banger import aesthetic, state
from banger.frames import discover_frames
from banger.preview import load_preview
from banger.report import encode_thumbnail_bytes

log = logging.getLogger("banger")


def _prepare(input_dir: Path) -> list[dict]:
    """Walk dir, ensure embedding + thumb cached for every frame, return view models."""
    frames = discover_frames(input_dir, recursive=True)
    log.info("found %d frames under %s", len(frames), input_dir)

    view: list[dict] = []
    for i, f in enumerate(frames, 1):
        sha = state.sha256_of(f.classify_path)
        thumb = state.thumbnail_path(sha)
        emb = state.load_embedding(sha)

        if not thumb.exists() or emb is None:
            try:
                preview = load_preview(f.classify_path)
            except Exception as e:
                log.warning("skip %s: %s", f.display_name, e)
                continue
            if not thumb.exists():
                state.cache_thumbnail(sha, encode_thumbnail_bytes(preview))
            if emb is None:
                state.cache_embedding(sha, aesthetic.encode_image(preview))
            log.info("prepared %d/%d: %s", i, len(frames), f.display_name)

        view.append(
            {
                "sha": sha,
                "stem": f.stem,
                "subdir": f.subdir,
                "display": f.display_name,
                "kind": f.kind,
                "src_path": str(f.classify_path),
            }
        )
    return view


def serve(input_dir: Path, port: int = 8000) -> None:
    if not input_dir.is_dir():
        raise SystemExit(f"not a directory: {input_dir}")

    frames_view = _prepare(input_dir)
    if not frames_view:
        raise SystemExit(f"no supported images in {input_dir} (recursive)")
    by_sha = {f["sha"]: f for f in frames_view}

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(
            _TEMPLATE,
            frames=frames_view,
            labels=state.labels_dict(),
            input_dir=str(input_dir),
        )

    @app.route("/api/thumb/<sha>")
    def thumb(sha):
        path = state.thumbnail_path(sha)
        if not path.exists():
            return ("not found", 404)
        return send_file(path, mimetype="image/jpeg")

    @app.route("/api/label", methods=["POST"])
    def set_label():
        data = request.get_json(silent=True) or {}
        sha = data.get("sha")
        score = data.get("score")
        if sha not in by_sha:
            return ("unknown sha", 404)
        if score is None:
            return ("missing score", 400)
        try:
            score = int(score)
        except (TypeError, ValueError):
            return ("score must be an integer", 400)
        if not state.SCORE_MIN <= score <= state.SCORE_MAX:
            return (f"score out of [{state.SCORE_MIN}, {state.SCORE_MAX}]", 400)

        f = by_sha[sha]
        state.add_label(sha, score, f["stem"], f["src_path"])
        return jsonify({"sha": sha, "score": score})

    @app.route("/api/label/<sha>", methods=["DELETE"])
    def clear_label(sha):
        if sha not in by_sha:
            return ("unknown sha", 404)
        import sqlite3

        with sqlite3.connect(state.LABELS_DB) as conn:
            conn.execute("DELETE FROM labels WHERE sha256=?", (sha,))
        return jsonify({"sha": sha, "score": None})

    @app.route("/api/stats")
    def stats():
        labels = state.labels_dict()
        in_view = {sha: s for sha, s in labels.items() if sha in by_sha}
        return jsonify(
            {
                "total": len(frames_view),
                "labelled": len(in_view),
                "global_labelled": len(labels),
            }
        )

    log.info("serving %s on http://127.0.0.1:%d", input_dir, port)
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


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
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: .75rem; }
  .card { margin: 0; background: #1a1a1a; border-radius: 6px; overflow: hidden; border: 1px solid #2a2a2a; }
  .card.labelled { border-color: #3a4a3a; }
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
</header>
<main class="grid" id="grid"></main>
<script>
const FRAMES = {{ frames | tojson }};
const LABELS = {{ labels | tojson }};
const SCORES = [-5,-4,-3,-2,-1,0,1,2,3,4,5];

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function cardHtml(f) {
  const score = LABELS[f.sha];
  const labelled = score !== undefined ? "labelled" : "";
  const buttons = SCORES.map(s => {
    const cls = (s > 0 ? "up" : s < 0 ? "down" : "zero") + (s === score ? " active" : "");
    const label = s > 0 ? "+" + s : s;
    return `<button data-score="${s}" class="${cls}">${label}</button>`;
  }).join("");
  const sub = f.subdir ? `<span class="subdir">${escapeHtml(f.subdir)}</span>` : "";
  return `
    <figure class="card ${labelled}" data-sha="${f.sha}">
      <img loading="lazy" src="/api/thumb/${f.sha}" alt="${escapeHtml(f.display)}">
      <figcaption>
        <div class="head">
          <span class="stem">${escapeHtml(f.stem)}</span>
          <span class="kind">${escapeHtml(f.kind)}</span>
        </div>
        ${sub}
        <div class="scores">${buttons}</div>
        <button class="clear">clear label</button>
      </figcaption>
    </figure>
  `;
}

function render() {
  document.getElementById("grid").innerHTML = FRAMES.map(cardHtml).join("");
  updateStats();
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
  const sha = card.dataset.sha;
  if (e.target.matches(".scores button")) {
    setScore(sha, parseInt(e.target.dataset.score));
  } else if (e.target.matches(".clear")) {
    clearScore(sha);
  }
});

render();
</script>
</body>
</html>
"""
