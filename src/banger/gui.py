"""Native-window GUI for banger.

pywebview opens an OS-native webview pointed at a local Flask app. Same
WebView2 / WKWebView control Tauri would use, so "snappy" comes from
the runtime, and the dev cost stays low (no Rust toolchain, no JS build
chain, no bundler).

Architecture:
- One Flask app on 127.0.0.1:<port>. Pages are rendered as a single SPA
  (inline HTML/CSS/JS, no build step).
- /api/run kicks off a background thread that walks a folder, runs the
  banger pipeline, and posts progress to a JobState object. The thread
  re-uses the per-frame logic from cmd_run rather than shelling out so
  we can stream progress at frame granularity.
- /api/folder-pick calls back into the pywebview process to show the
  native OS folder dialog. That's the one piece a browser fundamentally
  can't do as nicely as a native shell.
- Labelling is the existing server.py page; we embed it under /label so
  "label these frames" is one click from the results screen.

This module deliberately stays self-contained: the SPA template, the
job runner, and the launcher all live here. If it gets larger than ~800
lines, that's the signal to break the template out into a static file.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_file

from banger import (
    aesthetic,
    dedup,
    face,
    face_names,
    fsutil,
    library,
    scene_kmeans,
    scenes,
    select,
    state,
    taste_head,
)
from banger import (
    eyes as eyes_mod,
)
from banger import (
    face_id as face_id_mod,
)
from banger import metrics as metrics_mod
from banger import settings as settings_mod
from banger import (
    tagger as tagger_mod,
)
from banger import (
    tags as tags_mod,
)
from banger import xmp as xmp_mod
from banger.frames import discover_frames
from banger.preview import load_preview
from banger.report import Row, encode_thumbnail_bytes
from banger.sharpness import sharpness_from_preview

log = logging.getLogger("banger.gui")


# ---------------------------------------------------------------------------
# Job state and runner
# ---------------------------------------------------------------------------

@dataclass
class JobState:
    """State for one running pipeline. Mutated by the worker thread, read by HTTP."""

    job_id: str
    input_dir: Path
    options: dict
    status: str = "queued"  # queued | running | done | error
    stage: str = "starting"
    progress: int = 0  # frames processed
    total: int = 0
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    error: str | None = None
    picks: list[dict] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    cluster_ids: dict[str, int] = field(default_factory=dict)
    xmp_written: int = 0

    def to_view(self) -> dict:
        elapsed = (self.finished_at or time.monotonic()) - self.started_at
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "total": self.total,
            "elapsed": round(elapsed, 1),
            "error": self.error,
            "picks": self.picks,
            "tail": self.log_lines[-30:],
            "xmp_written": self.xmp_written,
        }


_jobs: dict[str, JobState] = {}
_jobs_lock = threading.Lock()
# In-flight scoring jobs keyed by absolute input_dir path. A second /api/run
# for the same path returns the running job_id instead of spawning a duplicate.
_active_runs: dict[str, str] = {}  # path -> job_id
_recent_folders: list[str] = []
_label_subprocesses: dict[str, dict] = {}  # input_dir -> {"port": int, "proc": subprocess.Popen}

# Library-side state. _scan_progress holds the most recent ScanProgress object
# for each root being indexed; the API surfaces it for the progress bar.
_scan_progress: dict[int, library.ScanProgress] = {}
_scan_lock = threading.Lock()

# Auto-setup status, polled by the SPA splash. Updated by _auto_setup as it
# walks through scene fit / mediapipe install. The splash overlay polls
# /api/setup-status and disappears once phase == "ready".
_setup_state: dict = {
    "phase": "starting",   # starting | clip | scenes | mediapipe | ready
    "message": "starting up…",
    "started_at": time.monotonic(),
}


def _record_folder(folder: Path) -> None:
    """Most-recent-first list of folders the user has run on. Bounded at 6."""
    s = str(folder)
    if s in _recent_folders:
        _recent_folders.remove(s)
    _recent_folders.insert(0, s)
    del _recent_folders[6:]


def _run_job(job: JobState, sha_to_frame: dict[str, Any]) -> None:
    """Worker thread: run the banger pipeline on job.input_dir, fill job.picks."""

    def log_line(msg: str) -> None:
        job.log_lines.append(msg)
        log.info("[%s] %s", job.job_id[:8], msg)

    try:
        opts = job.options
        cfg = settings_mod.load()
        recursive = bool(opts.get("recursive", True))
        top_n = int(opts.get("top_n", cfg["top_n"]))
        strategy = str(opts.get("strategy", cfg["strategy"]))
        diversity = float(opts.get("diversity", cfg["mmr_diversity"]))
        face_gate = bool(opts.get("face_gate", cfg["face_gate"]))
        eye_gate = bool(opts.get("eye_gate", cfg["eye_gate"]))
        write_xmp = bool(opts.get("write_xmp", False))

        job.stage = "discovering"
        frames = discover_frames(job.input_dir, recursive=recursive)
        job.total = len(frames)
        if not frames:
            job.status = "error"
            job.error = f"No supported images found in {job.input_dir}"
            return
        log_line(f"discovered {len(frames)} frames")

        threshold = float(cfg["sharpness_threshold"])
        face_sharp_threshold = float(cfg["face_sharpness_threshold"])
        head = taste_head.load()
        sc_clusters = scene_kmeans.load()
        if eye_gate and not eyes_mod.mediapipe_available():
            log_line("mediapipe not installed; --eye-gate disabled")
            eye_gate = False

        job.stage = "scoring"
        job.status = "running"
        candidates: list[tuple[Row, float, np.ndarray]] = []
        cache_hits = 0
        cold = 0
        face_gated = 0
        eye_gated = 0

        for idx, f in enumerate(frames):
            job.progress = idx
            sha = state.sha256_of(f.classify_path)
            sha_to_frame[sha] = f

            cached_meta = state.load_frame_metadata(sha)
            cached_emb = state.load_embedding(sha)

            face_id_ok = (strategy != "faces") or (
                cached_meta is not None
                and ("face_detections" in cached_meta or "face_embeddings" in cached_meta)
            )
            if cached_meta and "phash_hex" in cached_meta and cached_emb is not None and face_id_ok:
                sharp = float(cached_meta["sharpness"])
                if sharp < threshold:
                    cache_hits += 1
                    continue
                emb = cached_emb
                ts = float(cached_meta["timestamp"])
                phash = imagehash.hex_to_hash(cached_meta["phash_hex"])
                face_count = cached_meta.get("face_count", 0)
                face_sharp = cached_meta.get("face_sharpness", 0.0)
                frame_metrics = cached_meta.get("metrics")
                frame_eyes = cached_meta.get("eyes")
                cache_hits += 1
            else:
                try:
                    preview = load_preview(f.classify_path)
                except Exception as e:
                    log_line(f"skip {f.display_name}: {e}")
                    continue
                sharp = sharpness_from_preview(preview)
                if sharp < threshold:
                    cold += 1
                    continue
                try:
                    emb = aesthetic.encode_image(preview)
                    state.cache_embedding(sha, emb)
                    phash = dedup.phash_from_preview(preview)
                    ts = dedup.best_timestamp(f.classify_path)
                    face_count, face_sharp = face.best_face_sharpness(preview)
                    frame_metrics = None
                    try:
                        frame_metrics = metrics_mod.compute_all(preview)
                    except Exception as e:
                        log_line(f"metrics skip {f.display_name}: {e}")
                    frame_eyes = None
                    if eye_gate:
                        try:
                            frame_eyes = eyes_mod.analyse_eyes(preview)
                        except Exception as e:
                            log_line(f"eyes skip {f.display_name}: {e}")
                    face_dets_payload = None
                    if strategy == "faces":
                        try:
                            dets = face_id_mod.extract_face_detections(preview)
                            face_dets_payload = face_id_mod.encode_detections_for_cache(dets)
                        except Exception as e:
                            log_line(f"face_id skip {f.display_name}: {e}")
                    state.cache_frame_metadata(
                        sha, sharp, str(phash), ts,
                        face_count=face_count, face_sharpness=face_sharp,
                        metrics=frame_metrics, eyes=frame_eyes,
                        face_detections=face_dets_payload,
                    )
                except Exception as e:
                    log_line(f"score fail {f.display_name}: {e}")
                    continue
                cold += 1

            if face_gate and face_count > 0 and face_sharp < face_sharp_threshold:
                face_gated += 1
                continue
            if (
                eye_gate
                and frame_eyes is not None
                and frame_eyes.get("face_count", 0) > 0
                and frame_eyes.get("ear_min", 1.0) < eyes_mod.EYE_AR_THRESHOLD
            ):
                eye_gated += 1
                continue

            if head is not None:
                score = taste_head.predict_score(head, emb)
                source = "head"
            else:
                score, _ = aesthetic.score_from_embedding(emb)
                source = "prompts"

            scene_preset = None
            cluster_id = None
            if sc_clusters is not None:
                info = sc_clusters.classify_embedding(emb)
                scene_preset = info.preset
                cluster_id = info.cluster_id
            else:
                try:
                    scene_preset = scenes.classify(emb).preset
                except Exception:
                    scene_preset = None

            row = Row(
                frame=f, sharpness=sharp, aesthetic=score,
                aesthetic_breakdown=None, aesthetic_source=source,
                thumb_b64="", scene_preset=scene_preset,
                metrics=frame_metrics, eyes=frame_eyes,
            )
            candidates.append((row, score, emb))
            if cluster_id is not None:
                job.cluster_ids[f.display_name] = cluster_id

        job.progress = job.total

        if not candidates:
            job.status = "done"
            job.finished_at = time.monotonic()
            log_line("0 frames survived the gates")
            return

        job.stage = "selecting"
        if strategy == "topk":
            chosen = select.select_top_k(candidates, n=top_n)
        elif strategy == "mmr":
            chosen = select.select_diverse_top_n(candidates, n=top_n, diversity_lambda=diversity)
        elif strategy == "faces":
            face_embs_per_item: list[list[np.ndarray]] = []
            for r, _s, _e in candidates:
                sha2 = state.sha256_of(r.frame.classify_path)
                meta2 = state.load_frame_metadata(sha2) or {}
                payload = meta2.get("face_detections") or meta2.get("face_embeddings")
                face_embs_per_item.append(face_id_mod.decode_from_cache(payload))
            n_people_frames = sum(1 for embs in face_embs_per_item if embs)
            log_line(f"face-id: {n_people_frames}/{len(candidates)} candidates carry embeddings")
            chosen = select.select_faces_top_n(
                candidates, face_embs_per_item=face_embs_per_item, n=top_n,
            )
        else:
            chosen = select.select_kmeans_top_n(candidates, n=top_n)

        log_line(
            f"selected {len(chosen)}/{len(candidates)} via {strategy} "
            f"(cache_hits={cache_hits}, cold={cold}, face_gated={face_gated}, eye_gated={eye_gated})"
        )

        for rank, (row, _score, _emb) in enumerate(chosen, start=1):
            sha = state.sha256_of(row.frame.classify_path)
            job.picks.append({
                "rank": rank,
                "sha": sha,
                "stem": row.frame.stem,
                "subdir": row.frame.subdir,
                "display": row.frame.display_name,
                "kind": row.frame.kind,
                "sharpness": round(row.sharpness, 1),
                "aesthetic": round(row.aesthetic, 3) if row.aesthetic is not None else None,
                "aesthetic_source": row.aesthetic_source,
                "scene_preset": row.scene_preset,
                "src_path": str(row.frame.classify_path),
            })

        if write_xmp:
            job.stage = "writing xmp"
            top_rows = [row for row, _s, _e in chosen]
            n = xmp_mod.write_for_rows(top_rows, cluster_ids=job.cluster_ids or None)
            job.xmp_written = n
            log_line(f"wrote {n} XMP sidecars")

        job.status = "done"
        job.finished_at = time.monotonic()
        job.stage = "done"
        log_line(f"finished in {(job.finished_at - job.started_at):.1f}s")
    except Exception as e:
        log.exception("job %s crashed", job.job_id)
        job.status = "error"
        job.error = str(e)
        job.finished_at = time.monotonic()


# ---------------------------------------------------------------------------
# Flask app + pywebview launcher
# ---------------------------------------------------------------------------

PREVIEW_JPEG_QUALITY = 88


_EXIF_FIELDS = {
    # PIL tag id -> friendly key. Subset that's actually useful for "why did
    # this shot work?" questions. Skips boring stuff like ColorSpace, YCbCrPositioning.
    271: "camera_make",
    272: "camera_model",
    42036: "lens_model",
    33434: "exposure_time",       # rational, seconds
    33437: "f_number",            # rational
    34855: "iso",
    37386: "focal_length",        # rational, mm
    41989: "focal_length_35mm",
    36867: "date_taken",
    37380: "exposure_bias",       # rational, stops
    37383: "metering_mode",
    37384: "light_source",
    37385: "flash",
    41986: "exposure_mode",
    41987: "white_balance",
    41988: "digital_zoom",
    41990: "scene_capture_type",
    34850: "exposure_program",
    40962: "pixel_x",
    40963: "pixel_y",
}


def _rational_to_float(v) -> float | None:
    try:
        if hasattr(v, "numerator") and hasattr(v, "denominator"):
            return v.numerator / v.denominator if v.denominator else None
        if isinstance(v, tuple) and len(v) == 2:
            return v[0] / v[1] if v[1] else None
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _format_exposure(secs: float | None) -> str | None:
    if secs is None:
        return None
    if secs >= 1:
        return f"{secs:.1f}s"
    if secs <= 0:
        return None
    return f"1/{round(1.0 / secs)}s"


_METERING_MODES = {0: "unknown", 1: "average", 2: "centre-weighted", 3: "spot",
                   4: "multi-spot", 5: "matrix", 6: "partial"}
def _FLASH_FIRED(v):
    return "fired" if (isinstance(v, int) and v & 1) else "no flash"


def _render_gallery_html(title: str, items: list[dict]) -> str:
    """Self-contained HTML gallery — thumbnails embedded as base64 so the
    file is portable / shareable on its own."""
    import html as _html

    def esc(s) -> str:
        return _html.escape(str(s)) if s is not None else ""

    def fmt_exif(exif: dict) -> str:
        if not exif:
            return ""
        bits = []
        cam = " ".join(filter(None, [exif.get("camera_make"), exif.get("camera_model")])).strip()
        if cam:
            bits.append(esc(cam))
        if exif.get("lens_model"):
            bits.append(esc(exif["lens_model"]))
        details = []
        if exif.get("focal_length"):
            details.append(f"{exif['focal_length']:.0f}mm")
        if exif.get("f_number"):
            details.append(f"f/{exif['f_number']:.1f}")
        if exif.get("shutter"):
            details.append(exif["shutter"])
        if exif.get("iso"):
            details.append(f"ISO {exif['iso']}")
        if details:
            bits.append(esc(" · ".join(details)))
        if exif.get("date_taken"):
            bits.append(esc(exif["date_taken"]))
        return "<br>".join(bits)

    cards = []
    for m in items:
        tags_html = " ".join(f"<span class='tag'>{esc(t)}</span>" for t in m.get("tags") or [])
        faces = m.get("faces") or []
        face_chips = "".join(f"<span class='face'>{esc(n)}</span>" for n in faces)
        face_note = ""
        if m.get("face_count") and not faces:
            face_note = f"<span class='face muted'>{m['face_count']} unnamed face{'s' if m['face_count'] > 1 else ''}</span>"
        aesthetic = m.get("aesthetic")
        aest_str = f"{aesthetic:+.2f}" if aesthetic is not None else "—"
        cards.append(f"""
<article class='card'>
  <div class='thumb'><img src='data:image/jpeg;base64,{m.get("thumb_b64", "")}' alt='{esc(m["source"])}'></div>
  <div class='body'>
    <header>
      <span class='rank'>#{m["rank"]:02d}</span>
      <span class='name'>{esc(m["source"])}</span>
    </header>
    <div class='stats'>
      <span title='Laplacian variance'>sharp <b>{m.get("sharpness", 0):.0f}</b></span>
      <span title='Aesthetic score ({esc(m.get("aesthetic_source") or "n/a")})'>aesthetic <b>{aest_str}</b></span>
    </div>
    <div class='tags'>{tags_html}</div>
    <div class='faces'>{face_chips}{face_note}</div>
    <div class='exif'>{fmt_exif(m.get("exif") or {})}</div>
  </div>
</article>
""")

    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'>
<title>{esc(title)} — banger gallery</title>
<style>
  :root {{ color-scheme: dark; --bg:#0c0c0c; --bg2:#161616; --bg3:#1f1f1f; --fg:#e6e6e6; --dim:#888; --line:#2a2a2a; --accent:#9be37b; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--fg); padding: 1.5rem 2rem 3rem; }}
  h1 {{ font-weight: 400; font-size: 1.3rem; margin: 0 0 .3rem; }}
  .meta {{ color: var(--dim); font-size: .8rem; margin-bottom: 1.4rem; }}
  .grid {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); }}
  .card {{ background: var(--bg2); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; display: flex; flex-direction: column; }}
  .thumb {{ background: #000; aspect-ratio: 4 / 3; overflow: hidden; }}
  .thumb img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
  .body {{ padding: .7rem .9rem 1rem; display: flex; flex-direction: column; gap: .5rem; min-height: 0; }}
  header {{ display: flex; gap: .6rem; align-items: baseline; }}
  .rank {{ font-variant-numeric: tabular-nums; color: var(--accent); font-weight: 600; }}
  .name {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .82rem; color: var(--fg); }}
  .stats {{ display: flex; gap: 1rem; font-size: .78rem; color: var(--dim); }}
  .stats b {{ color: var(--fg); font-weight: 600; font-variant-numeric: tabular-nums; }}
  .tags {{ display: flex; flex-wrap: wrap; gap: .25rem; }}
  .tag {{ background: var(--bg3); color: var(--fg); border: 1px solid var(--line); border-radius: 10px; padding: .1rem .55rem; font-size: .7rem; }}
  .faces {{ display: flex; flex-wrap: wrap; gap: .25rem; }}
  .face {{ background: rgba(155, 227, 123, .12); color: var(--accent); border: 1px solid rgba(155, 227, 123, .3); border-radius: 10px; padding: .1rem .55rem; font-size: .7rem; }}
  .face.muted {{ background: var(--bg3); color: var(--dim); border-color: var(--line); }}
  .exif {{ color: var(--dim); font-size: .73rem; line-height: 1.5; }}
</style></head>
<body>
<h1>{esc(title)}</h1>
<div class='meta'>{len(items)} frames · originals are in this same folder · generated by banger</div>
<div class='grid'>
{"".join(cards)}
</div>
</body></html>
"""


def _extract_exif(path: Path) -> dict:
    """Return a flat dict of friendly-named EXIF fields, or {} on any error."""
    if path.suffix.lower() not in (".jpg", ".jpeg"):
        return {}
    try:
        from PIL import Image
        with Image.open(path) as im:
            top = im.getexif() or {}
            # ExifIFD (0x8769) holds the camera-settings tags (aperture, shutter,
            # ISO, etc). PIL's top-level getexif only returns the main IFD, so
            # we have to fetch the sub-IFD explicitly and merge.
            try:
                sub = top.get_ifd(0x8769) or {}
            except Exception:
                sub = {}
            raw = {**dict(top), **dict(sub)}
    except Exception:
        return {}

    out: dict = {}
    for tag_id, key in _EXIF_FIELDS.items():
        if tag_id not in raw:
            continue
        v = raw[tag_id]
        if key in ("exposure_time", "f_number", "focal_length", "exposure_bias"):
            v = _rational_to_float(v)
        elif key == "metering_mode" and isinstance(v, int):
            v = _METERING_MODES.get(v, str(v))
        elif key == "flash" and isinstance(v, int):
            v = _FLASH_FIRED(v)
        elif isinstance(v, bytes):
            try:
                v = v.decode("utf-8", errors="replace").strip("\x00")
            except Exception:
                continue
        out[key] = v

    # Pretty-format the ones that read better as strings.
    if out.get("exposure_time") is not None:
        out["shutter"] = _format_exposure(out["exposure_time"])
    if out.get("f_number") is not None:
        out["aperture"] = f"f/{out['f_number']:.1f}"
    if out.get("focal_length") is not None:
        out["focal_length_mm"] = f"{out['focal_length']:.0f}mm"

    # PIL's IFDRational and Pillow byte-string surrogates aren't JSON safe.
    # Convert anything that survived to a primitive, drop the rest.
    safe: dict = {}
    for k, v in out.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe[k] = v
        else:
            f = _rational_to_float(v)
            if f is not None:
                safe[k] = f
            else:
                try:
                    safe[k] = str(v)
                except Exception:
                    continue
    return safe


def _subfolder_for(path: Path, scheme: str) -> str:
    """Compute the destination subfolder for an import file.

    by_date: YYYY-MM-DD pulled from EXIF DateTimeOriginal (JPEG) or file mtime.
    by_camera: camera model slug (slashes stripped), or "unknown".
    flat: empty string (everything goes in the destination root).
    """
    if scheme == "flat":
        return ""
    if scheme == "by_camera":
        if path.suffix.lower() in (".jpg", ".jpeg"):
            try:
                from PIL import Image
                with Image.open(path) as im:
                    raw = im.getexif() or {}
                model = raw.get(272)
                if isinstance(model, bytes):
                    model = model.decode("utf-8", errors="replace").strip("\x00")
                if isinstance(model, str) and model.strip():
                    return model.strip().replace("/", "_").replace("\\", "_")
            except Exception:
                pass
        return "unknown"
    # default: by_date
    if path.suffix.lower() in (".jpg", ".jpeg"):
        try:
            from PIL import Image
            with Image.open(path) as im:
                raw = im.getexif() or {}
            dt = raw.get(36867)
            if isinstance(dt, str):
                from datetime import datetime as _dt
                d = _dt.strptime(dt, "%Y:%m:%d %H:%M:%S")
                return d.strftime("%Y-%m-%d")
        except Exception:
            pass
    # Fallback to mtime.
    try:
        from datetime import datetime as _dt
        return _dt.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
    except OSError:
        return "unknown-date"


def _encode_preview_jpeg(preview_bgr) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", preview_bgr, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def build_app(window_holder: dict | None = None) -> Flask:
    """Construct the Flask app. `window_holder['win']` (when set) gives access
    to the pywebview window so the folder-pick endpoint can show a native
    dialog instead of falling back to a path-text input."""

    app = Flask(__name__)
    sha_to_frame: dict[str, Any] = {}

    def _resolve_path(sha: str) -> Path | None:
        """Find the source path for a sha. Prefers the in-memory map (frames
        from the current scoring job); falls back to the library index, which
        knows about every photo in every watched root. This is why a thumbnail
        click in the Library tab can reach a file the scoring pipeline has
        never touched."""
        f = sha_to_frame.get(sha)
        if f is not None:
            return f.classify_path
        return library.lookup_path(sha)

    def _resolve_frame(sha: str):
        """Like _resolve_path but returns a Frame-shaped object so the existing
        sha_to_frame consumers (which expect .classify_path) keep working
        regardless of which side the sha came from."""
        if sha in sha_to_frame:
            return sha_to_frame[sha]
        path = library.lookup_path(sha)
        if path is None:
            return None
        # Lazily synthesise a minimal frame-like object. We don't have a
        # banger.frames.Frame in the library row (no separate jpeg/raw pair),
        # but classify_path is all most consumers actually use.
        from banger.frames import Frame
        from banger.preview import JPEG_SUFFIXES
        is_jpeg = path.suffix in JPEG_SUFFIXES
        return Frame(
            stem=path.stem, subdir="",
            jpeg=path if is_jpeg else None,
            raw=None if is_jpeg else path,
        )

    @app.route("/")
    @app.route("/gui")
    def root():
        # WebView2 happily caches the HTML across launches, which means
        # "I shipped a fix" can look like "fix didn't land" to the user.
        # Force a no-store policy on the SPA shell so every launch fetches
        # fresh markup + inline CSS / JS.
        from flask import make_response
        resp = make_response(render_template_string(_SPA_TEMPLATE))
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    @app.route("/api/settings", methods=["GET"])
    def settings_get():
        return jsonify({
            "values": settings_mod.load(),
            "defaults": settings_mod.DEFAULTS,
            "fields": settings_mod.FIELD_META,
        })

    @app.route("/api/settings", methods=["POST"])
    def settings_save():
        data = request.get_json(silent=True) or {}
        if data.get("reset"):
            return jsonify({"values": settings_mod.reset()})
        return jsonify({"values": settings_mod.save(data.get("updates") or {})})

    @app.route("/api/cull", methods=["POST"])
    def cull():
        """Run selection from cached metadata only — no preview reads, no
        re-tagging. Returns picks for a given list of SHAs.

        Body: {shas: [...], top_n, strategy, diversity, bias_tags: [...],
               bias_strength: 0.0..1.0}

        If any sha lacks a cached embedding, returns
        {needs_indexing: true, missing: N} so the caller can fall back to
        /api/run for a full pass. Bias is applied additively to the
        normalised aesthetic score: matching tags push a frame upward but
        don't filter out anything.
        """
        from banger.report import Row as _Row

        data = request.get_json(silent=True) or {}
        shas = data.get("shas") or []
        if not shas:
            return jsonify({"error": "shas list required"}), 400
        cfg = settings_mod.load()
        top_n = int(data.get("top_n", cfg["top_n"]))
        strategy = str(data.get("strategy", cfg["strategy"]))
        diversity = float(data.get("diversity", cfg["mmr_diversity"]))
        bias_tags = [str(t).lower() for t in (data.get("bias_tags") or [])]
        bias_strength = float(data.get("bias_strength", 0.3))
        threshold = float(cfg["sharpness_threshold"])

        head = taste_head.load()
        candidates: list[tuple[Row, float, np.ndarray]] = []
        missing = 0
        face_embs_for_strategy: list[list[np.ndarray]] = []
        for sha in shas:
            emb = state.load_embedding(sha)
            meta = state.load_frame_metadata(sha)
            if emb is None or meta is None or "sharpness" not in meta:
                missing += 1
                continue
            sharp = float(meta["sharpness"])
            if sharp < threshold:
                continue
            frame = _resolve_frame(sha)
            if frame is None:
                continue
            if head is not None:
                try:
                    score = float(head.predict(emb.reshape(1, -1))[0])
                    source = "taste"
                except Exception:
                    score, _ = aesthetic.score_from_embedding(emb)
                    source = "aesthetic"
            else:
                score, _ = aesthetic.score_from_embedding(emb)
                source = "aesthetic"

            if bias_tags:
                tags = [t.lower() for t, _ in (meta.get("tags") or [])]
                if any(bt in tags for bt in bias_tags):
                    score = score + bias_strength

            row = _Row(
                frame=frame, sharpness=sharp, aesthetic=score,
                aesthetic_breakdown=None, aesthetic_source=source,
                thumb_b64="", scene_preset=None,
                metrics=meta.get("metrics"), eyes=meta.get("eyes"),
            )
            candidates.append((row, score, emb))
            if strategy == "faces":
                payload = meta.get("face_detections") or meta.get("face_embeddings")
                face_embs_for_strategy.append(face_id_mod.decode_from_cache(payload))

        if missing and missing > len(shas) * 0.05:
            return jsonify({"needs_indexing": True, "missing": missing, "total": len(shas)})

        if not candidates:
            return jsonify({"picks": [], "considered": len(shas), "missing": missing})

        if strategy == "topk":
            chosen = select.select_top_k(candidates, n=top_n)
        elif strategy == "mmr":
            chosen = select.select_diverse_top_n(candidates, n=top_n, diversity_lambda=diversity)
        elif strategy == "faces":
            chosen = select.select_faces_top_n(
                candidates, face_embs_per_item=face_embs_for_strategy, n=top_n,
            )
        else:
            chosen = select.select_kmeans_top_n(candidates, n=top_n)

        picks = []
        for rank, (row, _s, _e) in enumerate(chosen, start=1):
            sha = state.sha256_of(row.frame.classify_path)
            picks.append({
                "rank": rank, "sha": sha,
                "stem": row.frame.stem,
                "sharpness": round(row.sharpness, 1),
                "aesthetic": round(row.aesthetic, 3),
                "aesthetic_source": row.aesthetic_source,
            })
        return jsonify({"picks": picks, "considered": len(candidates), "missing": missing})

    @app.route("/api/export-bangers", methods=["POST"])
    def export_bangers():
        """Copy the given SHAs' originals into ~/Pictures/bangers/<ts>[_label]/.

        Body: {shas: [...], output_root: str | None, label: str | None}
        Also writes a manifest.json and a self-contained gallery.html
        (thumbnails embedded as base64) so the folder is shareable as-is.
        """
        import base64
        import datetime as _dt

        data = request.get_json(silent=True) or {}
        shas = data.get("shas") or []
        if not shas:
            return jsonify({"error": "shas list required"}), 400
        label_hint = (data.get("label") or "").strip()

        root_raw = (data.get("output_root") or "").strip()
        if root_raw:
            root = Path(root_raw).expanduser()
        else:
            root = Path.home() / "Pictures" / "bangers"
        ts = _dt.datetime.now().strftime("%Y-%m-%d_%H%M")
        if label_hint:
            ts = f"{ts}_{label_hint}"
        folder = root / ts

        # Resolve sources first so we can preflight free space against the
        # summed source size and reject up front rather than dying mid-copy
        # with a half-populated folder. We also remember which sha "owns" each
        # source path so atomic_copy can verify the copy against the known
        # library hash (siblings have their own hashes we don't know, so they
        # copy unverified).
        errors: list[str] = []
        # plan: list of (rank, sha, [(src_path, expected_sha|None), ...])
        plan: list[tuple[int, str, list[tuple[Path, str | None]]]] = []
        needed_bytes = 0
        for rank, sha in enumerate(shas, start=1):
            src = _resolve_path(sha)
            if src is None:
                errors.append(f"unknown sha: {sha}")
                continue
            siblings: list[tuple[Path, str | None]] = [(src, sha)]
            seen = {src}
            for suffix in (".ARW", ".CR2", ".CR3", ".NEF", ".RAF", ".DNG",
                           ".RW2", ".ORF", ".PEF", ".JPG", ".JPEG"):
                sib = src.with_suffix(suffix)
                if sib != src and sib.exists() and sib not in seen:
                    siblings.append((sib, None))
                    seen.add(sib)
            for sp, _exp in siblings:
                try:
                    needed_bytes += sp.stat().st_size
                except OSError:
                    pass
            plan.append((rank, sha, siblings))

        if not plan:
            return jsonify({"error": "no resolvable sources", "errors": errors[:10]}), 400

        folder.mkdir(parents=True, exist_ok=True)
        # Preflight: refuse to start if the destination can't hold the originals
        # (plus a little headroom for the manifest + gallery thumbnails).
        if not fsutil.has_free_space(folder, needed_bytes + (8 << 20)):
            return jsonify({
                "error": "insufficient free space",
                "folder": str(folder),
                "needed_bytes": needed_bytes,
            }), 507

        copied = 0
        manifest: list[dict] = []
        head = taste_head.load()

        for rank, sha, siblings in plan:
            src = siblings[0][0]
            copied_names = []
            for s, expected_sha in siblings:
                dst = folder / s.name
                # Only treat a pre-existing dst as done if it's byte-for-byte
                # the same size; a truncated/partial leftover must be re-copied.
                if dst.exists():
                    try:
                        if dst.stat().st_size == s.stat().st_size:
                            copied_names.append(s.name)
                            continue
                    except OSError:
                        pass
                try:
                    fsutil.atomic_copy(s, dst, expected_sha=expected_sha)
                    copied += 1
                    copied_names.append(s.name)
                except (OSError, ValueError) as e:
                    errors.append(f"copy {s.name}: {e}")

            # Gather stats for the gallery.
            meta = state.load_frame_metadata(sha) or {}
            emb = state.load_embedding(sha)
            aesthetic_score = None
            aesthetic_source = None
            if emb is not None:
                try:
                    if head is not None:
                        aesthetic_score = float(head.predict(emb.reshape(1, -1))[0])
                        aesthetic_source = "taste"
                    else:
                        aesthetic_score, _ = aesthetic.score_from_embedding(emb)
                        aesthetic_source = "aesthetic"
                except Exception:
                    pass

            exif = _extract_exif(src)
            tags = [t for t, _s in (meta.get("tags") or [])][:8]
            face_count = int(meta.get("face_count") or 0)
            face_names_list: list[str] = []
            for det in (meta.get("face_detections") or []):
                emb_arr = np.asarray(det.get("embedding") or [], dtype=np.float32)
                if emb_arr.size == face_names.EMB_DIM:
                    nm, _sim = face_names.match(emb_arr)
                    if nm:
                        face_names_list.append(nm)

            thumb_b64 = ""
            tpath = state.thumbnail_path(sha)
            if tpath.exists():
                try:
                    thumb_b64 = base64.b64encode(tpath.read_bytes()).decode("ascii")
                except OSError:
                    pass
            if not thumb_b64:
                try:
                    preview = load_preview(src)
                    thumb_b64 = base64.b64encode(encode_thumbnail_bytes(preview)).decode("ascii")
                except Exception as e:
                    log.warning("gallery thumb fail %s: %s", src.name, e)

            manifest.append({
                "rank": rank, "sha": sha, "source": src.name,
                "files": copied_names,
                "sharpness": round(float(meta.get("sharpness") or 0.0), 1),
                "aesthetic": round(aesthetic_score, 3) if aesthetic_score is not None else None,
                "aesthetic_source": aesthetic_source,
                "tags": tags,
                "face_count": face_count,
                "faces": sorted(set(face_names_list)),
                "exif": exif,
                "thumb_b64": thumb_b64,
            })

        try:
            import json as _json
            # The manifest on disk doesn't need the giant thumb_b64 blobs.
            slim = [{k: v for k, v in m.items() if k != "thumb_b64"} for m in manifest]
            (folder / "manifest.json").write_text(
                _json.dumps({
                    "exported_at": int(time.time()),
                    "count": len(manifest),
                    "items": slim,
                }, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

        # Render the gallery.
        try:
            html_path = folder / "gallery.html"
            html_path.write_text(_render_gallery_html(folder.name, manifest), encoding="utf-8")
        except OSError as e:
            errors.append(f"gallery write: {e}")

        # A non-empty errors list means the export is partial (some originals
        # failed to copy or verify). Surface that as a 207-ish result the
        # client can flag, NOT a bare 200 "success".
        body = {
            "folder": str(folder),
            "copied": copied,
            "errors": errors[:10],
            "error_count": len(errors),
            "partial": bool(errors),
        }
        return jsonify(body), (207 if errors else 200)

    @app.route("/api/xmp-writeback", methods=["POST"])
    def xmp_writeback():
        """Write XMP sidecars (ratings + colour labels + subjects) next to the
        ORIGINAL files for the given scope, leaving the source pixels untouched.

        Body: {shas: [...]}. This is the local-first "annotate the catalog you
        already own" action — previously XMP was only reachable from the legacy
        Quick-run path. We rank the frames by their effective score (taste head
        if trained, else aesthetic, else sharpness fallback) so stars_from_rank
        matches the Export ranking, then delegate to xmp.write_for_rows.
        """
        from banger.report import Row as _Row

        data = request.get_json(silent=True) or {}
        shas = data.get("shas") or []
        if not shas:
            return jsonify({"error": "shas list required"}), 400

        head = taste_head.load()
        rows: list = []
        errors: list[str] = []
        for sha in shas:
            f = _resolve_frame(sha)
            if f is None:
                errors.append(f"unknown sha: {sha}")
                continue
            meta = state.load_frame_metadata(sha) or {}
            emb = state.load_embedding(sha)
            aesthetic_score = None
            aesthetic_source = None
            if emb is not None:
                try:
                    if head is not None:
                        aesthetic_score = float(head.predict(emb.reshape(1, -1))[0])
                        aesthetic_source = "head"
                    else:
                        aesthetic_score, _ = aesthetic.score_from_embedding(emb)
                        aesthetic_source = "prompts"
                except Exception as e:
                    log.warning("xmp score fail %s: %s", sha, e)
            sharp = float(meta.get("sharpness") or 0.0)
            rows.append(_Row(
                frame=f,
                sharpness=sharp,
                aesthetic=aesthetic_score,
                aesthetic_breakdown=None,
                aesthetic_source=aesthetic_source,
                thumb_b64="",
                scene_preset=(meta.get("scene_preset") or None),
                metrics=meta.get("metrics"),
                eyes=meta.get("eyes"),
            ))

        written = 0
        if rows:
            try:
                written = xmp_mod.write_for_rows(rows)
            except Exception as e:
                log.warning("xmp writeback failed: %s", e)
                errors.append(f"xmp write: {e}")

        body = {
            "written": written,
            "considered": len(rows),
            "errors": errors[:10],
            "error_count": len(errors),
            "partial": bool(errors),
        }
        return jsonify(body), (207 if errors else 200)

    @app.route("/api/import", methods=["POST"])
    def import_files():
        """Copy photos from a source directory into a destination, organised
        by date / camera / flat, then add the destination to the library and
        kick a scan.

        For now this runs synchronously since most imports are small. A
        background-job version with progress would come if the typical
        import grows past ~1k files.
        """
        data = request.get_json(silent=True) or {}
        src_raw = (data.get("source") or "").strip()
        dst_raw = (data.get("destination") or "").strip()
        scheme = (data.get("scheme") or "by_date").strip()
        if not src_raw or not dst_raw:
            return jsonify({"error": "source and destination required"}), 400
        src = Path(src_raw).expanduser()
        dst = Path(dst_raw).expanduser()
        if not src.is_dir():
            return jsonify({"error": f"not a folder: {src}"}), 400
        dst.mkdir(parents=True, exist_ok=True)

        from banger.frames import discover_frames as _discover
        frames = _discover(src, recursive=True)

        # Preflight free space against the summed source sizes (over-estimates
        # slightly because of files we'll skip as already-present, which is the
        # safe direction).
        needed_bytes = 0
        for f in frames:
            for p in (f.classify_path, f.raw, f.jpeg):
                if p is None:
                    continue
                try:
                    needed_bytes += p.stat().st_size
                except OSError:
                    pass
        if not fsutil.has_free_space(dst, needed_bytes):
            return jsonify({
                "error": "insufficient free space at destination",
                "destination": str(dst),
                "needed_bytes": needed_bytes,
            }), 507

        copied = 0
        skipped = 0
        errors: list[str] = []
        for f in frames:
            srcp = f.classify_path
            subfolder = _subfolder_for(srcp, scheme)
            target_dir = dst / subfolder if subfolder else dst
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / srcp.name
            if target.exists() and target.stat().st_size == srcp.stat().st_size:
                skipped += 1
                continue
            try:
                fsutil.atomic_copy(srcp, target)
                # Also copy the sibling RAW or JPEG if present.
                other = f.raw if f.classify_path == f.jpeg else f.jpeg
                if other is not None and other != srcp and other.exists():
                    osib = target_dir / other.name
                    if not (osib.exists() and osib.stat().st_size == other.stat().st_size):
                        fsutil.atomic_copy(other, osib)
                copied += 1
            except (OSError, ValueError) as e:
                errors.append(f"copy {srcp.name}: {e}")

        root = library.add_root(dst)
        # Kick a scan (synchronous so the response reflects the new frames).
        progress = library.ScanProgress(
            root_id=root.id, root_path=root.path, started_at=time.monotonic()
        )
        with _scan_lock:
            _scan_progress[root.id] = progress
        library.scan_root(root.id, progress=progress)
        body = {
            "copied": copied,
            "skipped": skipped,
            "root_id": root.id,
            "scan": progress.to_view(),
            "errors": errors[:10],
            "error_count": len(errors),
            "partial": bool(errors),
        }
        return jsonify(body), (207 if errors else 200)

    @app.route("/api/library/roots", methods=["GET"])
    def library_roots():
        return jsonify({
            "roots": [
                {**r.__dict__, "frame_count": library.count_frames(r.id)}
                for r in library.all_roots()
            ],
        })

    @app.route("/api/library/roots", methods=["POST"])
    def library_add_root():
        data = request.get_json(silent=True) or {}
        path = (data.get("path") or "").strip()
        label = (data.get("label") or "").strip() or None
        if not path:
            return jsonify({"error": "path required"}), 400
        try:
            root = library.add_root(Path(path), label=label)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(root.__dict__)

    @app.route("/api/library/roots/<int:root_id>", methods=["DELETE"])
    def library_remove_root(root_id):
        if library.remove_root(root_id):
            return jsonify({"removed": root_id})
        return jsonify({"error": "unknown root"}), 404

    @app.route("/api/library/scan/<int:root_id>", methods=["POST"])
    def library_scan(root_id):
        root = library.get_root(root_id)
        if root is None:
            return jsonify({"error": "unknown root"}), 404
        with _scan_lock:
            existing = _scan_progress.get(root_id)
            if existing is not None and existing.phase not in ("done", "error"):
                return jsonify({"status": "already scanning", **existing.to_view()}), 200
            progress = library.ScanProgress(
                root_id=root_id, root_path=root.path, started_at=time.monotonic()
            )
            _scan_progress[root_id] = progress

        def _go():
            try:
                library.scan_root(root_id, progress=progress)
            except Exception as e:
                log.exception("scan failed: %s", e)
            # Kick the tagger so newly-indexed frames get embeddings + tags
            # without the user clicking anything. Idempotent: re-runs on
            # already-tagged frames are quick skips.
            try:
                tagger_mod.start()
            except Exception as e:
                log.warning("tagger auto-start failed: %s", e)
        threading.Thread(target=_go, daemon=True).start()
        return jsonify({"status": "started", **progress.to_view()})

    @app.route("/api/tagger/start", methods=["POST"])
    def tagger_start():
        data = request.get_json(silent=True) or {}
        full = bool(data.get("full"))
        return jsonify(tagger_mod.start(full=full).to_view())

    @app.route("/api/tagger/status")
    def tagger_status():
        return jsonify(tagger_mod.status().to_view())

    @app.route("/api/library/scan-status/<int:root_id>")
    def library_scan_status(root_id):
        with _scan_lock:
            progress = _scan_progress.get(root_id)
        if progress is None:
            return jsonify({"phase": "idle"})
        return jsonify(progress.to_view())

    @app.route("/api/library/cameras")
    def library_cameras():
        return jsonify({
            "cameras": [{"model": m, "count": c} for m, c in library.cameras()],
        })

    @app.route("/api/library/places")
    def library_places():
        return jsonify({
            "places": [{"name": p, "count": c} for p, c in library.places()],
        })

    @app.route("/api/library/geocode-backfill", methods=["POST"])
    def library_geocode_backfill():
        """Backfill place names for frames indexed before reverse_geocoder was
        installed. Cheap, runs once on demand."""
        n = library.reverse_geocode_unreferenced()
        return jsonify({"updated": n})

    @app.route("/api/library/folders/<int:root_id>")
    def library_folders(root_id):
        return jsonify({
            "folders": [{"name": n or "(root)", "count": c}
                        for n, c in library.folder_tree(root_id)],
        })

    # SQL pages are pulled this many rows at a time when a Python-side filter is
    # active, so face/place/text/rating filters scan the WHOLE matching set
    # rather than just the first page the client asked for.
    _SQL_PAGE = 2000

    @app.route("/api/library/frames")
    def library_frames():
        # All filters are optional; missing = no filter.
        root_id = request.args.get("root_id", type=int)
        subdir = request.args.get("subdir") or None
        camera = request.args.get("camera") or None
        after = request.args.get("after", type=int)
        before = request.args.get("before", type=int)
        limit = request.args.get("limit", type=int, default=500)
        offset = request.args.get("offset", type=int, default=0)
        face_name = request.args.get("face") or None
        min_score = request.args.get("min_score", type=int)
        q = (request.args.get("q") or "").strip().lower() or None
        place_filter = (request.args.get("place") or "").strip() or None

        # Python-side filters live outside the SQL DB (labels.db + metadata/*.json),
        # so the library can't paginate them. If ANY is active we must scan the
        # full SQL-matching set to filter correctly and report an honest total.
        py_filter_active = (
            face_name is not None or min_score is not None
            or q is not None or place_filter is not None
        )

        # Labels are read ONCE per request (a single dict) instead of a SQLite
        # round-trip per row.
        labels_map = state.labels_dict()
        # Cache metadata per sha so we hash/parse each frame's JSON at most once
        # per request even though filtering and view-building both consult it.
        meta_cache: dict[str, dict] = {}

        def _meta(sha: str) -> dict:
            m = meta_cache.get(sha)
            if m is None:
                m = state.load_frame_metadata(sha) or {}
                meta_cache[sha] = m
            return m

        def _passes(r) -> bool:
            if place_filter is not None and r.place_city != place_filter:
                return False
            if min_score is not None:
                label = labels_map.get(r.sha)
                if label is None or label < min_score:
                    return False
            if face_name is not None:
                detections = _meta(r.sha).get("face_detections") or []
                matched_names = set()
                for d in detections:
                    emb = np.asarray(d.get("embedding") or [], dtype=np.float32)
                    if emb.size == face_names.EMB_DIM:
                        n, _sim = face_names.match(emb)
                        if n:
                            matched_names.add(n)
                if face_name not in matched_names:
                    return False
            if q is not None:
                # Substring match over stem / rel_path / camera / place / tags.
                blob_parts = [r.stem.lower(), r.rel_path.lower()]
                if r.camera_model:
                    blob_parts.append(r.camera_model.lower())
                for place in (r.place_city, r.place_region, r.place_country):
                    if place:
                        blob_parts.append(place.lower())
                for t in (_meta(r.sha).get("tags") or []):
                    if isinstance(t, (list, tuple)) and t:
                        blob_parts.append(str(t[0]).lower())
                if q not in " | ".join(blob_parts):
                    return False
            return True

        def _view(r) -> dict:
            meta = _meta(r.sha)
            return {
                **r.to_view(),
                "label": labels_map.get(r.sha),
                "tags": meta.get("tags"),
                "has_face_data": bool(meta.get("face_detections")),
                "scored": "metrics" in meta,
            }

        if not py_filter_active:
            # Fast path: SQL does ALL the filtering, so we can paginate directly
            # and count without scanning every row.
            rows = library.query_frames(
                root_id=root_id, subdir=subdir, camera=camera,
                after=after, before=before, limit=limit, offset=offset,
            )
            # Accurate total for the SQL-expressible filter set. count_frames
            # only honours root_id, so when subdir/camera/date narrow further we
            # fall back to counting via a wide query (sha-only, no metadata).
            if subdir is None and camera is None and after is None and before is None:
                total = library.count_frames(root_id)
            else:
                total = 0
                page_off = 0
                while True:
                    page = library.query_frames(
                        root_id=root_id, subdir=subdir, camera=camera,
                        after=after, before=before, limit=_SQL_PAGE, offset=page_off,
                    )
                    if not page:
                        break
                    total += len(page)
                    if len(page) < _SQL_PAGE:
                        break
                    page_off += _SQL_PAGE
            out = [_view(r) for r in rows]
            return jsonify({
                "frames": out, "total": total,
                "offset": offset, "limit": limit,
                "returned": len(out),
                "has_more": offset + len(out) < total,
            })

        # Slow (but correct) path: walk the full SQL set in pages, apply the
        # Python filters, and slice the requested window out of the matches.
        # We only build views for the rows in the returned window.
        matched: list = []
        page_off = 0
        while True:
            page = library.query_frames(
                root_id=root_id, subdir=subdir, camera=camera,
                after=after, before=before, limit=_SQL_PAGE, offset=page_off,
            )
            if not page:
                break
            for r in page:
                if _passes(r):
                    matched.append(r)
            if len(page) < _SQL_PAGE:
                break
            page_off += _SQL_PAGE

        total = len(matched)
        window = matched[offset:offset + limit] if limit else matched[offset:]
        out = [_view(r) for r in window]
        return jsonify({
            "frames": out, "total": total,
            "offset": offset, "limit": limit,
            "returned": len(out),
            "has_more": offset + len(out) < total,
        })

    @app.route("/api/recent-folders")
    def recent_folders():
        return jsonify({"folders": list(_recent_folders)})

    @app.route("/api/folder-pick", methods=["POST"])
    def folder_pick():
        """Show the native folder picker via pywebview. Falls back to None when
        invoked from a browser tab rather than the pywebview shell."""
        win = (window_holder or {}).get("win")
        if win is None:
            return jsonify({"path": None, "reason": "no_native_window"}), 200
        try:
            import webview

            result = win.create_file_dialog(webview.FOLDER_DIALOG)
        except Exception as e:  # pragma: no cover (UI flake)
            return jsonify({"path": None, "reason": str(e)}), 200
        if not result:
            return jsonify({"path": None, "reason": "cancelled"})
        path = result[0] if isinstance(result, (list, tuple)) else result
        return jsonify({"path": str(path)})

    @app.route("/api/run", methods=["POST"])
    def run_pipeline():
        data = request.get_json(silent=True) or {}
        raw = data.get("input_dir") or ""
        input_dir = Path(raw).expanduser()
        if not input_dir.is_dir():
            return jsonify({"error": f"Not a folder: {input_dir}"}), 400
        _record_folder(input_dir)

        key = str(input_dir.resolve())
        with _jobs_lock:
            existing_id = _active_runs.get(key)
            if existing_id is not None:
                existing = _jobs.get(existing_id)
                if existing is not None and existing.status in ("running", "queued"):
                    return jsonify({"job_id": existing_id, "reused": True})
            job = JobState(job_id=str(uuid.uuid4()), input_dir=input_dir, options=data)
            _jobs[job.job_id] = job
            _active_runs[key] = job.job_id

        def _runner():
            try:
                _run_job(job, sha_to_frame)
            finally:
                with _jobs_lock:
                    if _active_runs.get(key) == job.job_id:
                        del _active_runs[key]

        threading.Thread(target=_runner, daemon=True).start()
        return jsonify({"job_id": job.job_id})

    @app.route("/api/jobs/<job_id>")
    def job_status(job_id):
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            return jsonify({"error": "unknown job"}), 404
        return jsonify(job.to_view())

    @app.route("/api/thumb/<sha>")
    def thumb(sha):
        path = state.thumbnail_path(sha)
        if path.exists():
            return send_file(path, mimetype="image/jpeg")
        src = _resolve_path(sha)
        if src is None:
            return ("not found", 404)
        try:
            preview = load_preview(src)
        except Exception as e:
            log.warning("thumb fail %s: %s", src, e)
            return ("preview failed", 500)
        state.cache_thumbnail(sha, encode_thumbnail_bytes(preview))
        return send_file(path, mimetype="image/jpeg")

    @app.route("/api/preview/<sha>")
    def preview(sha):
        path = state.preview_jpeg_path(sha)
        if path.exists():
            return send_file(path, mimetype="image/jpeg")
        src = _resolve_path(sha)
        if src is None:
            return ("not found", 404)
        try:
            preview_arr = load_preview(src)
        except Exception as e:
            log.warning("preview fail %s: %s", src, e)
            return ("preview failed", 500)
        state.cache_preview_jpeg(sha, _encode_preview_jpeg(preview_arr))
        return send_file(path, mimetype="image/jpeg")

    @app.route("/api/open-folder", methods=["POST"])
    def open_folder():
        """Open a folder in the OS file explorer (used by 'reveal output' buttons)."""
        data = request.get_json(silent=True) or {}
        path = Path(data.get("path", "")).expanduser()
        if not path.exists():
            return jsonify({"error": "no such path"}), 400
        try:
            if os.name == "nt":
                os.startfile(str(path))  # noqa: S606 (this is the OS-blessed call)
            else:
                webbrowser.open(path.as_uri())
        except OSError as e:
            return jsonify({"error": str(e)}), 500
        return jsonify({"ok": True})

    @app.route("/api/faces/<sha>")
    def faces_for(sha):
        """Return detected faces for a frame: bbox + sha-of-face + matched name.

        Uses cached face_detections from metadata. If not yet cached (older
        run that skipped insightface), extract on the fly. Each face gets a
        face_idx so the naming endpoint can identify which one to update.
        """
        meta = state.load_frame_metadata(sha) or {}
        detections = meta.get("face_detections")
        if detections is None:
            # Older metadata only has face_embeddings (no bboxes). If we have
            # those, fall back to inferring positions by re-extracting; that
            # requires reloading the preview though.
            f = _resolve_frame(sha)
            if f is None:
                return jsonify({"faces": [], "error": "unknown sha"}), 404
            try:
                preview = load_preview(f.classify_path)
            except Exception as e:
                return jsonify({"faces": [], "error": str(e)}), 200
            dets = face_id_mod.extract_face_detections(preview)
            detections = []
            for d in dets:
                detections.append({
                    "bbox": d["bbox"],
                    "embedding": d["embedding"].astype(np.float32).tolist(),
                    "det_score": d["det_score"],
                })
            state.update_frame_metadata(sha, face_detections=detections)

        # Match each detection's embedding to the persistent name DB.
        out = []
        for idx, d in enumerate(detections):
            emb = np.asarray(d.get("embedding") or [], dtype=np.float32)
            matched_name = None
            sim = 0.0
            if emb.size == face_names.EMB_DIM:
                matched_name, sim = face_names.match(emb)
            out.append({
                "face_idx": idx,
                "bbox": d.get("bbox"),
                "matched_name": matched_name,
                "match_sim": round(sim, 3),
                "det_score": d.get("det_score"),
            })
        return jsonify({"faces": out})

    @app.route("/api/face-thumb/<sha>/<int:face_idx>")
    def face_thumb(sha, face_idx):
        """Crop the face region from the preview and return as JPEG."""
        import cv2

        meta = state.load_frame_metadata(sha) or {}
        detections = meta.get("face_detections") or []
        if face_idx < 0 or face_idx >= len(detections):
            return ("face_idx out of range", 404)
        bbox = detections[face_idx].get("bbox") or []
        if len(bbox) != 4:
            return ("no bbox", 404)

        f = _resolve_frame(sha)
        if f is None:
            return ("unknown sha", 404)
        try:
            preview = load_preview(f.classify_path)
        except Exception as e:
            return (f"preview failed: {e}", 500)
        h, w = preview.shape[:2]
        x1, y1, x2, y2 = bbox
        # Pad 25% around the bbox so we get hair + chin, not just face plane.
        pw, ph = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(w, x2 + pw)
        y2 = min(h, y2 + ph)
        if x2 <= x1 or y2 <= y1:
            return ("empty crop", 500)
        crop = preview[y1:y2, x1:x2]
        crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return ("encode failed", 500)
        from flask import Response
        return Response(buf.tobytes(), mimetype="image/jpeg")

    @app.route("/api/face/name", methods=["POST"])
    def face_name():
        """Assign a name to a specific (sha, face_idx). Also captures a thumbnail
        if one isn't already stored for this name."""
        import cv2

        data = request.get_json(silent=True) or {}
        sha = (data.get("sha") or "").strip()
        face_idx = data.get("face_idx")
        name = (data.get("name") or "").strip()
        if not sha or face_idx is None or not name:
            return jsonify({"error": "sha, face_idx, name all required"}), 400
        meta = state.load_frame_metadata(sha) or {}
        detections = meta.get("face_detections") or []
        if face_idx < 0 or face_idx >= len(detections):
            return jsonify({"error": "face_idx out of range"}), 400
        emb = np.asarray(detections[face_idx].get("embedding") or [], dtype=np.float32)
        if emb.size != face_names.EMB_DIM:
            return jsonify({"error": "no embedding for that face"}), 400

        # Crop a thumbnail for the library view (only used if this is a new
        # name; existing names keep whatever thumbnail they already have).
        thumb_bytes = None
        bbox = detections[face_idx].get("bbox") or []
        f = _resolve_frame(sha)
        if len(bbox) == 4 and f is not None:
            try:
                preview = load_preview(f.classify_path)
                h, w = preview.shape[:2]
                x1, y1, x2, y2 = bbox
                pw, ph = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
                x1 = max(0, x1 - pw)
                y1 = max(0, y1 - ph)
                x2 = min(w, x2 + pw)
                y2 = min(h, y2 + ph)
                if x2 > x1 and y2 > y1:
                    crop = cv2.resize(preview[y1:y2, x1:x2], (128, 128),
                                      interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        thumb_bytes = buf.tobytes()
            except Exception as e:
                log.warning("face thumb capture failed: %s", e)

        try:
            sim, count = face_names.assign(name, emb, thumbnail_jpeg=thumb_bytes)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"name": name, "merged_with_sim": sim, "count": count})

    @app.route("/api/face/rename", methods=["POST"])
    def face_rename():
        data = request.get_json(silent=True) or {}
        old = (data.get("old") or "").strip()
        new = (data.get("new") or "").strip()
        if not old or not new:
            return jsonify({"error": "old and new required"}), 400
        if not face_names.rename(old, new):
            return jsonify({"error": "name not found or new name invalid"}), 404
        return jsonify({"old": old, "new": new})

    @app.route("/api/face/forget", methods=["POST"])
    def face_forget():
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        if not name:
            return jsonify({"error": "name required"}), 400
        if not face_names.forget(name):
            return jsonify({"error": "name not found"}), 404
        return jsonify({"forgot": name})

    @app.route("/api/face/library")
    def face_library():
        return jsonify({"names": face_names.all_names()})

    @app.route("/api/face/discover", methods=["POST"])
    def face_discover():
        """Cluster every cached face embedding across the library, merging
        into named centroids where possible and creating 'Person N' entries
        for new clusters. Synchronous (runs in request thread), typically
        finishes in seconds even on 10k faces."""
        try:
            summary = face_names.discover_clusters()
        except Exception as e:
            log.exception("discover failed")
            return jsonify({"error": str(e)}), 500
        return jsonify(summary)

    @app.route("/api/face/library-thumb/<name>")
    def face_library_thumb(name):
        from flask import Response
        thumb = face_names.get_thumbnail(name)
        if thumb is None:
            return ("no thumbnail", 404)
        return Response(thumb, mimetype="image/jpeg")

    @app.route("/api/face/names")
    def face_names_list():
        return jsonify({"names": face_names.all_names()})

    @app.route("/api/tags/<sha>")
    def tags_for(sha):
        """Return cached tags, or compute them now from the cached CLIP embedding.

        Fast: tags reuse the embedding we already wrote during scoring, so
        this is one matmul plus a top-K. No model loading, no preview reload.
        Stored under metadata.tags as a list of [tag, similarity] pairs.
        """
        meta = state.load_frame_metadata(sha) or {}
        if "tags" in meta:
            return jsonify({"tags": meta["tags"], "cached": True})
        emb = state.load_embedding(sha)
        if emb is None:
            return jsonify({"tags": [], "error": "no cached embedding for this sha"}), 200
        try:
            pairs = tags_mod.tag_from_embedding(emb, min_sim=float(settings_mod.get("tag_min_sim")))
        except Exception as e:
            log.warning("tag compute failed for %s: %s", sha, e)
            return jsonify({"tags": [], "error": str(e)}), 200
        # Round the sim for compact JSON; stored once, read many times.
        serial = [[t, round(s, 4)] for t, s in pairs]
        state.update_frame_metadata(sha, tags=serial)
        return jsonify({"tags": serial, "cached": False})

    @app.route("/api/details/<sha>")
    def details(sha):
        """Return everything we know about one frame: scores, metrics, EXIF.

        Used by the click-on-photo detail overlay. Pulls cached metadata from
        disk (computed during the pipeline run) and reads EXIF from the source
        JPEG with PIL. RAW EXIF would need rawpy + a separate parser, deferred
        until a real ARW shoot lands in test_photos.
        """
        f = _resolve_frame(sha)
        if f is None:
            return jsonify({"error": "unknown sha"}), 404
        meta = state.load_frame_metadata(sha) or {}
        exif = _extract_exif(f.classify_path)
        # Augment with reverse-geocoded place from the library DB.
        gps_lat = exif.get("gps_lat")
        gps_lon = exif.get("gps_lon")
        if gps_lat is None or gps_lon is None:
            with library._conn() as conn:  # noqa: SLF001
                row = conn.execute(
                    "SELECT lat, lon, place_city, place_region, place_country "
                    "FROM frames WHERE sha=?",
                    (sha,),
                ).fetchone()
            if row:
                lat, lon, city, region, country = row
                if lat is not None:
                    exif["gps_lat"] = lat
                    exif["gps_lon"] = lon
                parts = [city, region, country]
                place = ", ".join(p for p in parts if p)
                if place:
                    exif["place"] = place
        # Eye-region sharpness: the "are the eyes tack-sharp" check. Prefer a
        # cached value if the pipeline already wrote one; otherwise compute it
        # lazily from the preview here (detail-open is a fine place to pay that
        # cost). `None` = no face found, which the panel renders as "no face" —
        # distinct from 0.0 (face found, eyes carry no detail).
        eye_sharpness = meta.get("eye_sharpness")
        eye_sharpness_present = "eye_sharpness" in meta
        if not eye_sharpness_present:
            try:
                preview_arr = load_preview(f.classify_path)
                eye_sharpness = face.best_face_eye_sharpness(preview_arr)
                eye_sharpness_present = True
            except Exception as e:
                log.warning("eye sharpness fail %s: %s", f.display_name, e)
                eye_sharpness_present = False

        # The original pick dict is held in the job; we don't have it here
        # without the job_id, so the SPA passes the pick info client-side and
        # this endpoint just adds the heavy-to-fetch bits (metrics, EXIF).
        return jsonify({
            "sha": sha,
            "display": f.display_name,
            "kind": f.kind,
            "src_path": str(f.classify_path),
            "metrics": meta.get("metrics"),
            "eyes": meta.get("eyes"),
            "face_count": meta.get("face_count"),
            "face_sharpness": meta.get("face_sharpness"),
            "eye_sharpness": eye_sharpness,
            "eye_sharpness_present": eye_sharpness_present,
            "sharpness": meta.get("sharpness"),
            "label": state.get_label(sha),
            "tags": meta.get("tags"),
            "caption": meta.get("caption"),
            "exif": exif,
        })

    @app.route("/api/label", methods=["POST"])
    def set_label():
        """Set a personal taste label on a frame from the main GUI.

        The README has long promised that labels show "in the corner", but
        until now there was no way to SET one without spawning the legacy
        labelling subprocess. This closes that loop: number keys / +/- in the
        Library grid + detail overlay post here. Works for any sha the library
        knows about, not just frames from the current scoring job.
        """
        data = request.get_json(silent=True) or {}
        sha = data.get("sha")
        score = data.get("score")
        if not sha:
            return jsonify({"error": "missing sha"}), 400
        f = _resolve_frame(sha)
        if f is None:
            return jsonify({"error": "unknown sha"}), 404
        if score is None:
            return jsonify({"error": "missing score"}), 400
        try:
            score = int(score)
        except (TypeError, ValueError):
            return jsonify({"error": "score must be an integer"}), 400
        if not state.SCORE_MIN <= score <= state.SCORE_MAX:
            return jsonify({
                "error": f"score out of [{state.SCORE_MIN}, {state.SCORE_MAX}]"
            }), 400

        # Cache an embedding lazily so the taste head can later train on this
        # label (mirrors server.py's set_label). Best-effort; a label is still
        # recorded even if the embedding can't be computed.
        if state.load_embedding(sha) is None:
            try:
                preview_arr = load_preview(f.classify_path)
                state.cache_embedding(sha, aesthetic.encode_image(preview_arr))
            except Exception as e:
                log.warning("embedding fail %s: %s", getattr(f, "display_name", sha), e)

        state.add_label(sha, score, f.stem, str(f.classify_path))
        return jsonify({"sha": sha, "score": score})

    @app.route("/api/label", methods=["DELETE"])
    def clear_label():
        """Clear a frame's taste label. Accepts the sha in the JSON body
        (DELETE /api/label {sha}) per the cross-module contract."""
        data = request.get_json(silent=True) or {}
        sha = data.get("sha") or request.args.get("sha")
        if not sha:
            return jsonify({"error": "missing sha"}), 400
        if _resolve_frame(sha) is None:
            return jsonify({"error": "unknown sha"}), 404
        import sqlite3

        with sqlite3.connect(state.LABELS_DB) as conn:
            conn.execute("DELETE FROM labels WHERE sha256=?", (sha,))
        return jsonify({"sha": sha, "score": None})

    @app.route("/api/label-spawn", methods=["POST"])
    def label_spawn():
        """Spawn (or reuse) a `banger ui <folder>` subprocess and return its URL.

        Labelling needs the full frame discovery + hashing path, which is
        already done well in server.py. Rather than duplicate the routes here,
        we run that module in a child process bound to a free port and embed
        it in an iframe. One subprocess per input folder, cached so the user
        can flip between Results and Label without re-walking the folder.
        """
        import socket
        import subprocess
        import sys

        data = request.get_json(silent=True) or {}
        folder = (data.get("input_dir") or "").strip()
        if not folder:
            return jsonify({"error": "input_dir required"}), 400
        folder_path = Path(folder).expanduser()
        if not folder_path.is_dir():
            return jsonify({"error": f"not a folder: {folder_path}"}), 400

        cached = _label_subprocesses.get(str(folder_path))
        if cached and cached["proc"].poll() is None:
            return jsonify({"url": f"http://127.0.0.1:{cached['port']}/", "reused": True})

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]

        proc = subprocess.Popen(
            [sys.executable, "-m", "banger", "ui", str(folder_path), "--port", str(free_port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _label_subprocesses[str(folder_path)] = {"port": free_port, "proc": proc}

        # Give the child Flask a moment to bind so the iframe's first nav succeeds.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            with socket.socket() as s:
                try:
                    s.connect(("127.0.0.1", free_port))
                    break
                except OSError:
                    time.sleep(0.1)

        return jsonify({"url": f"http://127.0.0.1:{free_port}/", "reused": False})

    @app.route("/api/setup-status")
    def setup_status():
        # eyes_mod.mediapipe_available() does a fresh import attempt so it
        # picks up a freshly-installed package without process restart.
        if _setup_state["phase"] == "mediapipe" and eyes_mod.mediapipe_available():
            _setup_state["phase"] = "ready"
            _setup_state["message"] = "ready"
        return jsonify({
            **_setup_state,
            "elapsed": round(time.monotonic() - _setup_state["started_at"], 1),
            "mediapipe": eyes_mod.mediapipe_available(),
            "scene_model": scene_kmeans.exists(),
        })

    @app.route("/api/state")
    def gui_state():
        """One-shot snapshot the UI needs at boot: defaults, head present, etc."""
        stats = state.cache_stats()
        labels = state.labels_dict()
        return jsonify({
            "defaults": {
                "top_n": int(os.environ.get("BANGER_TOP_N", "10")),
                "strategy": "kmeans",
                "diversity": 0.5,
                "recursive": True,
            },
            "head_loaded": taste_head.exists(),
            "scene_model": scene_kmeans.exists(),
            "mediapipe": eyes_mod.mediapipe_available(),
            "insightface": face_id_mod.insightface_available(),
            "labels_total": len(labels),
            "cache_counts": {k: v["count"] for k, v in stats.items()},
        })

    return app


def _install_file_logging() -> None:
    """Route logs to a rotating file under STATE_DIR/logs.

    The pywebview GUI has no terminal, so without this every warning, traceback
    and pipeline diagnostic vanishes — the only place to look after a crash is
    this file. Best-effort: if the log dir can't be created we just keep going
    with whatever handlers are already attached. Idempotent (won't double-add
    its handler across repeated serve() calls in the same process).
    """
    from logging.handlers import RotatingFileHandler

    try:
        log_dir = state.STATE_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "banger.log"

        root_logger = logging.getLogger("banger")
        # Don't stack duplicate file handlers if serve() is called twice.
        for h in root_logger.handlers:
            if isinstance(h, RotatingFileHandler) and getattr(h, "_banger_file", False):
                return
        handler = RotatingFileHandler(
            log_path, maxBytes=2 << 20, backupCount=5, encoding="utf-8"
        )
        handler._banger_file = True  # type: ignore[attr-defined]
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        ))
        root_logger.addHandler(handler)
        if root_logger.level == logging.NOTSET or root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)
        log.info("file logging installed at %s", log_path)
    except Exception as e:  # never let logging setup take down startup
        log.warning("file logging setup failed: %s", e)

    # Opportunistic, best-effort snapshot of the only irreplaceable user data
    # (labels.db) at startup. Tiny + idempotent; swallow every failure.
    try:
        backup = state.backup_labels()
        log.info("labels backed up to %s", backup)
    except Exception as e:
        log.warning("startup labels backup skipped: %s", e)


def serve(port: int = 8765, open_window: bool = True) -> None:
    """Entry point used by `banger gui`. Launches Flask in a background thread
    and pywebview on the main thread (which has to be the main thread on macOS)."""

    _install_file_logging()

    window_holder: dict = {}
    app = build_app(window_holder)

    def _flask():
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False, threaded=True)

    t = threading.Thread(target=_flask, daemon=True)
    t.start()

    # auto-setup loads CLIP synchronously at its start, so a separate
    # warm-clip thread would race it (both calling the same lru_cache'd
    # load triggers torch's "meta tensor" error). Single thread only.
    log.info("spawning auto-setup thread")
    threading.Thread(target=_auto_setup, daemon=True).start()

    # Wait briefly for the Flask socket so the webview's first nav doesn't
    # race-fail. 200 ms is plenty on a modern machine; on a slow one we
    # retry up to 2 s.
    import socket as _socket

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with _socket.socket() as s:
            try:
                s.connect(("127.0.0.1", port))
                break
            except OSError:
                time.sleep(0.05)

    if not open_window:
        log.info("flask running on http://127.0.0.1:%d (no webview, headless mode)", port)
        t.join()
        return

    try:
        import webview
    except ImportError:
        log.error("pywebview is not installed; install with `pip install banger[gui]`")
        return

    # Cache-bust the URL each launch. WebView2 (and WebKit on macOS) often
    # ignore HTTP cache headers for the navigation document, so the only
    # reliable way to force a fresh HTML+CSS fetch is to change the URL.
    cache_bust = int(time.time())
    win = webview.create_window(
        title="banger",
        url=f"http://127.0.0.1:{port}/gui?v={cache_bust}",
        width=1280, height=820, min_size=(900, 600),
        background_color="#0c0c0c",
    )
    window_holder["win"] = win
    webview.start()


def _auto_setup() -> None:
    """Background bootstrap. Updates _setup_state so the GUI splash shows
    progress. Three phases:

    1. scenes: fit KMeans on cached embeddings if not already fit
    2. mediapipe: spawn detached pip install if not installed (survives GUI
       close), poll for completion by trying the import periodically
    3. ready: splash dismisses
    """
    import subprocess
    import sys

    # Load CLIP synchronously before anything that needs text embeddings.
    # Otherwise scene_kmeans.fit can race the concurrent CLIP-warm thread
    # and hit "Cannot copy out of meta tensor" when both touch the same
    # lru_cache'd load at the same time.
    _setup_state["phase"] = "clip"
    _setup_state["message"] = "Loading CLIP model (one-time, ~8s)…"
    try:
        aesthetic._load()
    except Exception as e:
        log.warning("auto-setup: CLIP load failed: %s", e)

    _setup_state["phase"] = "scenes"
    _setup_state["message"] = "Fitting scene clusters…"
    try:
        if not scene_kmeans.exists():
            stats = state.cache_stats()
            n = stats.get("embeddings", {}).get("count", 0)
            if n >= 5:
                log.info("auto-setup: fitting scene KMeans (k=5) on %d embeddings", n)
                scene_kmeans.fit(k=5)
            else:
                log.info("auto-setup: skipping scene fit, only %d embeddings cached", n)
    except Exception as e:
        log.warning("auto-setup: scene fit failed: %s", e)

    # GPS / place backfill: always runs, regardless of mediapipe state. Cheap
    # when there's nothing to do (one SELECT).
    try:
        gps_n = library.backfill_gps()
        if gps_n:
            log.info("auto-setup: backfilled GPS on %d frames", gps_n)
        place_n = library.reverse_geocode_unreferenced()
        if place_n:
            log.info("auto-setup: reverse-geocoded %d frames", place_n)
    except Exception as e:
        log.warning("auto-setup: geocode backfill failed: %s", e)

    if eyes_mod.mediapipe_available():
        _setup_state["phase"] = "ready"
        _setup_state["message"] = "ready"
        return

    marker = state.STATE_DIR / "mediapipe_install_attempted"
    if marker.exists() and not eyes_mod.mediapipe_available():
        # Previous install attempt failed; don't loop. User can delete the
        # marker if they want to retry.
        _setup_state["phase"] = "ready"
        _setup_state["message"] = (
            "mediapipe install previously failed; --eye-gate disabled. "
            "Delete ~/.local/share/banger-pipeline/mediapipe_install_attempted to retry."
        )
        return

    _setup_state["phase"] = "mediapipe"
    _setup_state["message"] = "Installing mediapipe (one-time, ~30-60s)…"
    log_path = state.STATE_DIR / "mediapipe_install.log"
    proc = None
    try:
        state.STATE_DIR.mkdir(parents=True, exist_ok=True)
        marker.touch()
        creationflags = 0
        if os.name == "nt":
            creationflags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        # `--no-deps` skips opencv-contrib-python which conflicts with the
        # running process's loaded cv2.pyd (Windows file lock). mediapipe at
        # runtime uses the already-installed opencv-python for image ops.
        with open(log_path, "w", encoding="utf-8") as fp:
            proc = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", "--no-deps", "mediapipe",
                 "absl-py", "attrs", "flatbuffers", "jax", "matplotlib", "protobuf",
                 "sounddevice", "sentencepiece"],
                stdout=fp, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=creationflags, close_fds=True,
            )
        log.info("auto-setup: mediapipe install detached (log: %s)", log_path)
    except OSError as e:
        log.warning("auto-setup: mediapipe install spawn failed: %s", e)
        _setup_state["phase"] = "ready"
        _setup_state["message"] = f"mediapipe install spawn failed: {e}"
        return

    # Poll the subprocess + the import every 2s. Subprocess exit + still-not-
    # importable means the install failed; show the tail of the log so the
    # splash explains why.
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        time.sleep(2.0)
        if eyes_mod.mediapipe_available():
            _setup_state["phase"] = "ready"
            _setup_state["message"] = "mediapipe installed"
            log.info("auto-setup: mediapipe install completed")
            return
        if proc is not None and proc.poll() is not None:
            # Subprocess exited but import still fails: that's an install error.
            tail = ""
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                tail = " | ".join(tail[-3:])
            except OSError:
                pass
            _setup_state["phase"] = "ready"
            _setup_state["message"] = (
                f"mediapipe install failed (rc={proc.returncode}). "
                f"See {log_path}. Last lines: {tail[-200:]}"
            )
            log.warning("auto-setup: mediapipe install exited rc=%d", proc.returncode)
            return
    _setup_state["phase"] = "ready"
    _setup_state["message"] = "mediapipe install still running (check log later)"


# ---------------------------------------------------------------------------
# SPA template (single inline string; client-side routing across views)
# ---------------------------------------------------------------------------

_SPA_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>banger</title>
<style>
  :root { color-scheme: dark; --accent: #ffaa55; --bg: #0c0c0c; --bg2: #161616; --bg3: #1f1f1f; --line: #2a2a2a; --fg: #eee; --dim: #888; --green: #5fa05f; --red: #a05f5f; }
  html, body { height: 100%; }
  body { margin: 0; background: var(--bg); color: var(--fg); font-family: -apple-system, ui-sans-serif, system-ui, "Segoe UI", sans-serif; display: flex; flex-direction: column; overflow: hidden; }
  * { box-sizing: border-box; }
  button { font: inherit; cursor: pointer; }

  header { flex: 0 0 auto; padding: .6rem 1rem; background: #111; border-bottom: 1px solid var(--line); display: flex; align-items: center; gap: 1rem; }
  header h1 { margin: 0; font-size: 1rem; letter-spacing: .04em; }
  header nav { display: flex; gap: .35rem; margin-left: auto; }
  header nav button { background: transparent; color: var(--dim); border: 1px solid transparent; padding: .3rem .7rem; border-radius: 4px; font-size: .8rem; }
  header nav button:hover { color: var(--fg); }
  header nav button.active { color: var(--fg); background: var(--bg3); border-color: var(--line); }

  main { flex: 1 1 auto; min-height: 0; overflow: hidden; display: flex; }
  .view { flex: 1 1 auto; min-height: 0; overflow: auto; padding: 2rem; }
  .view.hidden { display: none; }

  /* welcome */
  .welcome-wrap { max-width: 720px; margin: 4rem auto; text-align: center; }
  .welcome-wrap h2 { margin: 0 0 .5rem; font-weight: 500; font-size: 1.4rem; }
  .welcome-wrap p.lead { color: var(--dim); margin: 0 0 2rem; }
  .drop-zone { border: 2px dashed var(--line); border-radius: 12px; padding: 3rem 2rem; transition: border-color .15s, background .15s; cursor: pointer; }
  .drop-zone:hover, .drop-zone.dragover { border-color: var(--accent); background: rgba(255,170,85,.04); }
  .drop-zone .icon { font-size: 3rem; line-height: 1; margin-bottom: .8rem; opacity: .7; }
  .drop-zone .hint { color: var(--dim); font-size: .85rem; margin-top: .5rem; }
  .recent { margin-top: 2.5rem; text-align: left; }
  .recent h3 { font-size: .75rem; color: var(--dim); font-weight: 400; text-transform: uppercase; letter-spacing: .1em; margin: 0 0 .6rem; }
  .recent-list { list-style: none; padding: 0; margin: 0; }
  .recent-list li { padding: .5rem .7rem; background: var(--bg2); border: 1px solid var(--line); border-radius: 4px; margin-bottom: .3rem; font-family: ui-monospace, "Consolas", monospace; font-size: .8rem; color: #ccc; cursor: pointer; display: flex; justify-content: space-between; align-items: center; }
  .recent-list li:hover { border-color: var(--accent); }
  .recent-list .go { color: var(--dim); font-size: .7rem; }

  .settings-row { display: flex; gap: 1.2rem; flex-wrap: wrap; max-width: 720px; margin: 1.5rem auto 0; }
  .opt { display: flex; align-items: center; gap: .4rem; font-size: .8rem; color: #bbb; }
  .opt input[type=number] { width: 60px; background: var(--bg2); border: 1px solid var(--line); color: var(--fg); border-radius: 3px; padding: .2rem .35rem; font: inherit; font-size: .8rem; }
  .opt select { background: var(--bg2); border: 1px solid var(--line); color: var(--fg); border-radius: 3px; padding: .2rem .35rem; font: inherit; font-size: .8rem; }
  .opt input[type=checkbox] { accent-color: var(--accent); }

  /* progress */
  .progress-wrap { max-width: 720px; margin: 4rem auto; }
  .progress-wrap h2 { font-weight: 500; margin: 0 0 .6rem; font-size: 1.2rem; }
  .progress-wrap .stage { color: var(--dim); font-size: .85rem; margin-bottom: 1.5rem; }
  .bar { height: 8px; background: var(--bg2); border: 1px solid var(--line); border-radius: 4px; overflow: hidden; }
  .bar > div { height: 100%; background: var(--accent); width: 0; transition: width .3s ease; }
  .progress-meta { display: flex; justify-content: space-between; font-size: .75rem; color: var(--dim); margin-top: .5rem; font-variant-numeric: tabular-nums; }
  .log { margin-top: 2rem; max-height: 320px; overflow: auto; background: var(--bg2); border: 1px solid var(--line); border-radius: 4px; padding: .6rem .8rem; font-family: ui-monospace, monospace; font-size: .72rem; color: #aaa; }
  .log .row { white-space: pre-wrap; line-height: 1.4; }

  /* results */
  .results-bar { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }
  .results-bar h2 { margin: 0; font-weight: 500; font-size: 1.1rem; }
  .results-bar .meta { color: var(--dim); font-size: .8rem; font-variant-numeric: tabular-nums; }
  .results-bar button { background: var(--bg3); color: var(--fg); border: 1px solid var(--line); border-radius: 4px; padding: .4rem .8rem; font-size: .8rem; margin-left: auto; }
  .results-bar button + button { margin-left: .35rem; }
  .results-bar button.primary { background: var(--accent); color: #111; border-color: var(--accent); }
  .results-bar button:hover { border-color: var(--accent); }

  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: .8rem; }
  .card { background: var(--bg2); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; transition: border-color .12s; cursor: pointer; }
  .card:hover { border-color: var(--accent); }
  .card .img-wrap { aspect-ratio: 3/2; background: #000; position: relative; }
  .card .img-wrap img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .card .rank { position: absolute; top: 6px; left: 6px; background: rgba(0,0,0,.65); color: var(--accent); font-family: ui-monospace, monospace; font-size: .7rem; padding: 1px 6px; border-radius: 3px; }
  .card .stars { position: absolute; top: 6px; right: 6px; background: rgba(0,0,0,.65); color: #ffd56a; font-size: .8rem; padding: 1px 5px; border-radius: 3px; letter-spacing: -1px; }
  .card .meta { padding: .45rem .55rem; font-size: .72rem; color: #ccc; display: flex; justify-content: space-between; align-items: baseline; }
  .card .meta .stem { font-family: ui-monospace, monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .card .meta .score { color: var(--dim); font-variant-numeric: tabular-nums; }
  .card .scene { font-size: .65rem; color: #c9c; background: #2a1f2a; padding: 1px 5px; border-radius: 2px; font-family: ui-monospace, monospace; }

  /* settings */
  .settings-pane { max-width: 720px; margin: 0 auto; }
  .settings-pane h2 { font-weight: 500; margin: 0 0 1.2rem; font-size: 1.1rem; }
  .settings-card { background: var(--bg2); border: 1px solid var(--line); border-radius: 6px; padding: 1rem 1.2rem; margin-bottom: 1rem; }
  .settings-card h3 { margin: 0 0 .6rem; font-size: .85rem; color: var(--dim); font-weight: 400; text-transform: uppercase; letter-spacing: .08em; }
  .settings-card label { display: flex; align-items: center; gap: .6rem; margin: .35rem 0; font-size: .85rem; }
  .settings-card label span.k { color: var(--dim); width: 130px; flex-shrink: 0; }
  .lever-row { display: grid; grid-template-columns: 180px 1fr 70px; align-items: center; gap: .6rem; margin: .5rem 0; font-size: .82rem; }
  .lever-row .lever-name { display: flex; align-items: center; gap: .35rem; color: var(--fg); }
  .lever-row .lever-info { display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px; border-radius: 50%; background: var(--bg3); color: var(--dim); font-size: .65rem; font-style: italic; cursor: help; user-select: none; border: 1px solid var(--line); }
  .lever-row .lever-info:hover { background: var(--accent); color: #000; border-color: var(--accent); }
  .lever-row input[type=range] { width: 100%; }
  .lever-row input[type=number], .lever-row select { background: var(--bg3); color: var(--fg); border: 1px solid var(--line); border-radius: 3px; padding: .25rem .4rem; font-size: .8rem; width: 100%; }
  .lever-row .lever-val { color: var(--dim); font-variant-numeric: tabular-nums; text-align: right; font-size: .78rem; }
  /* Bottom background-work strip. Subtle, hides when idle. Click expand to
     jump to the full Running view. */
  .bg-strip { position: fixed; left: 0; right: 0; bottom: 0; height: 28px; background: rgba(15,15,15,.94); border-top: 1px solid var(--line); display: flex; align-items: center; gap: .75rem; padding: 0 .8rem; z-index: 40; font-size: .72rem; color: #bbb; backdrop-filter: blur(4px); }
  .bg-strip-bar { flex: 0 0 160px; height: 4px; background: var(--bg2); border-radius: 2px; overflow: hidden; }
  .bg-strip-bar > div { height: 100%; background: var(--accent); width: 0; transition: width .4s ease; }
  .bg-strip-text { flex: 1 1 auto; font-family: ui-monospace, monospace; font-size: .7rem; color: #ccc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .bg-strip button { background: transparent; color: var(--dim); border: 1px solid var(--line); border-radius: 3px; padding: 1px 7px; font-size: .65rem; cursor: pointer; }
  .bg-strip button:hover { color: var(--accent); border-color: var(--accent); }

  .toast { position: fixed; bottom: 1rem; right: 1rem; background: var(--bg3); border: 1px solid var(--accent); padding: .5rem .9rem; border-radius: 4px; font-size: .8rem; opacity: 0; transition: opacity .2s; pointer-events: none; }
  .toast.show { opacity: 1; }
  .badge { padding: 1px 6px; border-radius: 3px; font-size: .65rem; font-family: ui-monospace, monospace; }
  .badge.ok { background: rgba(95,160,95,.15); color: var(--green); border: 1px solid rgba(95,160,95,.3); }
  .badge.warn { background: rgba(160,95,95,.15); color: var(--red); border: 1px solid rgba(160,95,95,.3); }

  /* hero detail overlay */
  .overlay { position: fixed; inset: 0; background: rgba(0,0,0,.92); display: none; z-index: 50; padding: 2rem; cursor: zoom-out; }
  .overlay.show { display: flex; }
  .overlay-inner { margin: auto; display: flex; gap: 1.5rem; max-width: 1500px; width: 100%; height: 100%; align-items: stretch; cursor: default; }
  .overlay-image { flex: 1 1 auto; display: flex; align-items: center; justify-content: center; min-width: 0; }
  .overlay-image img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 4px; box-shadow: 0 10px 60px rgba(0,0,0,.6); }
  .overlay-panel { flex: 0 0 360px; overflow-y: auto; padding-right: .4rem; }
  .overlay-panel h3 { margin: 1.2rem 0 .4rem; font-size: .7rem; color: var(--dim); font-weight: 500; text-transform: uppercase; letter-spacing: .12em; }
  .overlay-panel h3:first-child { margin-top: 0; }
  .overlay-panel .head { display: flex; align-items: baseline; gap: .6rem; flex-wrap: wrap; }
  .overlay-panel .head .rank { font-family: ui-monospace, monospace; color: var(--accent); font-size: .85rem; }
  .overlay-panel .head .stars { color: #ffd56a; font-size: 1rem; letter-spacing: -1px; }
  .overlay-panel .head .stem { font-family: ui-monospace, monospace; font-size: .85rem; word-break: break-all; }
  .overlay-panel .rating-bar { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; margin: .4rem 0 .2rem; }
  .overlay-panel .overlay-rating { color: #ffd56a; font-size: .85rem; font-family: ui-monospace, monospace; }
  .overlay-panel .rating-hint { color: var(--dim); font-size: .65rem; }
  .overlay-panel dl { display: grid; grid-template-columns: minmax(110px, auto) 1fr; gap: .25rem .8rem; margin: 0; font-size: .8rem; }
  .overlay-panel dt { color: var(--dim); font-size: .75rem; }
  .overlay-panel dd { margin: 0; font-variant-numeric: tabular-nums; font-family: ui-monospace, monospace; color: #ddd; word-break: break-all; }
  .overlay-panel .bar-row { display: grid; grid-template-columns: 110px 1fr 40px; gap: .5rem; align-items: center; margin: .15rem 0; font-size: .75rem; }
  .overlay-panel .bar-row .label { color: var(--dim); }
  .overlay-panel .bar-row .meter { height: 5px; background: var(--bg3); border-radius: 2px; overflow: hidden; }
  .overlay-panel .bar-row .meter > div { height: 100%; background: var(--accent); border-radius: 2px; }
  .overlay-panel .bar-row .val { font-family: ui-monospace, monospace; color: #ddd; text-align: right; font-variant-numeric: tabular-nums; }
  .overlay-panel .flag { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: .65rem; margin-right: .25rem; font-family: ui-monospace, monospace; }
  .overlay-panel .flag.on { background: rgba(160,95,95,.2); color: #e0a0a0; border: 1px solid rgba(160,95,95,.4); }
  .overlay-panel .flag.off { background: rgba(95,160,95,.12); color: #a0d0a0; border: 1px solid rgba(95,160,95,.3); }
  .overlay-panel .empty { color: var(--dim); font-style: italic; font-size: .75rem; }
  .overlay-panel .tag-chips { display: flex; flex-wrap: wrap; gap: .25rem; margin: .3rem 0; }
  .overlay-panel .tag-chip { background: var(--bg3); color: #cce0cc; border: 1px solid #2a3a2a; border-radius: 3px; padding: 2px 7px; font-size: .72rem; font-family: ui-monospace, monospace; }
  .overlay-panel .face-row { display: flex; gap: .5rem; margin: .35rem 0; align-items: flex-start; flex-wrap: wrap; }
  .overlay-panel .face-tile { display: flex; flex-direction: column; align-items: center; gap: .25rem; background: var(--bg2); border: 1px solid var(--line); border-radius: 4px; padding: .35rem; }
  .overlay-panel .face-tile img { width: 64px; height: 64px; object-fit: cover; border-radius: 3px; background: #000; }
  .overlay-panel .face-tile .name { font-size: .7rem; color: #ddd; font-family: ui-monospace, monospace; max-width: 90px; text-align: center; word-break: break-word; }
  .overlay-panel .face-tile .name.editable { cursor: pointer; text-decoration: underline dotted; text-decoration-color: rgba(255,170,85,.4); }
  .overlay-panel .face-tile .name.editable:hover { color: var(--accent); }
  .overlay-panel .face-tile .name.unnamed { color: var(--accent); }
  .overlay-panel .face-tile input { background: var(--bg3); border: 1px solid var(--accent); color: var(--fg); border-radius: 3px; padding: 1px 4px; font: inherit; font-size: .7rem; width: 90px; }

  /* face library */
  .face-library { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: .6rem; margin-top: .5rem; }
  .face-card { background: var(--bg3); border: 1px solid var(--line); border-radius: 6px; padding: .5rem; display: flex; flex-direction: column; align-items: center; gap: .35rem; }
  .face-card img { width: 96px; height: 96px; object-fit: cover; border-radius: 4px; background: #000; }
  .face-card .lib-name { font-family: ui-monospace, monospace; font-size: .8rem; color: var(--fg); cursor: pointer; text-decoration: underline dotted; }
  .face-card .lib-name:hover { color: var(--accent); }
  .face-card .lib-meta { font-size: .65rem; color: var(--dim); }
  .face-card .lib-actions { display: flex; gap: .3rem; margin-top: .15rem; }
  .face-card .lib-actions button { background: transparent; color: var(--dim); border: 1px solid var(--line); border-radius: 3px; padding: 1px 6px; font-size: .65rem; }
  .face-card .lib-actions button:hover { color: var(--red); border-color: var(--red); }
  .face-card input { background: var(--bg2); border: 1px solid var(--accent); color: var(--fg); border-radius: 3px; padding: 2px 5px; font: inherit; font-size: .75rem; width: 110px; text-align: center; }
  .overlay .close { position: absolute; top: 1rem; right: 1rem; background: var(--bg3); color: var(--fg); border: 1px solid var(--line); padding: .35rem .8rem; border-radius: 4px; cursor: pointer; z-index: 1; }
  .overlay .hint { position: absolute; bottom: 1rem; left: 50%; transform: translateX(-50%); color: var(--dim); font-size: .7rem; pointer-events: none; }

  /* library */
  .lib-shell { display: flex; height: 100%; min-height: 0; }
  .lib-sidebar { flex: 0 0 240px; background: #0a0a0a; border-right: 1px solid var(--line); overflow-y: auto; padding: 1rem .8rem; }
  .lib-sidebar h3 { margin: 0 0 .5rem; font-size: .68rem; color: var(--dim); font-weight: 500; text-transform: uppercase; letter-spacing: .1em; }
  .lib-section { margin-bottom: 1.5rem; }
  .lib-roots, .lib-folders { list-style: none; margin: 0; padding: 0; }
  .lib-roots li, .lib-folders li { padding: .35rem .55rem; border-radius: 4px; cursor: pointer; font-size: .8rem; color: #ccc; display: flex; justify-content: space-between; align-items: center; gap: .4rem; font-family: ui-monospace, monospace; }
  .lib-roots li:hover, .lib-folders li:hover { background: var(--bg2); }
  .lib-roots li.active, .lib-folders li.active { background: var(--bg3); color: var(--accent); }
  .lib-roots li .count, .lib-folders li .count { color: var(--dim); font-size: .7rem; }
  .lib-roots li .remove { color: var(--dim); font-size: .8rem; cursor: pointer; opacity: 0; transition: opacity .15s; padding: 0 .25rem; }
  .lib-roots li:hover .remove { opacity: 1; }
  .lib-roots li .remove:hover { color: var(--red); }
  .lib-add button { background: transparent; color: var(--dim); border: 1px dashed var(--line); border-radius: 4px; padding: .35rem .6rem; width: 100%; font: inherit; font-size: .75rem; cursor: pointer; margin-top: .35rem; }
  .lib-add button:hover { color: var(--accent); border-color: var(--accent); }
  .lib-filter-row { display: flex; justify-content: space-between; align-items: center; gap: .4rem; font-size: .75rem; color: var(--dim); margin: .25rem 0; }
  .lib-filter-row select { background: var(--bg2); border: 1px solid var(--line); color: var(--fg); border-radius: 3px; padding: 2px 5px; font: inherit; font-size: .75rem; flex: 1 1 auto; max-width: 140px; }
  #lib-search { width: 100%; background: var(--bg2); border: 1px solid var(--line); color: var(--fg); border-radius: 4px; padding: .35rem .55rem; font: inherit; font-size: .8rem; }
  #lib-search:focus { border-color: var(--accent); outline: none; }
  .lib-main { flex: 1 1 auto; display: flex; flex-direction: column; min-width: 0; }
  .lib-toolbar { flex: 0 0 auto; padding: .8rem 1rem; border-bottom: 1px solid var(--line); display: flex; align-items: center; gap: .8rem; background: #0f0f0f; }
  .lib-toolbar h2 { margin: 0; font-weight: 500; font-size: 1rem; }
  .lib-toolbar .lib-meta { color: var(--dim); font-size: .75rem; font-variant-numeric: tabular-nums; }
  .lib-toolbar button { background: var(--bg3); color: var(--fg); border: 1px solid var(--line); border-radius: 4px; padding: .35rem .8rem; font-size: .8rem; }
  .lib-toolbar button:hover { border-color: var(--accent); }
  .lib-toolbar button.primary { background: var(--accent); color: #111; border-color: var(--accent); }
  /* WebView2 silently ignores both `aspect-ratio` and the padding-bottom
     trick inside a CSS Grid with auto rows. Pinning the row height in pixels
     and using flex inside each cell is the only thing that holds. */
  .lib-grid { flex: 1 1 auto; overflow-y: auto; padding: .8rem; display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); grid-auto-rows: 200px; gap: .5rem; align-content: start; }
  .lib-cell { background: var(--bg2); border: 1px solid var(--line); border-radius: 4px; overflow: hidden; cursor: pointer; transition: border-color .12s; position: relative; height: 200px; display: flex; flex-direction: column; }
  .lib-cell:hover { border-color: var(--accent); }
  .lib-cell.focused { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent); }
  .lib-cell .lib-img-wrap { flex: 1 1 auto; min-height: 0; background: #000; position: relative; overflow: hidden; }
  .lib-cell .lib-img-wrap img { width: 100%; height: 100%; object-fit: contain; display: block; }
  .lib-cell .lib-label { padding: .3rem .5rem; font-size: .7rem; color: #ccc; font-family: ui-monospace, monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .lib-cell .lib-label .lib-rating { color: #ffd56a; margin-right: .35rem; }
  .lib-cell .lib-badges { position: absolute; top: 4px; left: 4px; display: flex; gap: 3px; }
  .lib-cell .lib-badge { background: rgba(0,0,0,.7); color: #ddd; font-size: .6rem; padding: 1px 5px; border-radius: 2px; font-family: ui-monospace, monospace; }
  .lib-cell .lib-badge.scored { color: var(--green); }
  .lib-cell .lib-badge.face { color: #c9d; }

  .lib-onboarding { grid-column: 1 / -1; max-width: 560px; margin: 4rem auto; text-align: center; padding: 2rem 1rem; }
  .lib-onboarding h2 { font-weight: 500; margin: 0 0 .6rem; font-size: 1.3rem; }
  .lib-onboarding p { color: var(--dim); line-height: 1.5; margin: 0 0 1.6rem; font-size: .85rem; }
  .onboarding-actions { display: flex; gap: .8rem; justify-content: center; flex-wrap: wrap; }
  .onboarding-actions button { background: var(--bg3); color: var(--fg); border: 1px solid var(--line); border-radius: 6px; padding: 1rem 1.5rem; font-size: .9rem; cursor: pointer; transition: border-color .12s, background .12s; }
  .onboarding-actions button:hover { border-color: var(--accent); background: var(--bg2); }

  /* editor */
  .ed-shell { display: flex; height: 100%; min-height: 0; overflow: hidden; }
  .ed-canvas { flex: 1 1 auto; background: #000; display: flex; align-items: center; justify-content: center; min-width: 0; min-height: 0; position: relative; padding: 1rem; overflow: hidden; }
  .ed-canvas img { max-width: 100%; max-height: 100%; object-fit: contain; }
  .ed-info { position: absolute; top: 1rem; left: 1rem; color: var(--dim); font-size: .75rem; font-family: ui-monospace, monospace; background: rgba(0,0,0,.6); padding: .25rem .55rem; border-radius: 4px; }
  .ed-sidebar { flex: 0 0 320px; background: #0a0a0a; border-left: 1px solid var(--line); overflow-y: auto; overflow-x: hidden; min-height: 0; }
  .ed-sidebar::-webkit-scrollbar { width: 8px; }
  .ed-sidebar::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
  .ed-sidebar::-webkit-scrollbar-thumb:hover { background: #444; }
  .ed-sidebar::-webkit-scrollbar-track { background: transparent; }
  .ed-toolbar { position: sticky; top: 0; background: #0a0a0a; padding: .6rem .8rem; border-bottom: 1px solid var(--line); display: flex; gap: .3rem; flex-wrap: wrap; z-index: 2; }
  .ed-toolbar button { background: var(--bg3); color: var(--fg); border: 1px solid var(--line); border-radius: 4px; padding: .35rem .7rem; font-size: .75rem; cursor: pointer; }
  .ed-toolbar button:hover { border-color: var(--accent); }
  .ed-toolbar button.primary { background: var(--accent); color: #111; border-color: var(--accent); }
  .ed-toolbar button.muted { background: transparent; color: var(--dim); }
  .ed-section { padding: .8rem 1rem; border-bottom: 1px solid var(--line); }
  .ed-section h3 { margin: 0 0 .6rem; font-size: .65rem; color: var(--dim); font-weight: 500; text-transform: uppercase; letter-spacing: .12em; }
  .slider-row { display: grid; grid-template-columns: 90px 1fr 40px; gap: .5rem; align-items: center; font-size: .8rem; margin: .3rem 0; }
  .slider-row label { color: #bbb; font-size: .75rem; }
  .slider-row .val { text-align: right; color: var(--fg); font-family: ui-monospace, monospace; font-size: .72rem; }
  .slider-row input[type=range] { width: 100%; accent-color: var(--accent); }

  /* import modal */
  .modal-backdrop { position: fixed; inset: 0; background: rgba(0,0,0,.7); display: flex; align-items: center; justify-content: center; z-index: 90; }
  .modal { background: var(--bg2); border: 1px solid var(--line); border-radius: 8px; padding: 1.4rem 1.6rem; max-width: 520px; width: 90%; }
  .modal h2 { margin: 0 0 .4rem; font-weight: 500; font-size: 1.1rem; }
  .modal-hint { color: var(--dim); font-size: .8rem; line-height: 1.45; margin: 0 0 1rem; }
  .modal-row { display: flex; flex-direction: column; gap: .25rem; margin: .6rem 0; }
  .modal-row > label { color: var(--dim); font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; }
  .modal-input { display: flex; gap: .35rem; }
  .modal-input input { flex: 1 1 auto; background: var(--bg3); border: 1px solid var(--line); color: var(--fg); border-radius: 4px; padding: .35rem .5rem; font: inherit; font-size: .8rem; font-family: ui-monospace, monospace; }
  .modal-input button { background: var(--bg3); color: var(--fg); border: 1px solid var(--line); border-radius: 4px; padding: .35rem .7rem; font-size: .75rem; cursor: pointer; }
  .modal-input button:hover { border-color: var(--accent); }
  .modal select { background: var(--bg3); border: 1px solid var(--line); color: var(--fg); border-radius: 4px; padding: .35rem .5rem; font: inherit; font-size: .8rem; }
  .modal-actions { display: flex; gap: .5rem; justify-content: flex-end; margin-top: 1rem; }
  .modal-actions button { background: var(--bg3); color: var(--fg); border: 1px solid var(--line); border-radius: 4px; padding: .4rem 1rem; font-size: .8rem; cursor: pointer; }
  .modal-actions button.primary { background: var(--accent); color: #111; border-color: var(--accent); }
  .modal-actions button.muted { color: var(--dim); }
  .modal-status { color: var(--dim); font-size: .75rem; margin: .6rem 0 0; min-height: 1em; }
  .look-chip { display: inline-flex; align-items: center; gap: .25rem; background: var(--bg3); color: #ccc; border: 1px solid var(--line); border-radius: 3px; padding: 2px 7px; font-size: .7rem; font-family: ui-monospace, monospace; cursor: pointer; }
  .look-chip input { accent-color: var(--accent); margin: 0; }
  .look-chip:has(input:checked) { color: var(--accent); border-color: var(--accent); background: rgba(255,170,85,.05); }

  /* setup splash (full-screen blocker during auto-setup) */
  .splash { position: fixed; inset: 0; background: var(--bg); display: none; align-items: center; justify-content: center; z-index: 100; flex-direction: column; gap: 1.2rem; }
  .splash.show { display: flex; }
  .splash h2 { margin: 0; font-weight: 500; font-size: 1.3rem; letter-spacing: .04em; }
  .splash .msg { color: var(--dim); font-size: .85rem; max-width: 480px; text-align: center; }
  .splash .bar { width: 360px; height: 4px; background: var(--bg2); border-radius: 2px; overflow: hidden; position: relative; }
  .splash .bar > div { height: 100%; background: var(--accent); width: 30%; position: absolute; left: -30%; animation: slide 1.4s ease-in-out infinite; }
  .splash .meta { color: var(--dim); font-size: .7rem; font-variant-numeric: tabular-nums; }
  @keyframes slide { 0% { left: -30%; } 100% { left: 100%; } }
</style>
</head>
<body>

<header>
  <h1>banger</h1>
  <nav>
    <button data-view="library" class="active">Library</button>
    <button data-view="welcome">Quick run</button>
    <button data-view="running" id="nav-running" style="display:none">Running</button>
    <button data-view="results" id="nav-results" style="display:none">Results</button>
    <button data-view="label" id="nav-label" style="display:none">Label</button>
    <button data-view="settings">Settings</button>
  </nav>
</header>

<main>

<section class="view hidden" id="view-welcome">
  <div class="welcome-wrap">
    <h2>Pick a folder, get your bangers</h2>
    <p class="lead">Drop a folder of photos here, or click to browse. Banger scores them, dedups bursts, and shows you the top picks.</p>
    <div class="drop-zone" id="drop-zone">
      <div class="icon">📁</div>
      <div><strong>Open folder</strong></div>
      <div class="hint">click to browse, or drag a folder from your file manager</div>
    </div>
    <div class="settings-row">
      <label class="opt">top<input type="number" id="opt-top-n" min="1" max="100" value="10"></label>
      <label class="opt">strategy
        <select id="opt-strategy">
          <option value="kmeans" selected>kmeans (portfolio)</option>
          <option value="faces">faces (one per person)</option>
          <option value="mmr">mmr (diverse)</option>
          <option value="topk">topk (pure score)</option>
        </select>
      </label>
      <label class="opt"><input type="checkbox" id="opt-recursive" checked>walk subfolders</label>
      <label class="opt"><input type="checkbox" id="opt-face-gate">face-gate</label>
      <label class="opt"><input type="checkbox" id="opt-eye-gate">eye-gate</label>
      <label class="opt"><input type="checkbox" id="opt-xmp">write XMP sidecars</label>
    </div>
    <div class="recent">
      <h3>Recent folders</h3>
      <ul class="recent-list" id="recent-list"></ul>
    </div>
  </div>
</section>

<section class="view hidden" id="view-running">
  <div class="progress-wrap">
    <h2>Working…</h2>
    <div class="stage" id="run-stage">starting</div>
    <div class="bar"><div id="run-bar"></div></div>
    <div class="progress-meta"><span id="run-counts">0 / 0</span><span id="run-elapsed">0.0s</span></div>
    <div class="log" id="run-log"></div>
  </div>
</section>

<section class="view hidden" id="view-results">
  <div class="results-bar">
    <h2>Top picks</h2>
    <div class="meta" id="results-meta"></div>
    <button id="btn-label">Label these</button>
    <button id="btn-rerun">Run again</button>
    <button id="btn-export-bangers" class="primary">Export bangers ↗</button>
  </div>
  <div class="grid" id="results-grid"></div>
</section>

<section class="view" id="view-library" style="padding:0">
  <div class="lib-shell">
    <aside class="lib-sidebar">
      <div class="lib-section">
        <h3>Watched folders</h3>
        <ul class="lib-roots" id="lib-roots"></ul>
        <div class="lib-add">
          <button id="lib-add-root">+ add folder</button>
        </div>
      </div>
      <div class="lib-section" id="lib-folder-section" style="display:none">
        <h3>Subfolders</h3>
        <ul class="lib-folders" id="lib-folders"></ul>
      </div>
      <div class="lib-section">
        <h3>Search</h3>
        <input type="text" id="lib-search" placeholder="tag, name, camera…" autocomplete="off">
      </div>
      <div class="lib-section">
        <h3>Filter</h3>
        <label class="lib-filter-row">camera
          <select id="lib-filter-camera"><option value="">any</option></select>
        </label>
        <label class="lib-filter-row">face
          <select id="lib-filter-face"><option value="">any</option></select>
        </label>
        <label class="lib-filter-row">place
          <select id="lib-filter-place"><option value="">any</option></select>
        </label>
        <label class="lib-filter-row">min star
          <select id="lib-filter-star">
            <option value="">any</option>
            <option value="-5">-5+</option><option value="-3">-3+</option><option value="0">0+</option>
            <option value="1">+1+</option><option value="3">+3+</option><option value="5">+5</option>
          </select>
        </label>
      </div>
    </aside>
    <div class="lib-main">
      <div class="lib-toolbar">
        <h2 id="lib-title">Library</h2>
        <span class="lib-meta" id="lib-meta"></span>
        <span class="lib-meta" id="lib-tagger-meta" style="color:#c9c"></span>
        <button id="lib-import" style="margin-left:auto">Import…</button>
        <button id="lib-scan">Rescan</button>
        <button id="lib-index" title="Pre-compute tags, faces, scenes, scores for everything in the library so culling later is instant">Index all</button>
        <button id="lib-score">Score this view</button>
        <button id="lib-xmp" title="Write ratings + colour labels to XMP sidecars next to your originals (source files untouched). Opens in Lightroom / Bridge / digiKam.">Write ratings to source (XMP)</button>
        <button id="lib-export" class="primary" title="Cull current view + copy top N originals to ~/Pictures/bangers/&lt;timestamp&gt;/">Export bangers ↗</button>
      </div>
      <div class="lib-grid" id="lib-grid">
        <div class="lib-onboarding" id="lib-onboarding">
          <h2>Welcome to your library</h2>
          <p>Point banger at the folders you already keep your photos in, or pull them in from a camera / SD card. We'll index, score, cull, tag, name faces, and write ratings back to your files — nothing leaves your machine.</p>
          <div class="onboarding-actions">
            <button id="onb-add-folder">📁 Open folder</button>
            <button id="onb-import">📷 Import from device</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<div class="modal-backdrop" id="export-modal" style="display:none">
  <div class="modal">
    <h2>Export bangers</h2>
    <p class="modal-hint">
      Culls the current scope and copies the top-N originals (RAW + JPEG) into
      <code>~/Pictures/bangers/&lt;timestamp&gt;[_label]/</code>. No editing,
      no re-encoding — files are copied verbatim.
    </p>
    <div class="modal-row">
      <label>Folder name</label>
      <div class="modal-input">
        <input type="text" id="exp-name" placeholder="(timestamp)">
      </div>
    </div>
    <div class="modal-row">
      <label>How many</label>
      <div class="modal-input">
        <input type="number" id="exp-count" value="10" min="1" max="500" style="width:100px;flex:0 0 100px">
      </div>
    </div>
    <div class="modal-row">
      <label>Preset</label>
      <div class="modal-input">
        <select id="exp-preset" style="flex:1">
          <option value="diverse">Diverse — one per visual cluster (default)</option>
          <option value="best">Best — pure aesthetic ranking</option>
          <option value="mixed">Mixed — balanced score/diversity (MMR λ=0.5)</option>
          <option value="people">People — one good shot per face</option>
          <option value="landscapes">Landscapes — bias toward outdoor / scenery</option>
          <option value="pets">Pets / wildlife — bias toward animals</option>
          <option value="food">Food — bias toward food / drink shots</option>
        </select>
      </div>
    </div>
    <div class="modal-actions">
      <button id="exp-cancel" class="muted">Cancel</button>
      <button id="exp-go" class="primary">Export</button>
    </div>
    <p class="modal-status" id="exp-status"></p>
  </div>
</div>

<div class="modal-backdrop" id="import-modal" style="display:none">
  <div class="modal">
    <h2>Import photos</h2>
    <p class="modal-hint">Copies files from a source folder (or SD card) into a destination, organised the way you choose. The destination is added to your library and scanned automatically.</p>
    <div class="modal-row">
      <label>Source</label>
      <div class="modal-input">
        <input type="text" id="imp-source" placeholder="e.g. E:/DCIM/100MSDCF">
        <button id="imp-pick-source">Browse…</button>
      </div>
    </div>
    <div class="modal-row">
      <label>Destination</label>
      <div class="modal-input">
        <input type="text" id="imp-dest" placeholder="e.g. C:/Users/you/Pictures/2026">
        <button id="imp-pick-dest">Browse…</button>
      </div>
    </div>
    <div class="modal-row">
      <label>Organise as</label>
      <select id="imp-scheme">
        <option value="by_date">By date (YYYY-MM-DD/)</option>
        <option value="by_camera">By camera</option>
        <option value="flat">Flat (no subfolders)</option>
      </select>
    </div>
    <div class="modal-actions">
      <button id="imp-cancel" class="muted">Cancel</button>
      <button id="imp-go" class="primary">Import</button>
    </div>
    <p class="modal-status" id="imp-status"></p>
  </div>
</div>

<section class="view hidden" id="view-label" style="padding:0">
  <iframe id="label-frame" src="about:blank" style="width:100%;height:100%;border:0;background:#0c0c0c;display:block"></iframe>
</section>

<section class="view hidden" id="view-settings">
  <div class="settings-pane">
    <h2>Pipeline levers</h2>
    <div class="settings-card">
      <h3>Cull tuning</h3>
      <p style="margin:.3rem 0 .8rem;color:var(--dim);font-size:.75rem">
        Hover the (i) next to each name for what it does. Changes take effect
        on the next scan / score / export.
      </p>
      <div id="settings-levers"><p class="empty" style="color:var(--dim);font-size:.8rem">loading…</p></div>
      <div style="margin-top:.8rem;display:flex;gap:.5rem;align-items:center">
        <button id="settings-save" class="primary" style="background:var(--accent);color:#000;border:0;border-radius:4px;padding:.4rem .9rem;font-size:.8rem;cursor:pointer">Save</button>
        <button id="settings-reset" style="background:var(--bg3);color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:.4rem .9rem;font-size:.8rem;cursor:pointer">Reset to defaults</button>
        <span id="settings-status" style="color:var(--dim);font-size:.75rem"></span>
      </div>
    </div>
    <h2 style="margin-top:1.5rem">Status</h2>
    <div class="settings-card">
      <h3>Pipeline state</h3>
      <label><span class="k">Taste head</span><span id="st-head" class="badge">…</span></label>
      <label><span class="k">Scene clusters</span><span id="st-scene" class="badge">…</span></label>
      <label><span class="k">Mediapipe (eyes)</span><span id="st-eyes" class="badge">…</span></label>
      <label><span class="k">InsightFace (faces)</span><span id="st-faces" class="badge">…</span></label>
      <label><span class="k">Labels collected</span><span id="st-labels"></span></label>
      <label><span class="k">Embeddings cached</span><span id="st-embs"></span></label>
    </div>
    <div class="settings-card">
      <h3>Face library</h3>
      <p style="margin:.3rem 0;color:var(--dim);font-size:.75rem">
        Names persist across runs. Click a name to rename, × forgets the
        centroid. <strong>Discover</strong> clusters every detected face
        in the library and auto-creates 'Person N' entries for everyone
        not yet named.
      </p>
      <div style="margin:.4rem 0">
        <button id="face-discover" style="background:var(--bg3);color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:.35rem .7rem;font-size:.75rem;cursor:pointer">Discover people</button>
      </div>
      <div class="face-library" id="face-library">
        <p class="empty">loading…</p>
      </div>
    </div>
    <div class="settings-card">
      <h3>Actions</h3>
      <p style="margin:.3rem 0;color:var(--dim);font-size:.8rem">
        Labelling, training, and scene fitting live in the existing CLI for now.
        From a terminal: <code>banger train</code>, <code>banger scenes fit -k 5</code>.
      </p>
    </div>
  </div>
</section>

</main>

<div class="toast" id="toast"></div>

<div id="bg-strip" class="bg-strip" style="display:none">
  <div class="bg-strip-bar"><div id="bg-strip-fill"></div></div>
  <div class="bg-strip-text" id="bg-strip-text">working…</div>
  <button id="bg-strip-expand" title="See full status">expand ↗</button>
</div>

<div class="splash" id="splash">
  <h2>banger</h2>
  <div class="msg" id="splash-msg">starting up…</div>
  <div class="bar"><div></div></div>
  <div class="meta" id="splash-meta"></div>
</div>

<div class="overlay" id="overlay">
  <button class="close" id="overlay-close">close</button>
  <div class="overlay-inner" id="overlay-inner">
    <div class="overlay-image"><img id="overlay-img"></div>
    <aside class="overlay-panel" id="overlay-panel"></aside>
  </div>
  <div class="hint">click outside, or press Esc, to close</div>
</div>

<script>
const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));

let currentView = "welcome";
let currentJob = null;
let pollTimer = null;
let lastResults = null;
let lastInputDir = null;

function show(view) {
  currentView = view;
  $$(".view").forEach(v => v.classList.toggle("hidden", v.id !== "view-" + view));
  $$("header nav button").forEach(b => b.classList.toggle("active", b.dataset.view === view));
}

$$("header nav button").forEach(b => b.addEventListener("click", () => {
  show(b.dataset.view);
  if (b.dataset.view === "settings") { loadFaceLibrary(); loadSettings(); }
  if (b.dataset.view === "library") loadLibrary();
}));

// ===== Settings (pipeline levers) =====
let _settingsFields = {};
async function loadSettings() {
  try {
    const res = await fetch("/api/settings");
    const d = await res.json();
    _settingsFields = d.fields || {};
    renderSettings(d.values || {}, d.fields || {});
  } catch (e) {
    $("#settings-levers").innerHTML = `<p style="color:var(--red)">load failed: ${escapeHtml(e.message)}</p>`;
  }
}
function renderSettings(values, fields) {
  const order = ["sharpness_threshold","face_sharpness_threshold","top_n",
                 "strategy","mmr_diversity","tag_min_sim","face_gate","eye_gate"];
  const host = $("#settings-levers");
  host.innerHTML = order.map(key => {
    const meta = fields[key]; if (!meta) return "";
    const v = values[key];
    const info = `<span class="lever-info" title="${escapeHtml(meta.info || '')}">i</span>`;
    const name = `<span class="lever-name">${escapeHtml(meta.label || key)} ${info}</span>`;
    if (meta.type === "bool") {
      return `<div class="lever-row" data-key="${key}">${name}
        <label style="display:flex;align-items:center;gap:.4rem;font-size:.78rem"><input type="checkbox" ${v ? "checked" : ""}> on</label>
        <span class="lever-val"></span></div>`;
    }
    if (meta.type === "choice") {
      const opts = (meta.choices || []).map(c => `<option value="${escapeHtml(c)}" ${c === v ? "selected" : ""}>${escapeHtml(c)}</option>`).join("");
      return `<div class="lever-row" data-key="${key}">${name}
        <select>${opts}</select><span class="lever-val"></span></div>`;
    }
    const step = meta.step ?? 1;
    const isSlider = (meta.max - meta.min) <= 500;
    if (isSlider) {
      return `<div class="lever-row" data-key="${key}">${name}
        <input type="range" min="${meta.min}" max="${meta.max}" step="${step}" value="${v}">
        <span class="lever-val">${v}</span></div>`;
    }
    return `<div class="lever-row" data-key="${key}">${name}
      <input type="number" min="${meta.min}" max="${meta.max}" step="${step}" value="${v}">
      <span class="lever-val"></span></div>`;
  }).join("");
  host.querySelectorAll(".lever-row input[type=range]").forEach(inp => {
    const val = inp.parentElement.querySelector(".lever-val");
    inp.addEventListener("input", () => { val.textContent = inp.value; });
  });
}
function readSettings() {
  const out = {};
  $$(".lever-row").forEach(row => {
    const key = row.dataset.key;
    const meta = _settingsFields[key] || {};
    if (meta.type === "bool") out[key] = row.querySelector("input[type=checkbox]").checked;
    else if (meta.type === "choice") out[key] = row.querySelector("select").value;
    else out[key] = parseFloat(row.querySelector("input").value);
  });
  return out;
}
document.addEventListener("click", async (e) => {
  if (e.target && e.target.id === "settings-save") {
    e.target.disabled = true;
    $("#settings-status").textContent = "saving…";
    try {
      const res = await fetch("/api/settings", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({updates: readSettings()}),
      });
      if (!res.ok) throw new Error(await res.text());
      $("#settings-status").textContent = "saved";
      setTimeout(() => $("#settings-status").textContent = "", 1800);
    } catch (err) {
      $("#settings-status").textContent = "save failed: " + err.message;
    } finally {
      e.target.disabled = false;
    }
  }
  if (e.target && e.target.id === "settings-reset") {
    if (!confirm("Reset all pipeline levers to defaults?")) return;
    const res = await fetch("/api/settings", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({reset: true}),
    });
    const d = await res.json();
    renderSettings(d.values || {}, _settingsFields);
    $("#settings-status").textContent = "reset to defaults";
    setTimeout(() => $("#settings-status").textContent = "", 1800);
  }
});

function toast(msg, ms=1800) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), ms);
}

function readOpts() {
  return {
    top_n: parseInt($("#opt-top-n").value) || 10,
    strategy: $("#opt-strategy").value,
    recursive: $("#opt-recursive").checked,
    face_gate: $("#opt-face-gate").checked,
    eye_gate: $("#opt-eye-gate").checked,
    write_xmp: $("#opt-xmp").checked,
  };
}

async function startRun(input_dir) {
  lastInputDir = input_dir;
  const body = { input_dir, ...readOpts() };
  const res = await fetch("/api/run", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body) });
  if (!res.ok) { toast(await res.text() || "run failed"); return; }
  const data = await res.json();
  currentJob = data.job_id;
  $("#nav-running").style.display = "";
  $("#run-log").innerHTML = "";
  toast(`Scoring kicked off in the background — see the strip at the bottom`, 2200);
  pollJob();
}

async function pollJob() {
  if (!currentJob) return;
  try {
    const res = await fetch(`/api/jobs/${currentJob}`);
    if (!res.ok) throw new Error(await res.text());
    const j = await res.json();
    renderJob(j);
    updateBgStrip();
    if (j.status === "done" || j.status === "error") {
      clearTimeout(pollTimer);
      if (j.status === "done") {
        lastResults = j;
        $("#nav-results").style.display = "";
        renderResults(j);
        if (_pendingExportOpts !== null) {
          _maybeAutoExport(j);
        } else if (currentView === "library") {
          show("results");
        } else {
          toast("Scoring complete — see Results tab");
        }
      } else {
        toast(j.error || "job errored");
      }
      updateBgStrip();
      return;
    }
  } catch (e) { console.error(e); }
  pollTimer = setTimeout(pollJob, 600);
}

function renderJob(j) {
  _bgJobSnapshot = j;
  $("#run-stage").textContent = j.stage;
  const pct = j.total ? (j.progress / j.total * 100) : 0;
  $("#run-bar").style.width = pct + "%";
  $("#run-counts").textContent = `${j.progress} / ${j.total}`;
  $("#run-elapsed").textContent = `${j.elapsed.toFixed(1)}s`;
  const log = $("#run-log");
  log.innerHTML = j.tail.map(l => `<div class="row">${escapeHtml(l)}</div>`).join("");
  log.scrollTop = log.scrollHeight;
  updateBgStrip();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function starsFromRank(rank, total) {
  if (total <= 0) return 0;
  const pct = (rank - 1) / total;
  if (pct < 0.10) return 5;
  if (pct < 0.30) return 4;
  if (pct < 0.60) return 3;
  if (pct < 0.90) return 2;
  return 1;
}

function renderResults(j) {
  const grid = $("#results-grid");
  const total = j.picks.length;
  $("#results-meta").textContent =
    `${total} picks · ${j.elapsed.toFixed(1)}s` +
    (j.xmp_written ? ` · ${j.xmp_written} XMP written` : "");
  grid.innerHTML = j.picks.map((p, i) => {
    const stars = "★".repeat(starsFromRank(p.rank, total)) + "☆".repeat(5 - starsFromRank(p.rank, total));
    const score = p.aesthetic !== null ? p.aesthetic.toFixed(2) : "—";
    return `
      <div class="card" data-idx="${i}">
        <div class="img-wrap">
          <img loading="lazy" src="/api/thumb/${p.sha}" alt="${escapeHtml(p.display)}">
          <div class="rank">#${p.rank}</div>
          <div class="stars">${stars}</div>
        </div>
        <div class="meta">
          <span class="stem" title="${escapeHtml(p.display)}">${escapeHtml(p.stem)}</span>
          <span class="score">${score}</span>
        </div>
        <div style="padding:0 .55rem .55rem">
          ${p.scene_preset ? `<span class="scene">${escapeHtml(p.scene_preset)}</span>` : ""}
        </div>
      </div>`;
  }).join("");
  grid.querySelectorAll(".card").forEach(card => {
    const idx = parseInt(card.dataset.idx);
    card.addEventListener("click", () => openHero(j.picks[idx]));
  });
}

async function openHero(pick) {
  overlaySha = pick.sha;
  $("#overlay-img").src = "/api/preview/" + pick.sha;
  $("#overlay-panel").innerHTML = renderPanelLoading(pick);
  $("#overlay").classList.add("show");
  let details = null;
  try {
    const res = await fetch("/api/details/" + pick.sha);
    if (!res.ok) throw new Error(await res.text());
    details = await res.json();
    $("#overlay-panel").innerHTML = renderPanel(pick, details);
    updateLabelChips(pick.sha, details.label);
  } catch (e) {
    $("#overlay-panel").innerHTML = renderPanelLoading(pick) +
      `<p class="empty">details fetch failed: ${escapeHtml(e.message)}</p>`;
    return;
  }
  // If no cached tags, fetch them lazily (fast: just a matmul + top-K).
  if (!details.tags) {
    try {
      const res = await fetch("/api/tags/" + pick.sha);
      const d = await res.json();
      const slot = document.getElementById("tags-slot-" + pick.sha);
      if (slot) slot.innerHTML = renderTagChips(d.tags || []);
    } catch (e) {
      const slot = document.getElementById("tags-slot-" + pick.sha);
      if (slot) slot.innerHTML = '<p class="empty">tag fetch failed</p>';
    }
  }

  // Lazy-fetch faces (with name matching against the persistent DB).
  try {
    const res = await fetch("/api/faces/" + pick.sha);
    const d = await res.json();
    const slot = document.getElementById("faces-slot-" + pick.sha);
    if (slot) slot.innerHTML = renderFaceRow(pick.sha, d.faces || []);
    if (slot) wireFaceTiles(pick.sha);
  } catch (e) {
    const slot = document.getElementById("faces-slot-" + pick.sha);
    if (slot) slot.innerHTML = '<p class="empty">faces unavailable</p>';
  }
}

function renderFaceRow(sha, faces) {
  if (!faces.length) return '<p class="empty">no faces detected</p>';
  return '<div class="face-row">' + faces.map(f => {
    // Every face name is clickable; matched ones come pre-filled so the user
    // can correct misidentifications.
    const cls = f.matched_name ? "name editable" : "name unnamed editable";
    const text = f.matched_name || "name…";
    const title = f.matched_name
      ? `matched at sim ${f.match_sim} (click to rename / correct)`
      : "click to name this face";
    return `
      <div class="face-tile" data-face-idx="${f.face_idx}">
        <img src="/api/face-thumb/${sha}/${f.face_idx}" alt="face">
        <span class="${cls}" data-current="${escapeHtml(f.matched_name || '')}" title="${title}">${escapeHtml(text)}</span>
      </div>`;
  }).join("") + '</div>';
}

function wireFaceTiles(sha) {
  document.querySelectorAll(`#faces-slot-${sha} .name.editable`).forEach(el => {
    el.addEventListener("click", () => beginFaceEdit(sha, el));
  });
}

function beginFaceEdit(sha, el) {
  const tile = el.closest(".face-tile");
  const faceIdx = parseInt(tile.dataset.faceIdx);
  const current = el.dataset.current || "";
  el.outerHTML = `<input type="text" placeholder="name" value="${escapeHtml(current)}" data-face-idx="${faceIdx}">`;
  const input = tile.querySelector("input");
  input.focus();
  input.select();
  const cancel = () => {
    input.outerHTML = current
      ? `<span class="name editable" data-current="${escapeHtml(current)}">${escapeHtml(current)}</span>`
      : `<span class="name unnamed editable" data-current="">name…</span>`;
    wireFaceTiles(sha);
  };
  input.addEventListener("blur", cancel);
  input.addEventListener("keydown", async (e) => {
    if (e.key === "Escape") { input.removeEventListener("blur", cancel); cancel(); return; }
    if (e.key !== "Enter") return;
    const name = input.value.trim();
    if (!name) { input.removeEventListener("blur", cancel); cancel(); return; }
    input.removeEventListener("blur", cancel);
    try {
      const res = await fetch("/api/face/name", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({sha, face_idx: faceIdx, name}),
      });
      if (!res.ok) throw new Error(await res.text());
      input.outerHTML = `<span class="name editable" data-current="${escapeHtml(name)}">${escapeHtml(name)}</span>`;
      wireFaceTiles(sha);
      const msg = current && current !== name
        ? `Re-tagged "${current}" → "${name}"`
        : `Named "${name}"`;
      toast(msg, 1800);
    } catch (e) {
      toast("name save failed: " + e.message);
    }
  });
}

function renderTagChips(pairs) {
  if (!pairs || !pairs.length) return '<p class="empty">no confident tags</p>';
  return '<div class="tag-chips">' + pairs.map(([t, s]) =>
    `<span class="tag-chip" title="cosine ${s.toFixed(3)}">${escapeHtml(t)}</span>`
  ).join("") + '</div>';
}

function bar(label, value, max=10) {
  if (value === null || value === undefined) return "";
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return `<div class="bar-row"><span class="label">${label}</span>` +
         `<span class="meter"><div style="width:${pct}%"></div></span>` +
         `<span class="val">${Number(value).toFixed(1)}</span></div>`;
}

function renderPanelLoading(pick) {
  const total = lastResults ? lastResults.picks.length : 10;
  const stars = "★".repeat(starsFromRank(pick.rank, total)) + "☆".repeat(5 - starsFromRank(pick.rank, total));
  const score = pick.aesthetic !== null ? pick.aesthetic.toFixed(2) : "—";
  return `
    <div class="head">
      <span class="rank">#${pick.rank}</span>
      <span class="stars">${stars}</span>
      <span class="stem">${escapeHtml(pick.display)}</span>
    </div>
    <h3>Why it was picked</h3>
    <dl>
      <dt>aesthetic</dt><dd>${score} <span style="color:var(--dim);font-size:.7rem">(${escapeHtml(pick.aesthetic_source || "—")})</span></dd>
      <dt>sharpness</dt><dd>${pick.sharpness.toFixed(0)}</dd>
      <dt>scene</dt><dd>${escapeHtml(pick.scene_preset || "—")}</dd>
      <dt>file kind</dt><dd>${escapeHtml(pick.kind)}</dd>
    </dl>
    <p class="empty">loading metrics + EXIF…</p>`;
}

function renderPanel(pick, d) {
  const total = lastResults ? lastResults.picks.length : 10;
  const stars = "★".repeat(starsFromRank(pick.rank, total)) + "☆".repeat(5 - starsFromRank(pick.rank, total));
  const score = pick.aesthetic !== null ? pick.aesthetic.toFixed(2) : "—";

  // Quality bars from the cv2 metrics dict.
  const m = d.metrics || {};
  const dims = ["exposure", "contrast", "color_harmony", "composition", "leading_lines"];
  const bars = dims.map(k => bar(k.replace("_", " "), m[k])).join("");

  // Flags from exposure analysis.
  const flagList = [];
  if (m.is_silhouette) flagList.push('<span class="flag off">silhouette</span>');
  if (m.shadow_clipped) flagList.push('<span class="flag on">shadows clipped</span>');
  if (m.highlight_clipped) flagList.push('<span class="flag on">highlights clipped</span>');
  if (m.is_monochrome) flagList.push('<span class="flag off">monochrome</span>');
  const flags = flagList.length ? `<div style="margin:.4rem 0">${flagList.join("")}</div>` : "";

  // Extra numeric fields below the bars.
  const extra = [];
  if (m.noise !== undefined) extra.push(["noise", m.noise.toFixed(1) + " σ"]);
  if (m.dynamic_range !== undefined) extra.push(["dynamic range", m.dynamic_range.toFixed(1) + " stops"]);
  if (m.mean_luminance !== undefined) extra.push(["mean luminance", m.mean_luminance.toFixed(2)]);
  const extraHtml = extra.length
    ? `<dl>${extra.map(([k,v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`
    : "";

  // Eye-region sharpness ("are the eyes tack-sharp"): null = no face found,
  // a number = eye-region Laplacian variance. Render "no face" for null so the
  // empty result is legible rather than a bare dash.
  const eyesSharpText = (d.eye_sharpness === null || d.eye_sharpness === undefined)
    ? "no face"
    : d.eye_sharpness.toFixed(0);

  // Face / eyes block. Names + thumbs get loaded async after openHero. We show
  // the block whenever we have a face count OR a computed eye-sharpness value
  // so the "eyes sharpness" metric is always surfaced.
  let faceBlock = "";
  if (d.face_count || d.eye_sharpness_present) {
    const faceRows = [
      ["faces detected", d.face_count || 0],
      ["face sharpness", d.face_sharpness !== null && d.face_sharpness !== undefined ? d.face_sharpness.toFixed(0) : "—"],
      ["eyes sharpness", eyesSharpText],
    ];
    if (d.eyes) {
      faceRows.push(["min EAR", d.eyes.ear_min !== undefined ? d.eyes.ear_min.toFixed(3) : "—"]);
      faceRows.push(["blink detected", d.eyes.any_blink ? "yes" : "no"]);
    }
    faceBlock = `
      <h3>Faces</h3>
      <dl>${faceRows.map(([k,v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>
      ${d.face_count ? `<div id="faces-slot-${pick.sha}"><p class="empty">loading faces…</p></div>` : ""}`;
  }

  // EXIF block.
  const exif = d.exif || {};
  const exifPairs = [];
  if (exif.camera_make || exif.camera_model) {
    exifPairs.push(["camera", [exif.camera_make, exif.camera_model].filter(Boolean).join(" ")]);
  }
  if (exif.lens_model) exifPairs.push(["lens", exif.lens_model]);
  if (exif.aperture) exifPairs.push(["aperture", exif.aperture]);
  if (exif.shutter) exifPairs.push(["shutter", exif.shutter]);
  if (exif.iso) exifPairs.push(["ISO", exif.iso]);
  if (exif.focal_length_mm) exifPairs.push(["focal length", exif.focal_length_mm + (exif.focal_length_35mm ? ` (≈${exif.focal_length_35mm}mm FF)` : "")]);
  if (exif.exposure_bias !== undefined && exif.exposure_bias !== null) exifPairs.push(["exposure comp", `${exif.exposure_bias > 0 ? "+" : ""}${exif.exposure_bias.toFixed(1)} EV`]);
  if (exif.date_taken) exifPairs.push(["taken", exif.date_taken]);
  if (exif.gps_lat !== undefined && exif.gps_lat !== null) {
    const placeText = exif.place
      ? exif.place + ` (${exif.gps_lat.toFixed(4)}, ${exif.gps_lon.toFixed(4)})`
      : `${exif.gps_lat.toFixed(4)}, ${exif.gps_lon.toFixed(4)}`;
    exifPairs.push(["location", placeText]);
  }
  if (exif.metering_mode) exifPairs.push(["metering", exif.metering_mode]);
  if (exif.flash) exifPairs.push(["flash", exif.flash]);
  if (exif.pixel_x && exif.pixel_y) exifPairs.push(["dimensions", `${exif.pixel_x} × ${exif.pixel_y}`]);

  const exifHtml = exifPairs.length
    ? `<dl>${exifPairs.map(([k,v]) => `<dt>${k}</dt><dd>${escapeHtml(String(v))}</dd>`).join("")}</dl>`
    : '<p class="empty">no EXIF data (or non-JPEG source)</p>';

  const tagsHtml = d.tags && d.tags.length
    ? renderTagChips(d.tags)
    : `<div id="tags-slot-${pick.sha}"><p class="empty">loading tags…</p></div>`;

  const ratingText = (d.label === null || d.label === undefined)
    ? "no rating" : `rating ${d.label >= 0 ? "+" : ""}${d.label}`;
  return `
    <div class="head">
      <span class="rank">#${pick.rank}</span>
      <span class="stars">${stars}</span>
      <span class="stem">${escapeHtml(pick.display)}</span>
    </div>
    <div class="rating-bar">
      <span id="overlay-rating" class="overlay-rating">${ratingText}</span>
      <span class="rating-hint">1–5 / +/- rate · Bksp clear · ←→ navigate</span>
    </div>
    <h3>Tags</h3>
    ${tagsHtml}
    <h3>Why it was picked</h3>
    <dl>
      <dt>aesthetic</dt><dd>${score} <span style="color:var(--dim);font-size:.7rem">(${escapeHtml(pick.aesthetic_source || "—")})</span></dd>
      <dt>sharpness</dt><dd>${pick.sharpness.toFixed(0)} (global), ${d.face_sharpness ? d.face_sharpness.toFixed(0) + " face" : "no face"}</dd>
      <dt>scene</dt><dd>${escapeHtml(pick.scene_preset || "—")}</dd>
      <dt>file</dt><dd>${escapeHtml(pick.kind)}</dd>
    </dl>
    <h3>Quality dims</h3>
    ${bars || '<p class="empty">metrics not computed</p>'}
    ${flags}
    ${extraHtml}
    ${faceBlock}
    <h3>EXIF</h3>
    ${exifHtml}
    <h3>Source</h3>
    <dl>
      <dt>path</dt><dd style="font-size:.7rem">${escapeHtml(d.src_path)}</dd>
      <dt>sha</dt><dd style="font-size:.7rem">${escapeHtml(pick.sha.slice(0, 16))}…</dd>
    </dl>`;
}

function closeOverlay() { $("#overlay").classList.remove("show"); overlaySha = null; }

$("#overlay-close").addEventListener("click", closeOverlay);
// Click on the overlay backdrop closes; clicks inside .overlay-inner don't.
$("#overlay").addEventListener("click", e => {
  if (e.target === $("#overlay")) closeOverlay();
});

// Build a minimal "pick" from a Library grid cell so the detail overlay can be
// opened by keyboard navigation (mirrors the click handler in refreshLibraryGrid).
function _pickFromCell(cell) {
  const display = cell.dataset.display || "";
  return {
    sha: cell.dataset.sha, rank: 0,
    stem: display.split("/").pop().replace(/\.[^.]+$/, ""),
    subdir: display.includes("/") ? display.substring(0, display.lastIndexOf("/")) : "",
    display, kind: "jpeg", sharpness: 0, aesthetic: null, aesthetic_source: null,
    scene_preset: null,
  };
}

// In-overlay navigation: move the underlying grid focus and re-open the hero.
function overlayNavigate(delta) {
  const cells = _libCells();
  if (!cells.length) return;
  setLibFocus(libFocusIdx + delta);
  const cell = cells[libFocusIdx];
  if (cell) openHero(_pickFromCell(cell));
}

// Is the library tab the active view? Keyboard culling only applies there
// (currentView is the SPA's view switch, set by show()).
function _libraryActive() { return currentView === "library"; }

document.addEventListener("keydown", e => {
  // Never hijack typing in inputs / textareas / contenteditable.
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) {
    if (e.key === "Escape" && overlaySha) closeOverlay();
    return;
  }

  // Undo works globally (Cmd/Ctrl+Z) as long as we have a stack.
  if ((e.metaKey || e.ctrlKey) && (e.key === "z" || e.key === "Z")) {
    e.preventDefault();
    undoLastLabel();
    return;
  }

  if (e.key === "Escape") { if (overlaySha) closeOverlay(); return; }

  const overlayOpen = !!overlaySha;
  // Culling shortcuts only apply in the Library tab (grid) or its overlay.
  if (!overlayOpen && !_libraryActive()) return;

  const sha = activeSha();

  // Navigation. In the overlay, arrows step prev/next; in the grid, arrows move
  // the focus (left/right by one, up/down by a row). Space advances.
  if (e.key === "ArrowLeft") {
    e.preventDefault();
    if (overlayOpen) overlayNavigate(-1); else setLibFocus(libFocusIdx - 1);
    return;
  }
  if (e.key === "ArrowRight" || e.key === " " || e.key === "Spacebar") {
    e.preventDefault();
    if (overlayOpen) overlayNavigate(1); else setLibFocus(libFocusIdx + 1);
    return;
  }
  if (e.key === "ArrowUp") {
    e.preventDefault();
    if (overlayOpen) overlayNavigate(-1); else setLibFocus(libFocusIdx - _libCols());
    return;
  }
  if (e.key === "ArrowDown") {
    e.preventDefault();
    if (overlayOpen) overlayNavigate(1); else setLibFocus(libFocusIdx + _libCols());
    return;
  }
  // Enter opens the focused cell in the grid.
  if (e.key === "Enter" && !overlayOpen) {
    const cell = _libCells()[libFocusIdx];
    if (cell) { e.preventDefault(); openHero(_pickFromCell(cell)); }
    return;
  }

  if (!sha) return;

  // Number keys 1–5 set a star-like positive taste label; P = pick (+5),
  // X = reject (-5); +/- nudge; Backspace / Delete / 0 clears.
  if (e.key >= "1" && e.key <= "5") {
    e.preventDefault();
    applyLabel(sha, parseInt(e.key, 10));
    return;
  }
  if (e.key === "p" || e.key === "P") { e.preventDefault(); applyLabel(sha, 5); return; }
  if (e.key === "x" || e.key === "X") { e.preventDefault(); applyLabel(sha, -5); return; }
  if (e.key === "+" || e.key === "=") {
    e.preventDefault();
    const cur = _currentLabelForSha(sha) || 0;
    applyLabel(sha, Math.min(5, cur + 1));
    return;
  }
  if (e.key === "-" || e.key === "_") {
    e.preventDefault();
    const cur = _currentLabelForSha(sha) || 0;
    applyLabel(sha, Math.max(-5, cur - 1));
    return;
  }
  if (e.key === "0" || e.key === "Backspace" || e.key === "Delete") {
    e.preventDefault();
    clearLabel(sha);
    return;
  }
});

$("#btn-rerun").addEventListener("click", () => show("welcome"));
$("#btn-export-bangers").addEventListener("click", () => {
  if (!lastResults || !lastResults.picks.length) { toast("Nothing to export yet"); return; }
  openExportModal({source: "results"});
});
$("#btn-label").addEventListener("click", async () => {
  if (!lastResults) return;
  // We spawn (or reuse) a `banger ui <folder>` subprocess; once it's up,
  // embed it in the iframe. First call costs a few seconds while frames
  // are hashed; subsequent ones reuse the running child.
  toast("Opening labelling UI…", 1200);
  try {
    const res = await fetch("/api/label-spawn", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({input_dir: lastResults.input_dir || lastInputDir})
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    $("#label-frame").src = data.url;
    $("#nav-label").style.display = "";
    show("label");
  } catch (e) {
    toast("Label spawn failed: " + e.message);
  }
});

// Drop zone
const dz = $("#drop-zone");
dz.addEventListener("click", async () => {
  const res = await fetch("/api/folder-pick", { method: "POST" });
  const data = await res.json();
  if (data.path) startRun(data.path);
  else if (data.reason && data.reason !== "cancelled") toast("Folder pick unavailable: " + data.reason);
});
dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("dragover"); });
dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
dz.addEventListener("drop", e => {
  e.preventDefault();
  dz.classList.remove("dragover");
  // Webview can give us a real path via DataTransfer; if not, ask user to click.
  const file = e.dataTransfer.files[0];
  if (file && file.path) startRun(file.path);
  else if (e.dataTransfer.items && e.dataTransfer.items[0]) {
    const it = e.dataTransfer.items[0];
    if (it.kind === "file") {
      const f = it.getAsFile();
      if (f && f.path) startRun(f.path);
      else toast("Drop didn't expose a path; please click to browse.");
    }
  } else {
    toast("Drop didn't expose a path; please click to browse.");
  }
});

async function loadRecent() {
  const res = await fetch("/api/recent-folders");
  const data = await res.json();
  const ul = $("#recent-list");
  if (!data.folders.length) { ul.innerHTML = '<li style="color:var(--dim);cursor:default">no recent folders yet</li>'; return; }
  ul.innerHTML = data.folders.map(f => `<li data-path="${escapeHtml(f)}"><span>${escapeHtml(f)}</span><span class="go">↵ run</span></li>`).join("");
  ul.querySelectorAll("li[data-path]").forEach(li => li.addEventListener("click", () => startRun(li.dataset.path)));
}

// ===== Library tab =====
let libState = {
  activeRootId: null,
  activeSubdir: null,
  cameraFilter: "",
  faceFilter: "",
  starFilter: "",
  placeFilter: "",
  searchQuery: "",
  // Pagination: we fetch a page at a time and append (infinite scroll), so a
  // big library renders incrementally instead of jamming 500+ <img> into the
  // DOM at once.
  loaded: 0,      // rows currently rendered
  total: 0,       // total matching (from the server, over the FULL filter set)
  loading: false, // a page fetch is in flight
};
const LIB_PAGE = 200;
let _libObserver = null;
let libSearchTimer = null;
let taggerPollTimer = null;

// ----- In-app rating / keyboard culling / undo -----
// Focused cell index within the current Library grid (for arrow-key nav).
let libFocusIdx = -1;
// Sha of the frame currently open in the detail overlay (null = grid focus).
let overlaySha = null;
// Small client-side undo stack of label mutations: each entry is
// {sha, prev, next} where prev/next are score|null (null = no label).
let _undoStack = [];
const UNDO_MAX = 50;

function pushUndo(sha, prev, next) {
  _undoStack.push({sha, prev, next});
  if (_undoStack.length > UNDO_MAX) _undoStack.shift();
}

// Update every visible chip for a sha (grid corner + overlay header) and the
// cell's data attribute so re-renders and undo stay in sync.
function updateLabelChips(sha, score) {
  document.querySelectorAll(`.lib-cell[data-sha="${sha}"]`).forEach(cell => {
    if (score === null || score === undefined) delete cell.dataset.label;
    else cell.dataset.label = String(score);
    const labelEl = cell.querySelector(".lib-label");
    if (!labelEl) return;
    const existing = labelEl.querySelector(".lib-rating");
    if (score === null || score === undefined) {
      if (existing) existing.remove();
    } else {
      const txt = `${score >= 0 ? "+" : ""}${score}`;
      if (existing) existing.textContent = txt;
      else labelEl.insertAdjacentHTML("afterbegin", `<span class="lib-rating">${txt}</span>`);
    }
  });
  const ov = document.getElementById("overlay-rating");
  if (ov && overlaySha === sha) {
    ov.textContent = (score === null || score === undefined)
      ? "no rating" : `rating ${score >= 0 ? "+" : ""}${score}`;
  }
}

function _currentLabelForSha(sha) {
  const cell = document.querySelector(`.lib-cell[data-sha="${sha}"]`);
  if (cell && cell.dataset.label !== undefined && cell.dataset.label !== "")
    return parseInt(cell.dataset.label, 10);
  return null;
}

// Core mutators. recordUndo=false is used when replaying an undo so we don't
// push the revert back onto the stack.
async function applyLabel(sha, score, {recordUndo = true} = {}) {
  if (!sha) return;
  const prev = _currentLabelForSha(sha);
  try {
    const res = await fetch("/api/label", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({sha, score}),
    });
    if (!res.ok) { toast("rating failed: " + await res.text()); return; }
    updateLabelChips(sha, score);
    if (recordUndo) pushUndo(sha, prev, score);
    toast(`rated ${score >= 0 ? "+" : ""}${score}`, 900);
  } catch (e) { toast("rating failed: " + e.message); }
}

async function clearLabel(sha, {recordUndo = true} = {}) {
  if (!sha) return;
  const prev = _currentLabelForSha(sha);
  try {
    const res = await fetch("/api/label", {
      method: "DELETE", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({sha}),
    });
    if (!res.ok) { toast("clear failed: " + await res.text()); return; }
    updateLabelChips(sha, null);
    if (recordUndo) pushUndo(sha, prev, null);
    toast("rating cleared", 900);
  } catch (e) { toast("clear failed: " + e.message); }
}

async function undoLastLabel() {
  const entry = _undoStack.pop();
  if (!entry) { toast("nothing to undo", 900); return; }
  // Revert to entry.prev without re-recording.
  if (entry.prev === null || entry.prev === undefined) {
    await clearLabel(entry.sha, {recordUndo: false});
  } else {
    await applyLabel(entry.sha, entry.prev, {recordUndo: false});
  }
  toast("undid rating change", 1100);
}

function _libCells() { return Array.from(document.querySelectorAll("#lib-grid .lib-cell")); }

function setLibFocus(idx) {
  const cells = _libCells();
  if (!cells.length) { libFocusIdx = -1; return; }
  idx = Math.max(0, Math.min(idx, cells.length - 1));
  cells.forEach(c => c.classList.remove("focused"));
  cells[idx].classList.add("focused");
  cells[idx].scrollIntoView({block: "nearest"});
  libFocusIdx = idx;
  // Pull the next page when keyboard nav approaches the end of what's loaded.
  if (typeof libState !== "undefined" && idx >= cells.length - 5
      && libState.loaded < libState.total) {
    loadMoreLibrary(false);
  }
}

function _focusedSha() {
  const cells = _libCells();
  if (libFocusIdx < 0 || libFocusIdx >= cells.length) return null;
  return cells[libFocusIdx].dataset.sha;
}

// The sha the keyboard acts on: overlay frame if open, else focused grid cell.
function activeSha() { return overlaySha || _focusedSha(); }

// Number of grid columns, to make up/down arrows move a row at a time.
function _libCols() {
  const grid = $("#lib-grid");
  const cells = _libCells();
  if (!grid || cells.length < 2) return 1;
  const top = cells[0].offsetTop;
  let cols = 0;
  for (const c of cells) { if (c.offsetTop !== top) break; cols++; }
  return Math.max(1, cols);
}

async function loadLibrary() {
  await refreshLibraryRoots();
  await refreshLibraryFaces();
  await refreshLibraryCameras();
  await refreshLibraryPlaces();
  await refreshLibraryGrid();
  pollTaggerStatus();
}

async function refreshLibraryPlaces() {
  try {
    const res = await fetch("/api/library/places");
    const d = await res.json();
    const sel = $("#lib-filter-place");
    sel.innerHTML = '<option value="">any</option>' + (d.places || []).map(p =>
      `<option value="${escapeHtml(p.name)}">${escapeHtml(p.name)} (${p.count})</option>`
    ).join("");
    sel.value = libState.placeFilter;
  } catch (e) { console.error("places:", e); }
}

let _lastTaggerPhase = null;
let _taggerSnapshot = {phase: "idle", processed: 0, total: 0, tagged: 0};

async function pollTaggerStatus() {
  try {
    const res = await fetch("/api/tagger/status");
    const d = await res.json();
    _taggerSnapshot = d;
    const taggerMeta = $("#lib-tagger-meta");
    if (d.phase === "running" && d.total > 0) {
      const pct = ((d.processed / d.total) * 100).toFixed(0);
      taggerMeta.textContent = `tagging ${d.processed}/${d.total} (${pct}%)`;
    } else {
      taggerMeta.textContent = "";
    }
    if (d.phase === "done" && _lastTaggerPhase === "running") {
      refreshLibraryGrid();
    }
    _lastTaggerPhase = d.phase;
    updateBgStrip();
    const delay = d.phase === "running" ? 2000 : 5000;
    taggerPollTimer = setTimeout(() => { taggerPollTimer = null; pollTaggerStatus(); }, delay);
  } catch (e) {
    taggerPollTimer = setTimeout(() => { taggerPollTimer = null; pollTaggerStatus(); }, 5000);
  }
}

// Bottom strip combines tagger + active scoring job. Hides when both idle.
let _bgJobSnapshot = null;  // last seen renderJob payload
function updateBgStrip() {
  const strip = $("#bg-strip");
  const text = $("#bg-strip-text");
  const fill = $("#bg-strip-fill");
  const tagger = _taggerSnapshot;
  const job = _bgJobSnapshot;
  const jobRunning = job && (job.status === "running" || job.status === "queued");
  const tagging = tagger && tagger.phase === "running" && tagger.total > 0;
  if (!jobRunning && !tagging) {
    strip.style.display = "none";
    return;
  }
  strip.style.display = "flex";
  let parts = [];
  let pct = 0;
  if (jobRunning) {
    const jp = job.total ? (job.progress / job.total) * 100 : 0;
    parts.push(`scoring · ${job.stage} · ${job.progress}/${job.total}`);
    pct = Math.max(pct, jp);
  }
  if (tagging) {
    const tp = (tagger.processed / tagger.total) * 100;
    const label = (tagger.mode === "full") ? "indexing" : "tagging";
    parts.push(`${label} · ${tagger.processed}/${tagger.total}`);
    pct = Math.max(pct, tp);
  }
  text.textContent = parts.join(" · ");
  fill.style.width = pct.toFixed(0) + "%";
}

$("#bg-strip-expand").addEventListener("click", () => {
  if (_bgJobSnapshot && (_bgJobSnapshot.status === "running" || _bgJobSnapshot.status === "queued")) {
    show("running");
  } else {
    show("settings");  // tagger-only: settings has status badges
  }
});

async function refreshLibraryRoots() {
  try {
    const res = await fetch("/api/library/roots");
    const d = await res.json();
    const ul = $("#lib-roots");
    if (!d.roots.length) {
      ul.innerHTML = '<li style="color:var(--dim);cursor:default;font-style:italic">no folders yet</li>';
      libState.activeRootId = null;
    } else {
      ul.innerHTML = d.roots.map(r => `
        <li data-root-id="${r.id}" class="${libState.activeRootId === r.id ? 'active' : ''}" title="${escapeHtml(r.path)}">
          <span>${escapeHtml(r.label)}</span>
          <span class="count">${r.frame_count}<span class="remove" data-root-id="${r.id}" title="remove">×</span></span>
        </li>
      `).join("");
      ul.querySelectorAll("li[data-root-id]").forEach(li => {
        li.addEventListener("click", e => {
          if (e.target.classList.contains("remove")) return;
          libState.activeRootId = parseInt(li.dataset.rootId);
          libState.activeSubdir = null;
          refreshLibraryRoots();
          refreshLibraryFolders();
          refreshLibraryGrid();
        });
        li.querySelector(".remove").addEventListener("click", async e => {
          e.stopPropagation();
          const id = parseInt(e.target.dataset.rootId);
          if (!confirm("Stop watching this folder? (files stay on disk)")) return;
          await fetch(`/api/library/roots/${id}`, {method: "DELETE"});
          if (libState.activeRootId === id) libState.activeRootId = null;
          loadLibrary();
        });
      });
      if (libState.activeRootId === null) {
        libState.activeRootId = d.roots[0].id;
        refreshLibraryFolders();
      }
    }
  } catch (e) { console.error("roots:", e); }
}

async function refreshLibraryFolders() {
  const section = $("#lib-folder-section");
  if (libState.activeRootId === null) { section.style.display = "none"; return; }
  try {
    const res = await fetch("/api/library/folders/" + libState.activeRootId);
    const d = await res.json();
    if (!d.folders.length) { section.style.display = "none"; return; }
    section.style.display = "";
    const ul = $("#lib-folders");
    ul.innerHTML = '<li data-subdir="" class="' + (libState.activeSubdir === null ? 'active' : '') + '"><span>(all)</span></li>' +
      d.folders.filter(f => f.name !== "(root)").map(f => `
        <li data-subdir="${escapeHtml(f.name)}" class="${libState.activeSubdir === f.name ? 'active' : ''}">
          <span>${escapeHtml(f.name)}</span><span class="count">${f.count}</span>
        </li>
      `).join("");
    ul.querySelectorAll("li[data-subdir]").forEach(li => {
      li.addEventListener("click", () => {
        libState.activeSubdir = li.dataset.subdir || null;
        refreshLibraryFolders();
        refreshLibraryGrid();
      });
    });
  } catch (e) { console.error("folders:", e); }
}

async function refreshLibraryCameras() {
  try {
    const res = await fetch("/api/library/cameras");
    const d = await res.json();
    const sel = $("#lib-filter-camera");
    sel.innerHTML = '<option value="">any</option>' + d.cameras.map(c =>
      `<option value="${escapeHtml(c.model)}">${escapeHtml(c.model)} (${c.count})</option>`
    ).join("");
    sel.value = libState.cameraFilter;
  } catch (e) { console.error("cameras:", e); }
}

async function refreshLibraryFaces() {
  try {
    const res = await fetch("/api/face/library");
    const d = await res.json();
    const sel = $("#lib-filter-face");
    sel.innerHTML = '<option value="">any</option>' + (d.names || []).map(n =>
      `<option value="${escapeHtml(n.name)}">${escapeHtml(n.name)}</option>`
    ).join("");
    sel.value = libState.faceFilter;
  } catch (e) { console.error("faces:", e); }
}

async function refreshLibraryGrid() {
  const grid = $("#lib-grid");
  const meta = $("#lib-meta");
  if (libState.activeRootId === null) {
    grid.innerHTML = `
      <div class="lib-onboarding">
        <h2>Welcome to your library</h2>
        <p>Point banger at the folders you already keep your photos in, or pull them in from a camera / SD card. We'll index, score, cull, tag, name faces, and write ratings back to your files — nothing leaves your machine.</p>
        <div class="onboarding-actions">
          <button onclick="document.getElementById('lib-add-root').click()">📁 Open folder</button>
          <button onclick="openImporter()">📷 Import from device</button>
        </div>
      </div>`;
    meta.textContent = "";
    return;
  }
  // Fresh query → reset pagination and load the first page.
  libState.loaded = 0;
  libState.total = 0;
  libState.loading = false;
  libFocusIdx = -1;
  if (_libObserver) { _libObserver.disconnect(); _libObserver = null; }
  grid.innerHTML = "";
  meta.textContent = "loading…";
  await loadMoreLibrary(true);
}

// Build the query params for the current filter scope, with a paging window.
function _libFrameParams(offset, limit) {
  const params = new URLSearchParams();
  params.set("root_id", libState.activeRootId);
  if (libState.activeSubdir) params.set("subdir", libState.activeSubdir);
  if (libState.cameraFilter) params.set("camera", libState.cameraFilter);
  if (libState.faceFilter) params.set("face", libState.faceFilter);
  if (libState.starFilter !== "") params.set("min_score", libState.starFilter);
  if (libState.placeFilter) params.set("place", libState.placeFilter);
  if (libState.searchQuery) params.set("q", libState.searchQuery);
  params.set("offset", String(offset));
  params.set("limit", String(limit));
  return params;
}

// Render one cell's HTML. idx is its absolute position in the loaded grid so
// keyboard nav (which indexes _libCells()) stays in lockstep.
function _libCellHtml(f, idx) {
  const hasLabel = (f.label !== null && f.label !== undefined);
  const ratingChip = hasLabel
    ? `<span class="lib-rating">${f.label >= 0 ? '+' : ''}${f.label}</span>` : "";
  const badges = [];
  if (f.scored) badges.push('<span class="lib-badge scored">S</span>');
  if (f.has_face_data) badges.push('<span class="lib-badge face">F</span>');
  return `
    <div class="lib-cell" data-sha="${f.sha}" data-idx="${idx}" data-display="${escapeHtml(f.rel_path)}"${hasLabel ? ` data-label="${f.label}"` : ""}>
      <div class="lib-img-wrap">
        <img loading="lazy" src="/api/thumb/${f.sha}" alt="${escapeHtml(f.rel_path)}">
        ${badges.length ? `<div class="lib-badges">${badges.join("")}</div>` : ""}
      </div>
      <div class="lib-label">${ratingChip}${escapeHtml(f.stem)}</div>
    </div>`;
}

function _wireLibCell(cell) {
  const sha = cell.dataset.sha;
  const display = cell.dataset.display;
  cell.addEventListener("click", () => {
    setLibFocus(parseInt(cell.dataset.idx, 10));
    const pick = {
      sha, rank: 0, stem: display.split("/").pop().replace(/\.[^.]+$/, ""),
      subdir: display.includes("/") ? display.substring(0, display.lastIndexOf("/")) : "",
      display, kind: "jpeg", sharpness: 0, aesthetic: null, aesthetic_source: null,
      scene_preset: null,
    };
    openHero(pick);
  });
}

// Fetch + append the next page. `first` resets focus to the top of the grid.
async function loadMoreLibrary(first = false) {
  const grid = $("#lib-grid");
  const meta = $("#lib-meta");
  if (libState.activeRootId === null || libState.loading) return;
  if (!first && libState.loaded >= libState.total) return;
  libState.loading = true;
  try {
    const res = await fetch("/api/library/frames?" + _libFrameParams(libState.loaded, LIB_PAGE));
    const d = await res.json();
    libState.total = d.total;
    if (first && !d.frames.length) {
      grid.innerHTML = '<p class="empty" style="grid-column:1/-1;color:var(--dim);text-align:center;padding:3rem">No frames match (try clearing filters or rescanning).</p>';
      meta.textContent = "0 frames";
      return;
    }
    // Drop any prior sentinel before appending.
    const oldSentinel = document.getElementById("lib-sentinel");
    if (oldSentinel) oldSentinel.remove();

    const startIdx = libState.loaded;
    const html = d.frames.map((f, i) => _libCellHtml(f, startIdx + i)).join("");
    grid.insertAdjacentHTML("beforeend", html);
    // Wire only the newly-added cells.
    Array.from(grid.querySelectorAll(".lib-cell")).slice(startIdx).forEach(_wireLibCell);
    libState.loaded += d.frames.length;

    meta.textContent = `${libState.loaded} of ${libState.total} frame${libState.total === 1 ? '' : 's'}`;

    // Re-arm infinite scroll if there's more to load.
    if (libState.loaded < libState.total) {
      grid.insertAdjacentHTML("beforeend",
        '<div id="lib-sentinel" style="grid-column:1/-1;text-align:center;color:var(--dim);padding:1rem;cursor:pointer">Load more…</div>');
      const sentinel = document.getElementById("lib-sentinel");
      sentinel.addEventListener("click", () => loadMoreLibrary(false));
      if ("IntersectionObserver" in window) {
        if (_libObserver) _libObserver.disconnect();
        _libObserver = new IntersectionObserver((entries) => {
          if (entries.some(e => e.isIntersecting)) loadMoreLibrary(false);
        }, {root: grid, rootMargin: "400px"});
        _libObserver.observe(sentinel);
      }
    } else if (_libObserver) {
      _libObserver.disconnect(); _libObserver = null;
    }

    if (first) setLibFocus(0);
  } catch (e) {
    console.error("grid:", e);
    meta.textContent = "load failed";
    if (first) grid.innerHTML = `<p class="empty" style="grid-column:1/-1;color:var(--red);text-align:center;padding:3rem">${escapeHtml(e.message)}</p>`;
  } finally {
    libState.loading = false;
  }
}

$("#lib-import").addEventListener("click", () => openImporter());
$("#imp-cancel").addEventListener("click", () => $("#import-modal").style.display = "none");
$("#imp-pick-source").addEventListener("click", async () => {
  const res = await fetch("/api/folder-pick", {method: "POST"});
  const d = await res.json();
  if (d.path) $("#imp-source").value = d.path;
});
$("#imp-pick-dest").addEventListener("click", async () => {
  const res = await fetch("/api/folder-pick", {method: "POST"});
  const d = await res.json();
  if (d.path) $("#imp-dest").value = d.path;
});
$("#imp-go").addEventListener("click", async () => {
  const source = $("#imp-source").value.trim();
  const destination = $("#imp-dest").value.trim();
  const scheme = $("#imp-scheme").value;
  if (!source || !destination) { $("#imp-status").textContent = "source and destination required"; return; }
  $("#imp-status").textContent = "Copying… (this may take a while)";
  $("#imp-go").disabled = true;
  try {
    const res = await fetch("/api/import", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({source, destination, scheme}),
    });
    let d;
    try { d = await res.json(); } catch { d = null; }
    if (!res.ok && res.status !== 207) {
      throw new Error(d && d.error ? d.error : await res.text());
    }
    if (d && d.partial) {
      $("#imp-status").textContent = `Imported ${d.copied}, skipped ${d.skipped}, but ${d.error_count} FAILED: ${(d.errors || []).join("; ")}`;
    } else {
      $("#imp-status").textContent = `Imported ${d.copied}, skipped ${d.skipped} duplicates. Scanned: ${d.scan.indexed_new} new frames.`;
    }
    libState.activeRootId = d.root_id;
    setTimeout(() => {
      $("#import-modal").style.display = "none";
      loadLibrary();
    }, 1500);
  } catch (e) {
    $("#imp-status").textContent = "Failed: " + e.message;
  } finally {
    $("#imp-go").disabled = false;
  }
});

function openImporter() {
  $("#imp-status").textContent = "";
  $("#import-modal").style.display = "flex";
}

$("#lib-add-root").addEventListener("click", async () => {
  const res = await fetch("/api/folder-pick", {method: "POST"});
  const d = await res.json();
  if (!d.path) {
    if (d.reason && d.reason !== "cancelled") toast("Folder pick: " + d.reason);
    return;
  }
  const r = await fetch("/api/library/roots", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({path: d.path}),
  });
  if (!r.ok) { toast("add failed: " + await r.text()); return; }
  const added = await r.json();
  libState.activeRootId = added.id;
  await refreshLibraryRoots();
  await refreshLibraryFolders();
  // Kick a scan immediately so the user sees frames appear.
  startLibraryScan(added.id);
});

$("#lib-filter-camera").addEventListener("change", e => {
  libState.cameraFilter = e.target.value;
  refreshLibraryGrid();
});
$("#lib-filter-face").addEventListener("change", e => {
  libState.faceFilter = e.target.value;
  refreshLibraryGrid();
});
$("#lib-filter-place").addEventListener("change", e => {
  libState.placeFilter = e.target.value;
  refreshLibraryGrid();
});
$("#lib-filter-star").addEventListener("change", e => {
  libState.starFilter = e.target.value;
  refreshLibraryGrid();
});
$("#lib-search").addEventListener("input", e => {
  libState.searchQuery = e.target.value.trim();
  if (libSearchTimer) clearTimeout(libSearchTimer);
  libSearchTimer = setTimeout(refreshLibraryGrid, 200);
});
$("#lib-scan").addEventListener("click", () => {
  if (libState.activeRootId === null) return;
  startLibraryScan(libState.activeRootId);
});
$("#lib-index").addEventListener("click", async () => {
  if (!confirm("Pre-index every photo across all watched folders?\\n\\nThis runs tags + face identity + scene cluster + taste score on every frame so future culls are instant. Slow first time (~1s per face-bearing frame), idempotent on re-run.")) return;
  try {
    const res = await fetch("/api/tagger/start", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({full: true}),
    });
    if (!res.ok) throw new Error(await res.text());
    toast("Full index started — watch the bottom strip", 2200);
    pollTaggerStatus();
  } catch (e) {
    toast("Index start failed: " + e.message);
  }
});
$("#lib-score").addEventListener("click", async () => {
  if (libState.activeRootId === null) return;
  const rootsRes = await fetch("/api/library/roots");
  const rd = await rootsRes.json();
  const root = rd.roots.find(r => r.id === libState.activeRootId);
  if (!root) return;
  const path = libState.activeSubdir ? `${root.path}/${libState.activeSubdir}` : root.path;
  startRun(path);
});

$("#lib-export").addEventListener("click", () => {
  if (libState.activeRootId === null) { toast("Select a folder first"); return; }
  openExportModal({source: "library", defaultName: libState.activeSubdir || ""});
});

$("#lib-xmp").addEventListener("click", async () => {
  if (libState.activeRootId === null) { toast("Select a folder first"); return; }
  let shas;
  try { shas = await fetchLibraryShas(); }
  catch (e) { toast("Couldn't read library: " + e.message); return; }
  if (!shas.length) { toast("Nothing in scope"); return; }
  if (!confirm(`Write XMP sidecars next to ${shas.length} original file(s)?\\n\\nRatings + colour labels land in .xmp files alongside your photos (source pixels untouched). Lightroom / Bridge / digiKam read them.`)) return;
  const btn = $("#lib-xmp");
  btn.disabled = true; const orig = btn.textContent; btn.textContent = "Writing…";
  try {
    const res = await fetch("/api/xmp-writeback", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({shas}),
    });
    let d;
    try { d = await res.json(); } catch { d = null; }
    if (!res.ok && res.status !== 207) throw new Error(d && d.error ? d.error : await res.text());
    if (d && d.partial) toast(`Wrote ${d.written} XMP, ${d.error_count} failed: ${(d.errors || []).join("; ")}`, 5000);
    else toast(`Wrote ${d.written} XMP sidecar${d.written === 1 ? "" : "s"} next to your originals`, 2600);
  } catch (e) {
    toast("XMP writeback failed: " + e.message, 4000);
  } finally {
    btn.disabled = false; btn.textContent = orig;
  }
});

let _pendingExportLabel = null;
let _pendingExportN = null;
let _pendingExportOpts = null;

async function _maybeAutoExport(job) {
  if (_pendingExportOpts === null || !job.picks || !job.picks.length) return;
  const n = _pendingExportN || job.picks.length;
  const shas = job.picks.slice(0, n).map(p => p.sha);
  const opts = _pendingExportOpts;
  _pendingExportLabel = null;
  _pendingExportN = null;
  _pendingExportOpts = null;
  try {
    const res = await fetch("/api/export-bangers", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({shas, ...opts}),
    });
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();
    toast(`Copied ${d.copied} files to ${d.folder}`, 4000);
    await fetch("/api/open-folder", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({path: d.folder}),
    });
  } catch (e) {
    toast("Auto-export failed: " + e.message);
  }
}

// ===== Export modal =====
let _exportContext = null;  // {source: 'results' | 'library', defaultName?}

async function openExportModal({source, defaultName} = {}) {
  _exportContext = {source: source || "results"};
  $("#exp-name").value = "";
  $("#exp-name").placeholder = "(timestamp)" + (defaultName ? ` — e.g. ${defaultName}` : "");
  if (defaultName) $("#exp-name").value = defaultName;
  $("#exp-count").value = (source === "results" && lastResults) ? lastResults.picks.length : 10;
  $("#exp-status").textContent = "";
  $("#exp-go").disabled = false;
  $("#exp-go").textContent = "Export";
  $("#export-modal").style.display = "flex";
}

$("#exp-cancel").addEventListener("click", () => $("#export-modal").style.display = "none");

const EXP_PRESETS = {
  best:       {strategy: "topk",   bias_tags: []},
  diverse:    {strategy: "kmeans", bias_tags: []},
  mixed:      {strategy: "mmr",    bias_tags: [], diversity: 0.5},
  people:     {strategy: "faces",  bias_tags: []},
  landscapes: {strategy: "kmeans", bias_tags: [
    "mountain","hill","valley","forest","woods","meadow","beach","ocean","lake",
    "river","waterfall","desert","cave","wide-angle landscape","aerial view","drone shot",
  ]},
  pets:       {strategy: "kmeans", bias_tags: [
    "dog","cat","horse","bird","fish","lion","tiger","elephant","monkey",
    "giraffe","zebra","wolf","bear","deer","cow","sheep",
  ]},
  food:       {strategy: "kmeans", bias_tags: [
    "food","drink","coffee","wine","cake","pizza","restaurant","cafe","kitchen","eating","cooking",
  ]},
};

$("#exp-go").addEventListener("click", async () => {
  const label = $("#exp-name").value.trim();
  const n = parseInt($("#exp-count").value) || 10;
  const presetKey = $("#exp-preset").value || "diverse";
  const preset = EXP_PRESETS[presetKey] || EXP_PRESETS.diverse;
  const cullOpts = {top_n: n, ...preset};

  // Gather candidate SHAs depending on source.
  let shas = [];
  if (_exportContext && _exportContext.source === "library") {
    try { shas = await fetchLibraryShas(); }
    catch (e) { $("#exp-status").textContent = "Couldn't read library: " + e.message; return; }
  } else if (lastResults && lastResults.picks.length) {
    shas = lastResults.picks.map(p => p.sha);
  }
  if (!shas.length) { $("#exp-status").textContent = "Nothing in scope"; return; }

  $("#exp-go").disabled = true;
  $("#exp-go").textContent = "Culling…";
  $("#exp-status").textContent = `Picking ${n} from ${shas.length} via "${presetKey}"…`;

  try {
    const cullRes = await fetch("/api/cull", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({shas, ...cullOpts}),
    });
    if (!cullRes.ok) throw new Error(await cullRes.text());
    const cd = await cullRes.json();
    if (cd.needs_indexing) {
      $("#exp-status").textContent = `${cd.missing}/${cd.total} frames not indexed yet — run "Index all" first.`;
      $("#exp-go").disabled = false;
      $("#exp-go").textContent = "Export";
      return;
    }
    if (!cd.picks || !cd.picks.length) {
      $("#exp-status").textContent = "No frames survived the sharpness gate.";
      $("#exp-go").disabled = false;
      $("#exp-go").textContent = "Export";
      return;
    }
    const pickShas = cd.picks.map(p => p.sha);
    $("#exp-go").textContent = "Copying…";
    $("#exp-status").textContent = `Copying ${pickShas.length} originals…`;

    const res = await fetch("/api/export-bangers", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({shas: pickShas, label}),
    });
    // A 207 is a *partial* export (some files failed to copy/verify) and still
    // returns a JSON body with a folder + error list — don't treat it as a hard
    // failure, but flag it clearly. Only non-JSON / 4xx-5xx outside 207 throws.
    let d;
    try { d = await res.json(); } catch { d = null; }
    if (!res.ok && res.status !== 207) {
      throw new Error(d && d.error ? d.error : await res.text());
    }
    if (d && d.partial) {
      $("#exp-status").textContent =
        `Copied ${d.copied}, but ${d.error_count} file(s) FAILED: ${(d.errors || []).join("; ")}`;
    } else {
      $("#exp-status").textContent = `Done. Copied ${d.copied} files.`;
    }
    await fetch("/api/open-folder", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({path: d.folder}),
    });
    // Leave the modal up longer on a partial export so the user reads the error.
    setTimeout(() => $("#export-modal").style.display = "none", d && d.partial ? 6000 : 1500);
  } catch (e) {
    $("#exp-status").textContent = "Failed: " + e.message;
  } finally {
    $("#exp-go").disabled = false;
    $("#exp-go").textContent = "Export";
  }
});

async function fetchLibraryShas() {
  // Pull every sha currently in the active library scope by walking the
  // existing /api/library/frames endpoint with a generous page size.
  const params = new URLSearchParams();
  if (libState.activeRootId) params.set("root_id", libState.activeRootId);
  if (libState.activeSubdir) params.set("subdir", libState.activeSubdir);
  // NB: these must mirror the exact libState field names the handlers write
  // (searchQuery / cameraFilter / faceFilter / placeFilter / starFilter).
  // The old names (searchQ/filterCamera/…) silently read undefined, so the
  // export ignored every active filter and culled the whole root.
  if (libState.searchQuery) params.set("q", libState.searchQuery);
  if (libState.cameraFilter) params.set("camera", libState.cameraFilter);
  if (libState.faceFilter) params.set("face", libState.faceFilter);
  if (libState.placeFilter) params.set("place", libState.placeFilter);
  if (libState.starFilter !== "") params.set("min_score", libState.starFilter);
  params.set("limit", "5000");
  const res = await fetch("/api/library/frames?" + params.toString());
  if (!res.ok) throw new Error(await res.text());
  const d = await res.json();
  return (d.frames || []).map(f => f.sha);
}

async function startLibraryScan(rootId) {
  toast("Scanning…", 1500);
  try {
    await fetch("/api/library/scan/" + rootId, {method: "POST"});
    pollLibraryScan(rootId);
  } catch (e) { toast("scan failed: " + e.message); }
}

async function pollLibraryScan(rootId) {
  try {
    const res = await fetch("/api/library/scan-status/" + rootId);
    const d = await res.json();
    $("#lib-meta").textContent = `${d.phase} · ${d.indexed_new} new, ${d.indexed_updated} updated, ${d.removed} removed`;
    if (d.phase === "done" || d.phase === "error") {
      await refreshLibraryRoots();
      await refreshLibraryFolders();
      await refreshLibraryCameras();
      await refreshLibraryGrid();
      toast(d.phase === "done" ? "Scan complete" : "Scan errored: " + (d.error || ""));
      return;
    }
  } catch (e) { console.error("scan poll:", e); }
  setTimeout(() => pollLibraryScan(rootId), 800);
}

document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("face-discover");
  if (btn) btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Discovering…";
    try {
      const res = await fetch("/api/face/discover", {method: "POST"});
      if (!res.ok) throw new Error(await res.text());
      const d = await res.json();
      toast(`Found ${d.total_clusters} people across ${d.total_faces} faces (${d.added} new, ${d.matched} merged)`, 3000);
      loadFaceLibrary();
    } catch (e) {
      toast("Discover failed: " + e.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Discover people";
    }
  });
});

async function loadFaceLibrary() {
  const wrap = $("#face-library");
  try {
    const res = await fetch("/api/face/library");
    const d = await res.json();
    if (!d.names || !d.names.length) {
      wrap.innerHTML = '<p class="empty">No named faces yet. Open a photo with a detected face and click its name chip to start.</p>';
      return;
    }
    wrap.innerHTML = d.names.map(n => `
      <div class="face-card" data-name="${escapeHtml(n.name)}">
        <img src="/api/face/library-thumb/${encodeURIComponent(n.name)}?ts=${Date.now()}" alt="${escapeHtml(n.name)}" onerror="this.style.display='none'">
        <span class="lib-name">${escapeHtml(n.name)}</span>
        <span class="lib-meta">${n.count} sample${n.count === 1 ? '' : 's'}</span>
        <div class="lib-actions">
          <button class="forget" title="Forget this person (delete centroid)">×</button>
        </div>
      </div>
    `).join("");
    wireLibraryCards();
  } catch (e) {
    wrap.innerHTML = `<p class="empty">load failed: ${escapeHtml(e.message)}</p>`;
  }
}

function wireLibraryCards() {
  $$('#face-library .face-card').forEach(card => {
    const name = card.dataset.name;
    const nameEl = card.querySelector(".lib-name");
    nameEl.addEventListener("click", () => beginLibraryRename(card, name));
    card.querySelector(".forget").addEventListener("click", async () => {
      if (!confirm(`Forget "${name}"? Future faces won't match against this centroid.`)) return;
      try {
        const res = await fetch("/api/face/forget", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({name}),
        });
        if (!res.ok) throw new Error(await res.text());
        loadFaceLibrary();
        toast(`Forgot "${name}"`);
      } catch (e) {
        toast("forget failed: " + e.message);
      }
    });
  });
}

function beginLibraryRename(card, oldName) {
  const nameEl = card.querySelector(".lib-name");
  nameEl.outerHTML = `<input type="text" value="${escapeHtml(oldName)}">`;
  const input = card.querySelector("input");
  input.focus();
  input.select();
  const cancel = () => loadFaceLibrary();
  input.addEventListener("blur", cancel);
  input.addEventListener("keydown", async (e) => {
    if (e.key === "Escape") { input.removeEventListener("blur", cancel); cancel(); return; }
    if (e.key !== "Enter") return;
    const newName = input.value.trim();
    if (!newName || newName === oldName) { input.removeEventListener("blur", cancel); cancel(); return; }
    input.removeEventListener("blur", cancel);
    try {
      const res = await fetch("/api/face/rename", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({old: oldName, new: newName}),
      });
      if (!res.ok) throw new Error(await res.text());
      toast(`Renamed "${oldName}" → "${newName}"`);
      loadFaceLibrary();
    } catch (e) {
      toast("rename failed: " + e.message);
      loadFaceLibrary();
    }
  });
}

async function loadState() {
  const res = await fetch("/api/state");
  const s = await res.json();
  const setBadge = (sel, present, label) => {
    const el = $(sel);
    el.textContent = present ? "ready" : "absent";
    el.classList.toggle("ok", present);
    el.classList.toggle("warn", !present);
    el.title = label;
  };
  setBadge("#st-head", s.head_loaded, "Taste head trained via `banger train`");
  setBadge("#st-scene", s.scene_model, "Scene clusters fit via `banger scenes fit`");
  setBadge("#st-eyes", s.mediapipe, "mediapipe installed = --eye-gate works");
  setBadge("#st-faces", s.insightface, "insightface installed = 'faces' strategy works");
  $("#st-labels").textContent = s.labels_total + " frames";
  $("#st-embs").textContent = (s.cache_counts.embeddings || 0) + " cached";
  // Hide options that depend on missing deps
  if (!s.mediapipe) $("#opt-eye-gate").parentElement.style.opacity = ".4";
  if (!s.insightface) {
    const facesOpt = document.querySelector('#opt-strategy option[value="faces"]');
    if (facesOpt) facesOpt.disabled = true;
  }
}

async function pollSetup() {
  try {
    const res = await fetch("/api/setup-status");
    if (!res.ok) throw new Error(await res.text());
    const s = await res.json();
    $("#splash-msg").textContent = s.message;
    $("#splash-meta").textContent = `${s.phase} · ${s.elapsed.toFixed(1)}s`;
    if (s.phase === "ready") {
      $("#splash").classList.remove("show");
      loadState();  // refresh badges now that scenes/mediapipe may have flipped
      return;
    }
  } catch (e) { console.error("setup poll:", e); }
  setTimeout(pollSetup, 700);
}

async function bootstrap() {
  // Initial setup-status fetch decides whether to show the splash.
  try {
    const res = await fetch("/api/setup-status");
    const s = await res.json();
    if (s.phase !== "ready") {
      $("#splash").classList.add("show");
      pollSetup();
    }
  } catch (e) { /* server may still be coming up */ }
}

loadRecent();
loadState();
loadLibrary();
show("library");
bootstrap();
</script>

</body>
</html>
"""
