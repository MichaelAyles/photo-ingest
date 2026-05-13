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

from banger import aesthetic, dedup, eyes as eyes_mod, face, face_id as face_id_mod
from banger import metrics as metrics_mod
from banger import scene_kmeans, scenes, select, state, taste_head
from banger import xmp as xmp_mod
from banger.frames import discover_frames
from banger.preview import load_preview
from banger.report import Row, encode_thumbnail_bytes
from banger.sharpness import CONFIG as SHARP_CFG, sharpness_from_preview

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
_recent_folders: list[str] = []
_label_subprocesses: dict[str, dict] = {}  # input_dir -> {"port": int, "proc": subprocess.Popen}


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
        recursive = bool(opts.get("recursive", True))
        top_n = int(opts.get("top_n", 10))
        strategy = str(opts.get("strategy", "kmeans"))
        diversity = float(opts.get("diversity", 0.5))
        face_gate = bool(opts.get("face_gate", False))
        eye_gate = bool(opts.get("eye_gate", False))
        write_xmp = bool(opts.get("write_xmp", False))

        job.stage = "discovering"
        frames = discover_frames(job.input_dir, recursive=recursive)
        job.total = len(frames)
        if not frames:
            job.status = "error"
            job.error = f"No supported images found in {job.input_dir}"
            return
        log_line(f"discovered {len(frames)} frames")

        threshold = SHARP_CFG["threshold"]
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
                cached_meta is not None and "face_embeddings" in cached_meta
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
                    face_embs_payload = None
                    if strategy == "faces":
                        try:
                            embs = face_id_mod.extract_face_embeddings(preview)
                            face_embs_payload = face_id_mod.encode_for_cache(embs)
                        except Exception as e:
                            log_line(f"face_id skip {f.display_name}: {e}")
                    state.cache_frame_metadata(
                        sha, sharp, str(phash), ts,
                        face_count=face_count, face_sharpness=face_sharp,
                        metrics=frame_metrics, eyes=frame_eyes,
                        face_embeddings=face_embs_payload,
                    )
                except Exception as e:
                    log_line(f"score fail {f.display_name}: {e}")
                    continue
                cold += 1

            if face_gate and face_count > 0 and face_sharp < face.FACE_SHARPNESS_THRESHOLD:
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
                face_embs_per_item.append(
                    face_id_mod.decode_from_cache(meta2.get("face_embeddings"))
                )
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
_FLASH_FIRED = lambda v: "fired" if (isinstance(v, int) and v & 1) else "no flash"


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

    @app.route("/")
    @app.route("/gui")
    def root():
        return render_template_string(_SPA_TEMPLATE)

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

        job = JobState(job_id=str(uuid.uuid4()), input_dir=input_dir, options=data)
        with _jobs_lock:
            _jobs[job.job_id] = job
        t = threading.Thread(target=_run_job, args=(job, sha_to_frame), daemon=True)
        t.start()
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
        f = sha_to_frame.get(sha)
        if f is None:
            return ("not found", 404)
        try:
            preview = load_preview(f.classify_path)
        except Exception as e:
            log.warning("thumb fail %s: %s", f.display_name, e)
            return ("preview failed", 500)
        state.cache_thumbnail(sha, encode_thumbnail_bytes(preview))
        return send_file(path, mimetype="image/jpeg")

    @app.route("/api/preview/<sha>")
    def preview(sha):
        path = state.preview_jpeg_path(sha)
        if path.exists():
            return send_file(path, mimetype="image/jpeg")
        f = sha_to_frame.get(sha)
        if f is None:
            return ("not found", 404)
        try:
            preview_arr = load_preview(f.classify_path)
        except Exception as e:
            log.warning("preview fail %s: %s", f.display_name, e)
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

    @app.route("/api/details/<sha>")
    def details(sha):
        """Return everything we know about one frame: scores, metrics, EXIF.

        Used by the click-on-photo detail overlay. Pulls cached metadata from
        disk (computed during the pipeline run) and reads EXIF from the source
        JPEG with PIL. RAW EXIF would need rawpy + a separate parser, deferred
        until a real ARW shoot lands in test_photos.
        """
        f = sha_to_frame.get(sha)
        if f is None:
            return jsonify({"error": "unknown sha"}), 404
        meta = state.load_frame_metadata(sha) or {}
        exif = _extract_exif(f.classify_path)
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
            "sharpness": meta.get("sharpness"),
            "exif": exif,
        })

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


def serve(port: int = 8765, open_window: bool = True) -> None:
    """Entry point used by `banger gui`. Launches Flask in a background thread
    and pywebview on the main thread (which has to be the main thread on macOS)."""

    window_holder: dict = {}
    app = build_app(window_holder)

    def _flask():
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False, threaded=True)

    t = threading.Thread(target=_flask, daemon=True)
    t.start()

    # Warm-load CLIP in parallel with the window opening. Without this, the
    # first frame of the first run pays ~15s of model-load latency before
    # any progress shows. With it, model load overlaps with the user picking
    # a folder, so by the time they click 'open' the model is usually ready.
    def _warm_clip():
        t0 = time.monotonic()
        try:
            aesthetic._load()
            log.info("CLIP warmed in %.1fs", time.monotonic() - t0)
        except Exception as e:
            log.warning("CLIP warm-load failed: %s", e)

    threading.Thread(target=_warm_clip, daemon=True).start()
    threading.Thread(target=_auto_setup, daemon=True).start()


def _auto_setup() -> None:
    """Best-effort background bootstrap. Runs once per GUI launch:

    1. If scene KMeans isn't fit and we have enough cached embeddings,
       fit it silently. Fast (~5s on 1k embeddings), no network.
    2. If mediapipe isn't installed, pip-install it. Enables the eye-gate
       on the next run. Quiet on success, logged on failure. Doesn't block
       anything if the install fails (the gate has its own fallback path).
    """
    import subprocess
    import sys

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

    if not eyes_mod.mediapipe_available():
        marker = state.STATE_DIR / "mediapipe_install_attempted"
        if marker.exists():
            log.info("auto-setup: mediapipe install previously failed, skipping retry")
        else:
            log.info("auto-setup: spawning detached mediapipe install (~30-60s)…")
            try:
                state.STATE_DIR.mkdir(parents=True, exist_ok=True)
                marker.touch()  # set first so we don't retry on every crash-during-install
                log_path = state.STATE_DIR / "mediapipe_install.log"
                creationflags = 0
                if os.name == "nt":
                    # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP so the pip
                    # process survives the GUI window closing.
                    creationflags = 0x00000008 | 0x00000200
                with open(log_path, "w", encoding="utf-8") as fp:
                    subprocess.Popen(
                        [sys.executable, "-m", "pip", "install", "mediapipe"],
                        stdout=fp, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                        creationflags=creationflags,
                        close_fds=True,
                    )
                log.info(
                    "auto-setup: mediapipe install detached (log: %s). "
                    "Restart GUI when it finishes to enable --eye-gate.",
                    log_path,
                )
            except OSError as e:
                log.warning("auto-setup: mediapipe install spawn failed: %s", e)

    # Wait briefly for the Flask socket so the webview's first nav doesn't
    # race-fail. 200 ms is plenty on a modern machine; on a slow one we
    # retry up to 2 s.
    import socket

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with socket.socket() as s:
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

    win = webview.create_window(
        title="banger",
        url=f"http://127.0.0.1:{port}/gui",
        width=1280, height=820, min_size=(900, 600),
        background_color="#0c0c0c",
    )
    window_holder["win"] = win
    webview.start()


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
  .overlay .close { position: absolute; top: 1rem; right: 1rem; background: var(--bg3); color: var(--fg); border: 1px solid var(--line); padding: .35rem .8rem; border-radius: 4px; cursor: pointer; z-index: 1; }
  .overlay .hint { position: absolute; bottom: 1rem; left: 50%; transform: translateX(-50%); color: var(--dim); font-size: .7rem; pointer-events: none; }
</style>
</head>
<body>

<header>
  <h1>banger</h1>
  <nav>
    <button data-view="welcome" class="active">Welcome</button>
    <button data-view="running" id="nav-running" style="display:none">Running</button>
    <button data-view="results" id="nav-results" style="display:none">Results</button>
    <button data-view="label" id="nav-label" style="display:none">Label</button>
    <button data-view="settings">Settings</button>
  </nav>
</header>

<main>

<section class="view" id="view-welcome">
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
  </div>
  <div class="grid" id="results-grid"></div>
</section>

<section class="view hidden" id="view-label" style="padding:0">
  <iframe id="label-frame" src="about:blank" style="width:100%;height:100%;border:0;background:#0c0c0c;display:block"></iframe>
</section>

<section class="view hidden" id="view-settings">
  <div class="settings-pane">
    <h2>Status</h2>
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

$$("header nav button").forEach(b => b.addEventListener("click", () => show(b.dataset.view)));

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
  show("running");
  $("#run-log").innerHTML = "";
  pollJob();
}

async function pollJob() {
  if (!currentJob) return;
  try {
    const res = await fetch(`/api/jobs/${currentJob}`);
    if (!res.ok) throw new Error(await res.text());
    const j = await res.json();
    renderJob(j);
    if (j.status === "done" || j.status === "error") {
      clearTimeout(pollTimer);
      if (j.status === "done") {
        lastResults = j;
        $("#nav-results").style.display = "";
        renderResults(j);
        show("results");
      } else {
        toast(j.error || "job errored");
      }
      return;
    }
  } catch (e) { console.error(e); }
  pollTimer = setTimeout(pollJob, 600);
}

function renderJob(j) {
  $("#run-stage").textContent = j.stage;
  const pct = j.total ? (j.progress / j.total * 100) : 0;
  $("#run-bar").style.width = pct + "%";
  $("#run-counts").textContent = `${j.progress} / ${j.total}`;
  $("#run-elapsed").textContent = `${j.elapsed.toFixed(1)}s`;
  const log = $("#run-log");
  log.innerHTML = j.tail.map(l => `<div class="row">${escapeHtml(l)}</div>`).join("");
  log.scrollTop = log.scrollHeight;
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
  $("#overlay-img").src = "/api/preview/" + pick.sha;
  $("#overlay-panel").innerHTML = renderPanelLoading(pick);
  $("#overlay").classList.add("show");
  try {
    const res = await fetch("/api/details/" + pick.sha);
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();
    $("#overlay-panel").innerHTML = renderPanel(pick, d);
  } catch (e) {
    $("#overlay-panel").innerHTML = renderPanelLoading(pick) +
      `<p class="empty">details fetch failed: ${escapeHtml(e.message)}</p>`;
  }
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

  // Face / eyes block.
  let faceBlock = "";
  if (d.face_count) {
    const faceRows = [
      ["faces detected", d.face_count],
      ["face sharpness", d.face_sharpness !== null && d.face_sharpness !== undefined ? d.face_sharpness.toFixed(0) : "—"],
    ];
    if (d.eyes) {
      faceRows.push(["min EAR", d.eyes.ear_min !== undefined ? d.eyes.ear_min.toFixed(3) : "—"]);
      faceRows.push(["blink detected", d.eyes.any_blink ? "yes" : "no"]);
    }
    faceBlock = `<h3>Faces</h3><dl>${faceRows.map(([k,v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;
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
  if (exif.metering_mode) exifPairs.push(["metering", exif.metering_mode]);
  if (exif.flash) exifPairs.push(["flash", exif.flash]);
  if (exif.pixel_x && exif.pixel_y) exifPairs.push(["dimensions", `${exif.pixel_x} × ${exif.pixel_y}`]);

  const exifHtml = exifPairs.length
    ? `<dl>${exifPairs.map(([k,v]) => `<dt>${k}</dt><dd>${escapeHtml(String(v))}</dd>`).join("")}</dl>`
    : '<p class="empty">no EXIF data (or non-JPEG source)</p>';

  return `
    <div class="head">
      <span class="rank">#${pick.rank}</span>
      <span class="stars">${stars}</span>
      <span class="stem">${escapeHtml(pick.display)}</span>
    </div>
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

function closeOverlay() { $("#overlay").classList.remove("show"); }

$("#overlay-close").addEventListener("click", closeOverlay);
// Click on the overlay backdrop closes; clicks inside .overlay-inner don't.
$("#overlay").addEventListener("click", e => {
  if (e.target === $("#overlay")) closeOverlay();
});
document.addEventListener("keydown", e => {
  if (e.key === "Escape") closeOverlay();
});

$("#btn-rerun").addEventListener("click", () => show("welcome"));
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

loadRecent();
loadState();
show("welcome");
</script>

</body>
</html>
"""
