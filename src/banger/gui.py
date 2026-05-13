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
    aesthetic, dedup, editor as editor_mod, eyes as eyes_mod, face,
    face_id as face_id_mod, face_names, library, tagger as tagger_mod,
    tags as tags_mod,
)
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

    @app.route("/api/develop/<sha>", methods=["GET"])
    def develop_get(sha):
        p = editor_mod.load_params(sha)
        return jsonify({
            "params": editor_mod.to_dict(p),
            "has_edits": editor_mod.has_edits(sha),
        })

    @app.route("/api/develop/<sha>", methods=["POST"])
    def develop_save(sha):
        data = request.get_json(silent=True) or {}
        p = editor_mod.from_dict(data.get("params") or {})
        editor_mod.save_params(sha, p)
        return jsonify({"saved": True, "params": editor_mod.to_dict(p)})

    @app.route("/api/develop/<sha>/render")
    def develop_render(sha):
        """Render the preview-size image with the given params, return JPEG.

        Params come via query string so the SPA can fire a fast GET on each
        slider move. We always render from the 1024px preview cache; the
        full-size export endpoint is separate.
        """
        from flask import Response
        import cv2

        params_dict = {}
        for k in editor_mod.DevelopParams.__dataclass_fields__:
            v = request.args.get(k)
            if v is None:
                continue
            if k == "crop":
                # crop comes as "x,y,w,h" normalised.
                try:
                    parts = [float(x) for x in v.split(",")]
                    if len(parts) == 4:
                        params_dict["crop"] = {"x": parts[0], "y": parts[1],
                                              "w": parts[2], "h": parts[3]}
                except ValueError:
                    pass
            else:
                try:
                    params_dict[k] = float(v)
                except ValueError:
                    pass

        params = editor_mod.from_dict(params_dict)
        # Reuse cached preview if present; else load + cache.
        preview_path = state.preview_jpeg_path(sha)
        if preview_path.exists():
            img = cv2.imread(str(preview_path), cv2.IMREAD_COLOR)
        else:
            src = _resolve_path(sha)
            if src is None:
                return ("unknown sha", 404)
            img = load_preview(src)
            state.cache_preview_jpeg(sha, _encode_preview_jpeg(img))
        out = editor_mod.apply_develop(img, params)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return ("encode failed", 500)
        return Response(buf.tobytes(), mimetype="image/jpeg")

    @app.route("/api/develop/<sha>/export", methods=["POST"])
    def develop_export(sha):
        """Render full-res and write a JPEG next to the source (or at `path`)."""
        data = request.get_json(silent=True) or {}
        src = _resolve_path(sha)
        if src is None:
            return jsonify({"error": "unknown sha"}), 404
        params = editor_mod.from_dict(data.get("params") or {})
        # Persist the params so reopening shows the same edits.
        editor_mod.save_params(sha, params)
        dst_raw = (data.get("path") or "").strip()
        dst = Path(dst_raw).expanduser() if dst_raw else editor_mod.default_export_path(src)
        try:
            out = editor_mod.export_jpeg(src, params, dst)
        except Exception as e:
            log.exception("export failed")
            return jsonify({"error": str(e)}), 500
        return jsonify({"path": str(out)})

    @app.route("/api/import", methods=["POST"])
    def import_files():
        """Copy photos from a source directory into a destination, organised
        by date / camera / flat, then add the destination to the library and
        kick a scan.

        For now this runs synchronously since most imports are small. A
        background-job version with progress would come if the typical
        import grows past ~1k files.
        """
        import shutil
        from datetime import datetime as _dt

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
        copied = 0
        skipped = 0
        for f in frames:
            srcp = f.classify_path
            subfolder = _subfolder_for(srcp, scheme)
            target_dir = dst / subfolder if subfolder else dst
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / srcp.name
            if target.exists() and target.stat().st_size == srcp.stat().st_size:
                skipped += 1
                continue
            shutil.copy2(srcp, target)
            # Also copy the sibling RAW or JPEG if present.
            other = f.raw if f.classify_path == f.jpeg else f.jpeg
            if other is not None and other != srcp and other.exists():
                shutil.copy2(other, target_dir / other.name)
            copied += 1

        root = library.add_root(dst)
        # Kick a scan (synchronous so the response reflects the new frames).
        progress = library.ScanProgress(
            root_id=root.id, root_path=root.path, started_at=time.monotonic()
        )
        with _scan_lock:
            _scan_progress[root.id] = progress
        library.scan_root(root.id, progress=progress)
        return jsonify({
            "copied": copied,
            "skipped": skipped,
            "root_id": root.id,
            "scan": progress.to_view(),
        })

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

        rows = library.query_frames(
            root_id=root_id, subdir=subdir, camera=camera,
            after=after, before=before, limit=limit, offset=offset,
        )
        # Enrich each frame with whatever the existing metadata cache + label DB
        # carry. We're crossing a boundary here: library has fast SQL filters
        # for root/camera/date, label/metadata filters happen in Python over
        # the returned page. That's fine while pages stay <1k frames.
        labels_map = state.labels_dict() if min_score is not None else None
        out = []
        for r in rows:
            meta = state.load_frame_metadata(r.sha) or {}
            label = labels_map.get(r.sha) if labels_map is not None else state.get_label(r.sha)
            # Face-name post-filter: any of the matched names equals `face_name`.
            if face_name is not None:
                detections = meta.get("face_detections") or []
                matched_names = set()
                for d in detections:
                    emb = np.asarray(d.get("embedding") or [], dtype=np.float32)
                    if emb.size == face_names.EMB_DIM:
                        n, _sim = face_names.match(emb)
                        if n:
                            matched_names.add(n)
                if face_name not in matched_names:
                    continue
            if min_score is not None and (label is None or label < min_score):
                continue
            if place_filter is not None and r.place_city != place_filter:
                continue
            if q is not None:
                # Search across stem, rel_path, camera, place, and tags. Cheap
                # substring match; not BM25 but good enough for a 50k-frame
                # library where the user already has root+camera filters.
                blob_parts = [r.stem.lower(), r.rel_path.lower()]
                if r.camera_model:
                    blob_parts.append(r.camera_model.lower())
                for place in (r.place_city, r.place_region, r.place_country):
                    if place:
                        blob_parts.append(place.lower())
                for t in (meta.get("tags") or []):
                    if isinstance(t, (list, tuple)) and t:
                        blob_parts.append(str(t[0]).lower())
                if q not in " | ".join(blob_parts):
                    continue
            out.append({
                **r.to_view(),
                "label": label,
                "tags": meta.get("tags"),
                "has_face_data": bool(meta.get("face_detections")),
                "scored": "metrics" in meta,
            })
        return jsonify({"frames": out, "total": len(out)})

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
        x1 = max(0, x1 - pw); y1 = max(0, y1 - ph)
        x2 = min(w, x2 + pw); y2 = min(h, y2 + ph)
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
                x1 = max(0, x1 - pw); y1 = max(0, y1 - ph)
                x2 = min(w, x2 + pw); y2 = min(h, y2 + ph)
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
            pairs = tags_mod.tag_from_embedding(emb)
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
            "tags": meta.get("tags"),
            "caption": meta.get("caption"),
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


def serve(port: int = 8765, open_window: bool = True) -> None:
    """Entry point used by `banger gui`. Launches Flask in a background thread
    and pywebview on the main thread (which has to be the main thread on macOS)."""

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
    <button data-view="editor" id="nav-editor" style="display:none">Editor</button>
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
        <button id="lib-score" class="primary">Score this view</button>
      </div>
      <div class="lib-grid" id="lib-grid">
        <div class="lib-onboarding" id="lib-onboarding">
          <h2>Welcome to your library</h2>
          <p>Point banger at the folders you already keep your photos in, or pull them in from a camera / SD card. We'll index, score, tag, name faces, and let you edit; nothing leaves your machine.</p>
          <div class="onboarding-actions">
            <button id="onb-add-folder">📁 Open folder</button>
            <button id="onb-import">📷 Import from device</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="view hidden" id="view-editor" style="padding:0;overflow:hidden">
  <div class="ed-shell">
    <div class="ed-canvas">
      <img id="ed-img" alt="">
      <div class="ed-info" id="ed-info"></div>
    </div>
    <aside class="ed-sidebar">
      <div class="ed-toolbar">
        <button id="ed-back" title="Back to library">← Library</button>
        <button id="ed-before-after" title="Hold to see original">Before/After</button>
        <button id="ed-reset" class="muted">Reset</button>
        <button id="ed-export" class="primary">Export…</button>
      </div>
      <div class="ed-section">
        <h3>Light</h3>
        <div class="slider-row" data-key="exposure">
          <label>Exposure</label><span class="val">0</span>
          <input type="range" min="-2" max="2" step="0.05" value="0">
        </div>
        <div class="slider-row" data-key="contrast">
          <label>Contrast</label><span class="val">0</span>
          <input type="range" min="-100" max="100" step="1" value="0">
        </div>
        <div class="slider-row" data-key="highlights">
          <label>Highlights</label><span class="val">0</span>
          <input type="range" min="-100" max="100" step="1" value="0">
        </div>
        <div class="slider-row" data-key="shadows">
          <label>Shadows</label><span class="val">0</span>
          <input type="range" min="-100" max="100" step="1" value="0">
        </div>
        <div class="slider-row" data-key="whites">
          <label>Whites</label><span class="val">0</span>
          <input type="range" min="-100" max="100" step="1" value="0">
        </div>
        <div class="slider-row" data-key="blacks">
          <label>Blacks</label><span class="val">0</span>
          <input type="range" min="-100" max="100" step="1" value="0">
        </div>
      </div>
      <div class="ed-section">
        <h3>Colour</h3>
        <div class="slider-row" data-key="saturation">
          <label>Saturation</label><span class="val">0</span>
          <input type="range" min="-100" max="100" step="1" value="0">
        </div>
        <div class="slider-row" data-key="vibrance">
          <label>Vibrance</label><span class="val">0</span>
          <input type="range" min="-100" max="100" step="1" value="0">
        </div>
        <div class="slider-row" data-key="temp">
          <label>Temperature</label><span class="val">0</span>
          <input type="range" min="-100" max="100" step="1" value="0">
        </div>
        <div class="slider-row" data-key="tint">
          <label>Tint</label><span class="val">0</span>
          <input type="range" min="-100" max="100" step="1" value="0">
        </div>
      </div>
      <div class="ed-section">
        <h3>Geometry</h3>
        <div class="slider-row" data-key="rotation">
          <label>Rotation</label><span class="val">0°</span>
          <input type="range" min="-15" max="15" step="0.1" value="0">
        </div>
      </div>
    </aside>
  </div>
</section>

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
  <button class="close" id="overlay-edit" style="right: 5.5rem">Edit ✎</button>
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
  if (b.dataset.view === "settings") loadFaceLibrary();
  if (b.dataset.view === "library") loadLibrary();
}));

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
        // Auto-open Results only if user is still on Library (most common
        // flow). If they navigated elsewhere don't yank them.
        if (currentView === "library") show("results");
        else toast("Scoring complete — see Results tab");
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
  $("#overlay-img").src = "/api/preview/" + pick.sha;
  $("#overlay-panel").innerHTML = renderPanelLoading(pick);
  $("#overlay").classList.add("show");
  let details = null;
  try {
    const res = await fetch("/api/details/" + pick.sha);
    if (!res.ok) throw new Error(await res.text());
    details = await res.json();
    $("#overlay-panel").innerHTML = renderPanel(pick, details);
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

  // Face / eyes block. Names + thumbs get loaded async after openHero.
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
    faceBlock = `
      <h3>Faces</h3>
      <dl>${faceRows.map(([k,v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>
      <div id="faces-slot-${pick.sha}"><p class="empty">loading faces…</p></div>`;
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

  return `
    <div class="head">
      <span class="rank">#${pick.rank}</span>
      <span class="stars">${stars}</span>
      <span class="stem">${escapeHtml(pick.display)}</span>
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

function closeOverlay() { $("#overlay").classList.remove("show"); }

$("#overlay-close").addEventListener("click", closeOverlay);
$("#overlay-edit").addEventListener("click", () => {
  const img = $("#overlay-img");
  // Pull the sha from the preview URL we set when opening the overlay.
  const m = img.src.match(/\/api\/preview\/([0-9a-f]+)/);
  if (!m) return;
  const sha = m[1];
  const info = $("#overlay-info");
  const display = info ? info.textContent : sha.slice(0, 12);
  closeOverlay();
  openEditor(sha, display);
});
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

// ===== Editor =====
let edState = {
  sha: null,
  display: null,
  params: null,
  rendering: false,
  pendingParams: null,  // last params asked for while a render was in-flight
  renderTimer: null,
};

const ED_KEYS = ["exposure","contrast","highlights","shadows","whites","blacks",
                 "saturation","vibrance","temp","tint","rotation"];

function edDefaults() {
  const p = {};
  ED_KEYS.forEach(k => p[k] = 0);
  p.crop = null;
  return p;
}

async function openEditor(sha, display) {
  edState.sha = sha;
  edState.display = display;
  $("#nav-editor").style.display = "";
  $("#ed-info").textContent = display;
  $("#ed-img").src = "/api/preview/" + sha;  // baseline shown while we fetch params
  try {
    const res = await fetch("/api/develop/" + sha);
    const d = await res.json();
    edState.params = {...edDefaults(), ...(d.params || {})};
  } catch (e) {
    edState.params = edDefaults();
  }
  applyParamsToSliders();
  show("editor");
  scheduleRender();
}

function applyParamsToSliders() {
  document.querySelectorAll("#view-editor .slider-row").forEach(row => {
    const key = row.dataset.key;
    const input = row.querySelector("input[type=range]");
    const val = row.querySelector(".val");
    const v = edState.params[key] ?? 0;
    input.value = v;
    val.textContent = key === "rotation" ? `${v.toFixed(1)}°`
                    : key === "exposure" ? v.toFixed(2)
                    : v.toFixed(0);
  });
}

function readSliders() {
  const p = {};
  document.querySelectorAll("#view-editor .slider-row").forEach(row => {
    const key = row.dataset.key;
    const input = row.querySelector("input[type=range]");
    p[key] = parseFloat(input.value);
  });
  return p;
}

function paramsToQuery(p) {
  const parts = [];
  for (const k of ED_KEYS) {
    if (p[k] !== undefined && p[k] !== 0) parts.push(`${k}=${p[k]}`);
  }
  if (p.crop) parts.push(`crop=${p.crop.x},${p.crop.y},${p.crop.w},${p.crop.h}`);
  return parts.join("&");
}

function scheduleRender() {
  if (!edState.sha) return;
  if (edState.renderTimer) clearTimeout(edState.renderTimer);
  edState.renderTimer = setTimeout(doRender, 80);
}

async function doRender() {
  if (!edState.sha) return;
  if (edState.rendering) {
    edState.pendingParams = readSliders();
    return;
  }
  edState.rendering = true;
  try {
    const p = readSliders();
    edState.params = {...(edState.params || {}), ...p};
    const q = paramsToQuery(p);
    const url = `/api/develop/${edState.sha}/render${q ? '?' + q : ''}`;
    // Preload into a temp image so we don't see a flash; swap on load.
    await new Promise((resolve) => {
      const im = new Image();
      im.onload = () => { $("#ed-img").src = im.src; resolve(); };
      im.onerror = () => resolve();
      im.src = url;
    });
  } finally {
    edState.rendering = false;
    if (edState.pendingParams) {
      edState.pendingParams = null;
      scheduleRender();
    }
  }
}

document.querySelectorAll("#view-editor .slider-row input[type=range]").forEach(input => {
  const row = input.closest(".slider-row");
  const val = row.querySelector(".val");
  const key = row.dataset.key;
  input.addEventListener("input", () => {
    const v = parseFloat(input.value);
    val.textContent = key === "rotation" ? `${v.toFixed(1)}°`
                    : key === "exposure" ? v.toFixed(2)
                    : v.toFixed(0);
    scheduleRender();
  });
});

$("#ed-reset").addEventListener("click", () => {
  edState.params = edDefaults();
  applyParamsToSliders();
  scheduleRender();
});

$("#ed-back").addEventListener("click", async () => {
  // Auto-save the current params before leaving.
  if (edState.sha && edState.params) {
    try {
      await fetch("/api/develop/" + edState.sha, {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({params: edState.params}),
      });
    } catch (e) { /* ignore, edits stay in memory */ }
  }
  edState.sha = null;
  $("#nav-editor").style.display = "none";
  show("library");
});

$("#ed-export").addEventListener("click", async () => {
  if (!edState.sha) return;
  $("#ed-export").disabled = true;
  $("#ed-export").textContent = "Exporting…";
  try {
    const res = await fetch(`/api/develop/${edState.sha}/export`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({params: edState.params}),
    });
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();
    toast(`Exported to ${d.path}`, 3000);
  } catch (e) {
    toast("Export failed: " + e.message, 3000);
  } finally {
    $("#ed-export").disabled = false;
    $("#ed-export").textContent = "Export…";
  }
});

// Hold to see the original; release to see edited.
const beforeAfterBtn = $("#ed-before-after");
beforeAfterBtn.addEventListener("mousedown", () => {
  if (edState.sha) $("#ed-img").src = "/api/preview/" + edState.sha;
});
beforeAfterBtn.addEventListener("mouseup", () => doRender());
beforeAfterBtn.addEventListener("mouseleave", () => doRender());

// ===== Library tab =====
let libState = {
  activeRootId: null,
  activeSubdir: null,
  cameraFilter: "",
  faceFilter: "",
  starFilter: "",
  placeFilter: "",
  searchQuery: "",
};
let libSearchTimer = null;
let taggerPollTimer = null;

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
        <p>Point banger at the folders you already keep your photos in, or pull them in from a camera / SD card. We'll index, score, tag, name faces, and let you edit; nothing leaves your machine.</p>
        <div class="onboarding-actions">
          <button onclick="document.getElementById('lib-add-root').click()">📁 Open folder</button>
          <button onclick="openImporter()">📷 Import from device</button>
        </div>
      </div>`;
    meta.textContent = "";
    return;
  }
  const params = new URLSearchParams();
  params.set("root_id", libState.activeRootId);
  if (libState.activeSubdir) params.set("subdir", libState.activeSubdir);
  if (libState.cameraFilter) params.set("camera", libState.cameraFilter);
  if (libState.faceFilter) params.set("face", libState.faceFilter);
  if (libState.starFilter !== "") params.set("min_score", libState.starFilter);
  if (libState.placeFilter) params.set("place", libState.placeFilter);
  if (libState.searchQuery) params.set("q", libState.searchQuery);
  params.set("limit", "500");

  meta.textContent = "loading…";
  try {
    const res = await fetch("/api/library/frames?" + params);
    const d = await res.json();
    meta.textContent = `${d.total} frame${d.total === 1 ? '' : 's'}`;
    if (!d.frames.length) {
      grid.innerHTML = '<p class="empty" style="grid-column:1/-1;color:var(--dim);text-align:center;padding:3rem">No frames match (try clearing filters or rescanning).</p>';
      return;
    }
    grid.innerHTML = d.frames.map(f => {
      const ratingChip = (f.label !== null && f.label !== undefined)
        ? `<span class="lib-rating">${f.label >= 0 ? '+' : ''}${f.label}</span>` : "";
      const badges = [];
      if (f.scored) badges.push('<span class="lib-badge scored">S</span>');
      if (f.has_face_data) badges.push('<span class="lib-badge face">F</span>');
      return `
        <div class="lib-cell" data-sha="${f.sha}" data-display="${escapeHtml(f.rel_path)}">
          <div class="lib-img-wrap">
            <img loading="lazy" src="/api/thumb/${f.sha}" alt="${escapeHtml(f.rel_path)}">
            ${badges.length ? `<div class="lib-badges">${badges.join("")}</div>` : ""}
          </div>
          <div class="lib-label">${ratingChip}${escapeHtml(f.stem)}</div>
        </div>`;
    }).join("");
    grid.querySelectorAll(".lib-cell").forEach(cell => {
      const sha = cell.dataset.sha;
      const display = cell.dataset.display;
      cell.addEventListener("click", () => {
        const pick = {
          sha, rank: 0, stem: display.split("/").pop().replace(/\.[^.]+$/, ""),
          subdir: display.includes("/") ? display.substring(0, display.lastIndexOf("/")) : "",
          display, kind: "jpeg", sharpness: 0, aesthetic: null, aesthetic_source: null,
          scene_preset: null,
        };
        openHero(pick);
      });
      // Double-click jumps straight to the editor, skipping the previewer.
      cell.addEventListener("dblclick", () => openEditor(sha, display));
    });
  } catch (e) {
    console.error("grid:", e);
    meta.textContent = "load failed";
    grid.innerHTML = `<p class="empty" style="grid-column:1/-1;color:var(--red);text-align:center;padding:3rem">${escapeHtml(e.message)}</p>`;
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
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();
    $("#imp-status").textContent = `Imported ${d.copied}, skipped ${d.skipped} duplicates. Scanned: ${d.scan.indexed_new} new frames.`;
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
  // Resolve the folder path: root.path + (subdir if selected).
  const rootsRes = await fetch("/api/library/roots");
  const rd = await rootsRes.json();
  const root = rd.roots.find(r => r.id === libState.activeRootId);
  if (!root) return;
  const path = libState.activeSubdir ? `${root.path}/${libState.activeSubdir}` : root.path;
  startRun(path);
});

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
