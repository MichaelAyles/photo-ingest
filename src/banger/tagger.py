"""Background tagger: walk the library, embed everything, store tags.

The scoring pipeline already caches a CLIP embedding plus metadata for
every frame it processes; the tagger does the same work but skips the
expensive bits (taste head, scene routing, selection) so it can run
quietly in the background as soon as a library scan finishes. The user
sees tags appear in detail overlays and the search bar finds frames by
content, without ever clicking 'Score this view.'

The job is idempotent: it skips frames that already have an embedding
AND tags. If only the embedding is cached (older runs), it just adds
tags. If nothing is cached, it loads the preview, runs CLIP, and
writes everything.

One job at a time across the whole process. Calling start() while one
is running returns the running progress. This keeps it from racing
itself if the user mashes 'Rescan.'
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from banger import (
    aesthetic, dedup, eyes as eyes_mod, face, face_id as face_id_mod,
    library, scene_kmeans, state, tags as tags_mod, taste_head,
)
from banger.preview import load_preview

log = logging.getLogger("banger.tagger")


@dataclass
class TaggerProgress:
    phase: str = "idle"  # idle | running | done | error
    mode: str = "tags"  # tags | full
    total: int = 0
    processed: int = 0
    embedded: int = 0
    tagged: int = 0
    faces_done: int = 0
    scored: int = 0
    skipped: int = 0
    started_at: float = 0.0
    finished_at: float | None = None
    error: str | None = None
    current: str | None = None

    def to_view(self) -> dict:
        elapsed = (self.finished_at or time.monotonic()) - self.started_at
        return {
            **self.__dict__,
            "elapsed": round(elapsed, 1) if self.started_at else 0.0,
        }


_progress = TaggerProgress()
_lock = threading.Lock()


def status() -> TaggerProgress:
    return _progress


def _tag_one(sha: str, src_path, full: bool = False) -> dict:
    """Enrich one frame. Returns a dict of what was newly computed.

    Keys: embedded, tagged, faces, scored, scene. Each True only when that
    enrichment ran AND wasn't already cached. Idempotent.

    `full=True` adds insightface face detections, scene cluster assignment,
    and taste head scoring on top of the default tags-only pass. The full
    pass is slow (~1s/frame for insightface) but means later cull/sort
    operations don't need to touch the original file at all.
    """
    meta = state.load_frame_metadata(sha) or {}
    have_emb = state.load_embedding(sha) is not None
    have_tags = bool(meta.get("tags"))
    have_faces = "face_detections" in meta
    have_score = "taste_score" in meta
    have_scene = "scene_cluster" in meta

    if have_emb and have_tags and (not full or (have_faces and have_score and have_scene)):
        return {}

    out = {"embedded": False, "tagged": False, "faces": False, "scored": False, "scene": False}
    preview = None

    if have_emb:
        emb = state.load_embedding(sha)
    else:
        try:
            preview = load_preview(src_path)
        except Exception as e:
            log.warning("tagger: preview fail %s: %s", src_path, e)
            return out
        try:
            emb = aesthetic.encode_image(preview)
            state.cache_embedding(sha, emb)
            out["embedded"] = True
        except Exception as e:
            log.warning("tagger: embed fail %s: %s", src_path, e)
            return out
        try:
            from banger.sharpness import sharpness_from_preview
            sharp = sharpness_from_preview(preview)
            phash = dedup.phash_from_preview(preview)
            ts = dedup.best_timestamp(src_path)
            fc, fs = face.best_face_sharpness(preview)
            state.cache_frame_metadata(
                sha, sharp, str(phash), ts,
                face_count=fc, face_sharpness=fs,
            )
        except Exception as e:
            log.warning("tagger: side-cache fail %s: %s", src_path, e)

    if not have_tags:
        try:
            pairs = tags_mod.tag_from_embedding(emb)
            serial = [[t, round(s, 4)] for t, s in pairs]
            state.update_frame_metadata(sha, tags=serial)
            out["tagged"] = True
        except Exception as e:
            log.warning("tagger: tag compute fail %s: %s", src_path, e)

    if not full:
        return out

    # Full mode: face identity + scene cluster + taste score. Each is opt-in
    # and degrades gracefully if its dep isn't installed.
    if not have_faces:
        try:
            if preview is None:
                preview = load_preview(src_path)
            detections = face_id_mod.extract_face_detections(preview)
            payload = face_id_mod.encode_detections_for_cache(detections)
            state.update_frame_metadata(sha, face_detections=payload)
            out["faces"] = True
        except Exception as e:
            log.warning("tagger: face_id fail %s: %s", src_path, e)

    if not have_scene:
        try:
            sc = scene_kmeans.load()
            if sc is not None:
                info = sc.classify_embedding(emb)
                state.update_frame_metadata(
                    sha,
                    scene_cluster=int(info.cluster_id),
                    scene_preset=info.preset,
                )
                out["scene"] = True
        except Exception as e:
            log.warning("tagger: scene fail %s: %s", src_path, e)

    if not have_score:
        try:
            head = taste_head.load()
            if head is not None:
                score = taste_head.predict_score(head, emb)
                state.update_frame_metadata(sha, taste_score=round(float(score), 3))
                out["scored"] = True
        except Exception as e:
            log.warning("tagger: score fail %s: %s", src_path, e)

    return out


def _run(full: bool = False) -> None:
    p = _progress
    p.phase = "running"
    p.mode = "full" if full else "tags"
    p.started_at = time.monotonic()
    p.processed = 0
    p.embedded = 0
    p.tagged = 0
    p.faces_done = 0
    p.scored = 0
    p.skipped = 0
    p.error = None
    try:
        # Pull every frame from every root. We could batch by root for
        # better progress granularity but a single list is simpler.
        roots = library.all_roots()
        all_frames: list[tuple[str, str]] = []
        for r in roots:
            rows = library.query_frames(root_id=r.id, limit=100_000, offset=0)
            for row in rows:
                all_frames.append((row.sha, row.abs_path))
        p.total = len(all_frames)
        log.info("tagger: starting over %d frames across %d roots", p.total, len(roots))

        for sha, abs_path in all_frames:
            p.current = abs_path
            p.processed += 1
            try:
                r = _tag_one(sha, abs_path, full=full)
            except Exception as e_:
                log.warning("tagger: skip %s: %s", abs_path, e_)
                p.skipped += 1
                continue
            if not any(r.values()):
                p.skipped += 1
                continue
            if r.get("embedded"): p.embedded += 1
            if r.get("tagged"): p.tagged += 1
            if r.get("faces"): p.faces_done += 1
            if r.get("scored"): p.scored += 1

        p.phase = "done"
        p.finished_at = time.monotonic()
        p.current = None
        log.info(
            "tagger: finished %s in %.1fs (embedded=%d tagged=%d faces=%d scored=%d skipped=%d of %d)",
            p.mode, p.finished_at - p.started_at,
            p.embedded, p.tagged, p.faces_done, p.scored, p.skipped, p.total,
        )
    except Exception as e:
        log.exception("tagger crashed")
        p.phase = "error"
        p.error = str(e)
        p.finished_at = time.monotonic()


def start(full: bool = False) -> TaggerProgress:
    """Kick the tagger if not already running. Returns the progress object.

    `full=True` does the heavy enrichment pass (insightface faces, scene
    cluster, taste score) in addition to tags. Slow but means subsequent
    cull operations on the library are pure SQL/JSON reads.
    """
    global _progress
    with _lock:
        if _progress.phase == "running":
            return _progress
        _progress = TaggerProgress()
        _progress.mode = "full" if full else "tags"
        threading.Thread(target=_run, args=(full,), daemon=True).start()
        return _progress
