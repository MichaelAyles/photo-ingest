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
from dataclasses import dataclass

from banger import (
    aesthetic,
    dedup,
    face,
    library,
    scene_kmeans,
    state,
    taste_head,
)
from banger import (
    face_id as face_id_mod,
)
from banger import (
    tags as tags_mod,
)
from banger.preview import load_preview

log = logging.getLogger("banger.tagger")

# Frames are streamed from the library one page at a time so memory stays
# bounded regardless of library size (the old code accumulated every frame
# into one RAM list, silently capped at 100k). Within a page, CLIP previews
# are encoded in batches — the dominant scale win over one-image-at-a-time.
_PAGE_SIZE = 512
_CLIP_BATCH = 32


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


def _tag_one(sha: str, src_path, full: bool = False, *, emb=None, preview=None) -> dict:
    """Enrich one frame. Returns a dict of what was newly computed.

    Keys: embedded, tagged, faces, scored, scene. Each True only when that
    enrichment ran AND wasn't already cached. Idempotent.

    `full=True` adds insightface face detections, scene cluster assignment,
    and taste head scoring on top of the default tags-only pass. The full
    pass is slow (~1s/frame for insightface) but means later cull/sort
    operations don't need to touch the original file at all.

    `emb` / `preview` are optional pre-computed inputs. `_run` encodes CLIP
    embeddings in batches (the big scale win) and hands the resulting vector
    plus the already-decoded preview here so we don't re-embed or re-decode
    per frame. When both are None this falls back to the old single-frame
    path (still used by any direct caller / tests), embedding inline.
    """
    from pathlib import Path
    # Library passes abs_path as a string; load_preview wants Path so the
    # `.suffix` lookup works. Without this coercion the warm-path face_id
    # branch raises AttributeError silently inside its try, and 990 of
    # 1000 frames never actually get faces extracted.
    src_path = Path(src_path) if not isinstance(src_path, Path) else src_path
    meta = state.load_frame_metadata(sha) or {}
    # Single read of the cached embedding (was loaded twice before). `emb`
    # supplied by the batch path counts as "have it" without touching disk.
    cached_emb = None if emb is not None else state.load_embedding(sha)
    have_emb = emb is not None or cached_emb is not None
    have_tags = bool(meta.get("tags"))
    have_faces = "face_detections" in meta
    have_score = "taste_score" in meta
    have_scene = "scene_cluster" in meta

    if have_emb and have_tags and (not full or (have_faces and have_score and have_scene)):
        return {}

    out = {"embedded": False, "tagged": False, "faces": False, "scored": False, "scene": False}

    if emb is not None:
        # Freshly batch-encoded by _run; persist it and record the win. The
        # caller passes the matching preview so the side-cache below is free.
        state.cache_embedding(sha, emb)
        out["embedded"] = True
    elif cached_emb is not None:
        emb = cached_emb
    else:
        # Fallback single-frame path (no batch driver / direct caller).
        if preview is None:
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

    # Side-cache (sharpness / phash / timestamp / face) belongs with a fresh
    # embedding: it's the first time we've decoded this frame's pixels. Runs
    # whenever we just embedded (batch or fallback) and have the preview.
    if out["embedded"] and preview is not None:
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
            from banger import settings as settings_mod
            pairs = tags_mod.tag_from_embedding(emb, min_sim=float(settings_mod.get("tag_min_sim")))
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
            log.exception("tagger: face_id fail %s: %s", src_path, e)

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


def _needs_embedding(sha: str, full: bool) -> tuple[bool, bool]:
    """Cheap pre-check for one frame, mirroring _tag_one's skip logic.

    Returns (do_work, want_embed):
      do_work    - any enrichment is still missing (False => fully cached)
      want_embed - no .npy cached yet, so this frame needs a CLIP forward pass

    Lets _run decide which previews to decode + batch-encode without paying
    the per-frame embed cost itself.
    """
    meta = state.load_frame_metadata(sha) or {}
    have_emb = state.load_embedding(sha) is not None
    have_tags = bool(meta.get("tags"))
    if full:
        fully = have_emb and have_tags and (
            "face_detections" in meta and "taste_score" in meta and "scene_cluster" in meta
        )
    else:
        fully = have_emb and have_tags
    return (not fully, not have_emb)


def _process_page(frames: list[tuple[str, str]], full: bool, p: TaggerProgress) -> None:
    """Embed (in batches) + enrich one page of (sha, abs_path) frames.

    Frames already fully cached are counted as skipped without decoding.
    Frames missing only their embedding have their previews decoded and
    encoded together via aesthetic.encode_images_batch, then each frame is
    enriched with its in-hand embedding + preview so nothing is re-read.
    """
    from pathlib import Path

    # Phase 1: triage. Build the batch of previews to embed in one shot.
    batch_previews: list = []
    batch_idx: list[int] = []        # index into `frames` for each batched preview
    page_emb: dict[int, object] = {}      # frame index -> fresh embedding
    page_preview: dict[int, object] = {}  # frame index -> decoded preview
    skip_idx: set[int] = set()

    for i, (sha, abs_path) in enumerate(frames):
        do_work, want_embed = _needs_embedding(sha, full)
        if not do_work:
            skip_idx.add(i)
            continue
        if want_embed:
            try:
                preview = load_preview(Path(abs_path))
            except Exception as e:
                log.warning("tagger: preview fail %s: %s", abs_path, e)
                skip_idx.add(i)
                continue
            page_preview[i] = preview
            batch_previews.append(preview)
            batch_idx.append(i)

    # Phase 2: one batched CLIP forward pass for every preview in the page.
    if batch_previews:
        try:
            embs = aesthetic.encode_images_batch(batch_previews, batch_size=_CLIP_BATCH)
            for j, idx in enumerate(batch_idx):
                page_emb[idx] = embs[j]
        except Exception as e:
            # Batch failed wholesale; fall back to per-frame embedding inside
            # _tag_one (preview already decoded, so no double read).
            log.warning("tagger: batch embed fail (%d frames): %s", len(batch_previews), e)

    # Phase 3: per-frame enrichment, feeding the batched embedding + preview.
    for i, (sha, abs_path) in enumerate(frames):
        p.current = abs_path
        p.processed += 1
        if i in skip_idx:
            p.skipped += 1
            continue
        try:
            r = _tag_one(
                sha, abs_path, full=full,
                emb=page_emb.get(i), preview=page_preview.get(i),
            )
        except Exception as e_:
            log.warning("tagger: skip %s: %s", abs_path, e_)
            p.skipped += 1
            continue
        if not any(r.values()):
            p.skipped += 1
            continue
        if r.get("embedded"):
            p.embedded += 1
        if r.get("tagged"):
            p.tagged += 1
        if r.get("faces"):
            p.faces_done += 1
        if r.get("scored"):
            p.scored += 1


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
        # Stream frames root-by-root in fixed-size pages. count_frames gives
        # the up-front total for progress; query_frames(offset=...) walks each
        # root without ever holding more than one page in RAM. (The old code
        # built one giant list capped at 100k, truncating bigger libraries.)
        roots = library.all_roots()
        p.total = sum(library.count_frames(root_id=r.id) for r in roots)
        log.info("tagger: starting over %d frames across %d roots", p.total, len(roots))

        for r in roots:
            offset = 0
            while True:
                rows = library.query_frames(root_id=r.id, limit=_PAGE_SIZE, offset=offset)
                if not rows:
                    break
                page = [(row.sha, row.abs_path) for row in rows]
                _process_page(page, full, p)
                if len(rows) < _PAGE_SIZE:
                    break
                offset += _PAGE_SIZE

        # Full pass implies the user wants every signal computed. Cluster
        # face embeddings into named-or-anonymous people so the library shows
        # everyone, not just frames the user manually opened. Cheap on top of
        # the per-frame work that just finished.
        if full:
            try:
                from banger import face_names as fn
                summary = fn.discover_clusters()
                log.info(
                    "tagger: discovered people: added=%d matched=%d clusters=%d faces=%d",
                    summary["added"], summary["matched"],
                    summary["total_clusters"], summary["total_faces"],
                )
            except Exception as e:
                log.warning("tagger: discover failed: %s", e)

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
