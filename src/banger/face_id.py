"""Face identity extraction and clustering, the Aftershoot cheat.

Aftershoot, Narrative Select, and Optyx all share the same selection
trick: cluster photos by who appears in them, then surface one good
shot per person. Banger's prompt-based aesthetic + kmeans-on-CLIP
captures visual diversity but not identity diversity. If the same crew
is in every well-lit caminito frame and a few mountain frames, the head
will rank caminito frames higher and the visual-kmeans clusters all
sit in the same "people in light" subspace, so picks skew toward the
well-lit folder.

This module fixes that by extracting 512-dim face embeddings via
insightface (the same buffalo_l model facet uses), clustering them
across the candidate pool with DBSCAN on cosine distance, and feeding
the per-frame person-id set into a face-aware selector.

Why insightface: it ships a calibrated ArcFace embedding so cosine
distance ~ 0.5 is a robust "same person" cutoff, way more reliable than
DIY-ing one with a smaller face model. The first run downloads ~280 MB
of model weights to ~/.insightface/, cached forever after.

Why DBSCAN: agglomerative would also work but DBSCAN handles "noise"
(false-positive face detections, weird angles) by leaving outliers
unclustered, which we want, rogue side-of-head detections shouldn't
become their own "person."
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

log = logging.getLogger("banger.face_id")


# Cosine distance threshold for "same person." ArcFace embeddings are
# L2-normalised so cosine distance == 1 - dot. 0.5 is the value the
# insightface community settled on; tighter cuts the same-person recall,
# looser merges similar-looking people into one cluster.
SAME_PERSON_COS_DIST = 0.5
MIN_SAMPLES = 1  # one face is enough to be a cluster; otherwise everything's "noise"


def insightface_available() -> bool:
    try:
        import insightface  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _app():
    """Load the buffalo_l detector + recogniser once per process.

    First call downloads ~280 MB of model weights from the insightface
    GitHub release if they aren't already on disk. Subsequent calls
    return the cached app. We force CPU here because the user's box
    has 4 GB GPU shared with CLIP; insightface is fast enough on CPU
    (~80 ms per 1024 px preview on a modern laptop).
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        return None
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def extract_face_embeddings(preview_bgr: np.ndarray) -> list[np.ndarray]:
    """Return one L2-normalised 512-dim embedding per detected face, or []."""
    app = _app()
    if app is None or preview_bgr is None or preview_bgr.size == 0:
        return []
    try:
        faces = app.get(preview_bgr)
    except Exception as e:
        log.warning("insightface get() failed: %s", e)
        return []
    out: list[np.ndarray] = []
    for f in faces:
        if not hasattr(f, "embedding") or f.embedding is None:
            continue
        emb = np.asarray(f.embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm <= 0:
            continue
        out.append(emb / norm)
    return out


def encode_for_cache(embeddings: list[np.ndarray]) -> list[list[float]]:
    """Serialise embeddings for the JSON metadata cache (list of float lists)."""
    return [emb.astype(np.float32).tolist() for emb in embeddings]


def decode_from_cache(payload) -> list[np.ndarray]:
    """Reverse of encode_for_cache. Tolerates missing/legacy formats."""
    if not payload:
        return []
    out: list[np.ndarray] = []
    for entry in payload:
        try:
            arr = np.asarray(entry, dtype=np.float32)
            if arr.ndim == 1 and arr.size > 0:
                out.append(arr)
        except (TypeError, ValueError):
            continue
    return out


def cluster_faces(
    per_frame_embs: list[list[np.ndarray]],
    eps: float = SAME_PERSON_COS_DIST,
) -> list[set[int]]:
    """DBSCAN over the flat embedding list. Returns one set-of-person-ids per frame.

    `per_frame_embs[i]` is the list of face embeddings for frame i.
    Output `clusters[i]` is the set of cluster ids assigned to frame i; an
    empty set means no faces detected (or all detections were DBSCAN-noise).
    Cluster id -1 (DBSCAN's noise label) is dropped, so isolated faces
    don't bloat the "people in this batch" count.
    """
    flat: list[np.ndarray] = []
    owners: list[int] = []  # which frame each flat embedding belongs to
    for frame_idx, embs in enumerate(per_frame_embs):
        for e in embs:
            flat.append(e)
            owners.append(frame_idx)

    out: list[set[int]] = [set() for _ in per_frame_embs]
    if not flat:
        return out

    from sklearn.cluster import DBSCAN

    X = np.stack(flat)
    db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, metric="cosine")
    labels = db.fit_predict(X)
    for frame_idx, cid in zip(owners, labels.tolist(), strict=True):
        if cid >= 0:
            out[frame_idx].add(cid)
    return out


def person_count(clusters: list[set[int]]) -> int:
    """How many distinct people DBSCAN found across the batch."""
    return len({cid for s in clusters for cid in s})
