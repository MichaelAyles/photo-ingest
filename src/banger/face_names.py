"""Persistent face name database.

The faces strategy already clusters faces per-batch with DBSCAN. That gives
us "Person A appears in frames X, Y, Z within this run" but not "this is
Alice and she's the same person we saw in last week's hike too." Naming
needs to survive across runs, which means storing per-name centroid
embeddings on disk and matching new face embeddings against them at query
time.

Storage: one SQLite table at ~/.local/share/banger-pipeline/faces.db.
  - name      TEXT primary key
  - centroid  BLOB (512-float32, L2-normalised)
  - count     INTEGER (how many embeddings have been merged into this centroid)
  - updated   INTEGER (unix ts)

Matching: new embedding E is matched against every centroid C by cosine
similarity (== dot product on unit vectors). The best C wins if its sim
exceeds MATCH_THRESHOLD. The default 0.45 is tighter than DBSCAN's 0.5
because false-matching across users matters more here (you don't want
"Bob" to silently absorb a stranger).

When the user assigns a name, we update the centroid by averaging the
new embedding into it (weighted by count) so the centroid stabilises as
more examples accumulate.
"""

from __future__ import annotations

import logging
import sqlite3
import time

import numpy as np

from banger import state

log = logging.getLogger("banger.face_names")

FACE_DB = state.STATE_DIR / "faces.db"
MATCH_THRESHOLD = 0.45
EMB_DIM = 512


def _conn() -> sqlite3.Connection:
    state.STATE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(FACE_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS face_names (
            name     TEXT PRIMARY KEY,
            centroid BLOB NOT NULL,
            count    INTEGER NOT NULL DEFAULT 1,
            updated  INTEGER NOT NULL
        )
        """
    )
    # Migration: add thumbnail column if missing. Stored as JPEG bytes so the
    # library view doesn't have to re-crop from the original frame each time.
    cols = [r[1] for r in conn.execute("PRAGMA table_info(face_names)")]
    if "thumbnail" not in cols:
        conn.execute("ALTER TABLE face_names ADD COLUMN thumbnail BLOB")
        conn.commit()
    return conn


def _all() -> list[tuple[str, np.ndarray, int]]:
    """Read every (name, centroid, count) tuple from disk."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT name, centroid, count FROM face_names"
        ).fetchall()
    out: list[tuple[str, np.ndarray, int]] = []
    for name, blob, count in rows:
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size != EMB_DIM:
            continue
        out.append((name, arr.copy(), int(count)))
    return out


def match(embedding: np.ndarray, threshold: float = MATCH_THRESHOLD) -> tuple[str | None, float]:
    """Return (name, sim) of the best match above threshold, or (None, best_sim).

    `embedding` must already be L2-normalised. We don't normalise here because
    the face_id module always produces unit vectors; double-normalising is
    cheap but the assumption surfaces bugs faster.
    """
    rows = _all()
    if not rows:
        return None, 0.0
    sims = []
    for name, centroid, _count in rows:
        sims.append((name, float(embedding @ centroid)))
    sims.sort(key=lambda kv: -kv[1])
    best_name, best_sim = sims[0]
    if best_sim >= threshold:
        return best_name, best_sim
    return None, best_sim


def assign(
    name: str,
    embedding: np.ndarray,
    thumbnail_jpeg: bytes | None = None,
) -> tuple[float, int]:
    """Attach `embedding` to `name`, averaging into the existing centroid if any.

    Returns (cosine sim of embedding to the previous centroid before merge,
    new count). The sim is useful for the UI to surface "this looks like
    a strong match" vs "first sample, no comparison" cases.

    `thumbnail_jpeg` (optional) is the cropped face image used by the library
    view. We only store one per name, set on first assign and refreshed when
    explicitly passed; the library doesn't try to pick a "best" thumbnail
    automatically since taste in faces is too subjective.
    """
    name = name.strip()
    if not name:
        raise ValueError("name must be non-empty")
    if embedding.shape != (EMB_DIM,):
        raise ValueError(f"embedding must be shape ({EMB_DIM},); got {embedding.shape}")
    embedding = embedding.astype(np.float32)
    embedding = embedding / max(float(np.linalg.norm(embedding)), 1e-8)

    with _conn() as conn:
        row = conn.execute(
            "SELECT centroid, count, thumbnail FROM face_names WHERE name=?", (name,)
        ).fetchone()
        if row is None:
            blob = embedding.tobytes()
            conn.execute(
                "INSERT INTO face_names (name, centroid, count, updated, thumbnail) "
                "VALUES (?, ?, 1, ?, ?)",
                (name, blob, int(time.time()), thumbnail_jpeg),
            )
            return 0.0, 1

        prev_blob, prev_count, prev_thumb = row
        prev = np.frombuffer(prev_blob, dtype=np.float32).copy()
        sim = float(embedding @ prev / max(float(np.linalg.norm(prev)), 1e-8))
        # Weighted average of unit vectors, renormalised.
        new = prev * prev_count + embedding
        new = new / max(float(np.linalg.norm(new)), 1e-8)
        new_count = int(prev_count) + 1
        # Replace thumbnail only when explicitly passed AND none stored yet,
        # or when the caller wants to update it.
        thumb_to_store = thumbnail_jpeg if thumbnail_jpeg is not None else prev_thumb
        conn.execute(
            "UPDATE face_names SET centroid=?, count=?, updated=?, thumbnail=? "
            "WHERE name=?",
            (new.astype(np.float32).tobytes(), new_count, int(time.time()),
             thumb_to_store, name),
        )
        return sim, new_count


def set_thumbnail(name: str, thumbnail_jpeg: bytes) -> bool:
    """Replace the stored thumbnail for an existing name."""
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE face_names SET thumbnail=? WHERE name=?",
            (thumbnail_jpeg, name),
        )
        return cur.rowcount > 0


def get_thumbnail(name: str) -> bytes | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT thumbnail FROM face_names WHERE name=?", (name,)
        ).fetchone()
    return bytes(row[0]) if row and row[0] is not None else None


def rename(old: str, new: str) -> bool:
    new = new.strip()
    if not new:
        return False
    with _conn() as conn:
        cur = conn.execute("UPDATE face_names SET name=? WHERE name=?", (new, old))
        return cur.rowcount > 0


def forget(name: str) -> bool:
    with _conn() as conn:
        cur = conn.execute("DELETE FROM face_names WHERE name=?", (name,))
        return cur.rowcount > 0


def all_names() -> list[dict]:
    """For the settings panel / API: list every name + sample count + last update."""
    return [
        {"name": name, "count": count}
        for name, _, count in _all()
    ]


def _next_auto_name() -> str:
    """Find the next free 'Person N' slot for auto-discovered clusters."""
    existing = {n for n, _, _ in _all()}
    i = 1
    while True:
        candidate = f"Person {i}"
        if candidate not in existing:
            return candidate
        i += 1


def discover_clusters(eps: float = 0.5, min_samples: int = 3) -> dict:
    """DBSCAN every cached face embedding across the library, match each
    cluster to a named centroid where possible, and auto-create unnamed
    'Person N' entries for the rest.

    Returns a summary dict: {added, matched, total_clusters, total_faces}.

    The eps is the cosine-distance same-person threshold for clustering, same
    as banger.face_id. We hold min_samples=3 to avoid creating a face entry
    from a single weak detection; one-off faces stay anonymous and the user
    can name them manually via the detail overlay.
    """
    import json

    from banger import state

    # Collect every face embedding from every metadata file.
    embs: list[np.ndarray] = []
    owners: list[tuple[str, int, list[int]]] = []  # (sha, face_idx, bbox)
    meta_dir = state.METADATA_DIR
    if not meta_dir.exists():
        return {"added": 0, "matched": 0, "total_clusters": 0, "total_faces": 0}

    for p in meta_dir.glob("*.json"):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        dets = meta.get("face_detections") or []
        sha = p.stem
        for idx, d in enumerate(dets):
            e = d.get("embedding") if isinstance(d, dict) else d
            if not e:
                continue
            arr = np.asarray(e, dtype=np.float32)
            if arr.size != EMB_DIM:
                continue
            embs.append(arr)
            owners.append((sha, idx, d.get("bbox") if isinstance(d, dict) else None))

    if not embs:
        return {"added": 0, "matched": 0, "total_clusters": 0, "total_faces": 0}

    from sklearn.cluster import DBSCAN
    X = np.stack(embs)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = db.fit_predict(X)

    # Existing named centroids: cluster ids that match one of these go to that name.
    named = _all()

    # Group by cluster id (skip noise label -1).
    by_cluster: dict[int, list[int]] = {}
    for i, cid in enumerate(labels.tolist()):
        if cid < 0:
            continue
        by_cluster.setdefault(cid, []).append(i)

    added = 0
    matched = 0
    for _cid, member_indices in by_cluster.items():
        # Centroid = mean of L2-normalised embeddings, re-normalised.
        cluster_embs = X[member_indices]
        centroid = cluster_embs.mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-8)

        # Does this cluster match any existing named centroid?
        existing_match = None
        for name, named_centroid, _ in named:
            sim = float(centroid @ named_centroid)
            if sim >= (1.0 - eps):
                existing_match = name
                break

        if existing_match is not None:
            # Merge cluster members into the existing centroid by re-running
            # assign once with the cluster mean. count++ once is sufficient
            # signal; we don't multiply-add to avoid over-weighting.
            try:
                assign(existing_match, centroid)
                matched += 1
            except ValueError:
                pass
            continue

        # Auto-create a Person N entry. Pick a representative embedding (the
        # one closest to the centroid) and try to grab its thumbnail from the
        # owning frame for the library view.
        sims = cluster_embs @ centroid
        best = int(np.argmax(sims))
        rep_sha, rep_idx, rep_bbox = owners[member_indices[best]]
        thumb_bytes = _try_crop_thumbnail(rep_sha, rep_bbox)
        try:
            new_name = _next_auto_name()
            assign(new_name, centroid, thumbnail_jpeg=thumb_bytes)
            # The first assign() puts the centroid; we then bump count to
            # reflect the cluster size so the library view shows reality.
            with _conn() as conn:
                conn.execute(
                    "UPDATE face_names SET count=? WHERE name=?",
                    (len(member_indices), new_name),
                )
            added += 1
            named = _all()  # refresh so subsequent clusters can match this one
        except ValueError:
            continue

    return {
        "added": added,
        "matched": matched,
        "total_clusters": len(by_cluster),
        "total_faces": len(embs),
    }


def _try_crop_thumbnail(sha: str, bbox: list[int] | None) -> bytes | None:
    """Best-effort crop of a face thumbnail from the frame's preview.

    Used by discover_clusters to give each auto-discovered person a
    representative image in the face library. Returns None on any failure
    (missing preview, bad bbox, etc.) and the entry just gets the default
    no-thumbnail look.
    """
    if not bbox or len(bbox) != 4:
        return None
    try:
        import cv2

        from banger import library  # noqa: PLC0415
        from banger.preview import load_preview  # noqa: PLC0415
        src = library.lookup_path(sha)
        if src is None:
            return None
        img = load_preview(src)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        pw, ph = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(w, x2 + pw)
        y2 = min(h, y2 + ph)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = cv2.resize(img[y1:y2, x1:x2], (128, 128), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None
    except Exception:
        return None
