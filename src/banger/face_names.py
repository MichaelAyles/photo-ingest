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
from pathlib import Path

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


def assign(name: str, embedding: np.ndarray) -> tuple[float, int]:
    """Attach `embedding` to `name`, averaging into the existing centroid if any.

    Returns (cosine sim of embedding to the previous centroid before merge,
    new count). The sim is useful for the UI to surface "this looks like
    a strong match" vs "first sample, no comparison" cases.
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
            "SELECT centroid, count FROM face_names WHERE name=?", (name,)
        ).fetchone()
        if row is None:
            blob = embedding.tobytes()
            conn.execute(
                "INSERT INTO face_names (name, centroid, count, updated) VALUES (?, ?, 1, ?)",
                (name, blob, int(time.time())),
            )
            return 0.0, 1

        prev_blob, prev_count = row
        prev = np.frombuffer(prev_blob, dtype=np.float32).copy()
        sim = float(embedding @ prev / max(float(np.linalg.norm(prev)), 1e-8))
        # Weighted average of unit vectors, renormalised.
        new = prev * prev_count + embedding
        new = new / max(float(np.linalg.norm(new)), 1e-8)
        new_count = int(prev_count) + 1
        conn.execute(
            "UPDATE face_names SET centroid=?, count=?, updated=? WHERE name=?",
            (new.astype(np.float32).tobytes(), new_count, int(time.time()), name),
        )
        return sim, new_count


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
