"""On-disk state: cached CLIP embeddings, per-frame labels, trained taste head.

State lives at `~/.local/share/banger-pipeline/` on every OS — works fine on
Windows too (just creates the dir under the user's home). Per CLAUDE.md.
"""

import hashlib
import sqlite3
import time
from pathlib import Path

import numpy as np

STATE_DIR = Path.home() / ".local" / "share" / "banger-pipeline"
EMBEDDINGS_DIR = STATE_DIR / "embeddings"
LABELS_DB = STATE_DIR / "labels.db"
TASTE_HEAD = STATE_DIR / "taste_head.joblib"


def _ensure_dirs() -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


def _conn() -> sqlite3.Connection:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(LABELS_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS labels (
            sha256 TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            stem TEXT NOT NULL,
            src_path TEXT NOT NULL,
            ts INTEGER NOT NULL
        )
        """
    )
    return conn


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_embedding(sha: str, emb: np.ndarray) -> None:
    _ensure_dirs()
    np.save(EMBEDDINGS_DIR / f"{sha}.npy", emb.astype(np.float32))


def load_embedding(sha: str) -> np.ndarray | None:
    p = EMBEDDINGS_DIR / f"{sha}.npy"
    if not p.exists():
        return None
    return np.load(p)


def add_label(sha: str, label: str, stem: str, src_path: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO labels (sha256, label, stem, src_path, ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (sha, label, stem, src_path, int(time.time())),
        )


def all_labels() -> list[tuple[str, str, str, str, int]]:
    with _conn() as conn:
        return list(
            conn.execute(
                "SELECT sha256, label, stem, src_path, ts FROM labels ORDER BY ts"
            )
        )


def label_counts() -> tuple[int, int]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT label, COUNT(*) FROM labels GROUP BY label"
        ).fetchall()
    counts = dict(rows)
    return int(counts.get("up", 0)), int(counts.get("down", 0))
