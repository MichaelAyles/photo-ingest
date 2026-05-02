"""On-disk state: cached CLIP embeddings, thumbnails, per-frame scores, taste head.

State lives at `~/.local/share/banger-pipeline/` on every OS — works fine on
Windows too (just creates the dir under the user's home).
"""

import hashlib
import sqlite3
import time
from pathlib import Path

import numpy as np

STATE_DIR = Path.home() / ".local" / "share" / "banger-pipeline"
EMBEDDINGS_DIR = STATE_DIR / "embeddings"
THUMBS_DIR = STATE_DIR / "thumbs"
LABELS_DB = STATE_DIR / "labels.db"
TASTE_HEAD = STATE_DIR / "taste_head.joblib"

SCORE_MIN = -5
SCORE_MAX = 5


def _ensure_dirs() -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)


def _conn() -> sqlite3.Connection:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(LABELS_DB)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(labels)")]
    if cols and "score" not in cols:
        # v1 schema (label TEXT 'up'/'down') → v2 (score INTEGER -5..+5).
        conn.execute("ALTER TABLE labels RENAME TO labels_v1")
        conn.execute(
            """
            CREATE TABLE labels (
                sha256 TEXT PRIMARY KEY,
                score INTEGER NOT NULL,
                stem TEXT NOT NULL,
                src_path TEXT NOT NULL,
                ts INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO labels (sha256, score, stem, src_path, ts)
            SELECT sha256,
                   CASE WHEN label='up' THEN 5 ELSE -5 END,
                   stem, src_path, ts
            FROM labels_v1
            """
        )
        conn.execute("DROP TABLE labels_v1")
        conn.commit()
    elif not cols:
        conn.execute(
            """
            CREATE TABLE labels (
                sha256 TEXT PRIMARY KEY,
                score INTEGER NOT NULL,
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


def cache_thumbnail(sha: str, jpeg_bytes: bytes) -> None:
    _ensure_dirs()
    (THUMBS_DIR / f"{sha}.jpg").write_bytes(jpeg_bytes)


def thumbnail_path(sha: str) -> Path:
    return THUMBS_DIR / f"{sha}.jpg"


def add_label(sha: str, score: int, stem: str, src_path: str) -> None:
    if not SCORE_MIN <= score <= SCORE_MAX:
        raise ValueError(f"score must be in [{SCORE_MIN}, {SCORE_MAX}], got {score}")
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO labels (sha256, score, stem, src_path, ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (sha, score, stem, src_path, int(time.time())),
        )


def get_label(sha: str) -> int | None:
    with _conn() as conn:
        row = conn.execute("SELECT score FROM labels WHERE sha256=?", (sha,)).fetchone()
    return int(row[0]) if row else None


def all_labels() -> list[tuple[str, int, str, str, int]]:
    with _conn() as conn:
        return [
            (sha, int(score), stem, path, ts)
            for sha, score, stem, path, ts in conn.execute(
                "SELECT sha256, score, stem, src_path, ts FROM labels ORDER BY ts"
            )
        ]


def labels_dict() -> dict[str, int]:
    return {sha: score for sha, score, _, _, _ in all_labels()}
