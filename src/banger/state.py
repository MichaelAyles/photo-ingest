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
PREVIEWS_DIR = STATE_DIR / "previews"
METADATA_DIR = STATE_DIR / "metadata"
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


def cache_preview_jpeg(sha: str, jpeg_bytes: bytes) -> None:
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    (PREVIEWS_DIR / f"{sha}.jpg").write_bytes(jpeg_bytes)


def preview_jpeg_path(sha: str) -> Path:
    return PREVIEWS_DIR / f"{sha}.jpg"


def cache_frame_metadata(
    sha: str, sharpness: float, phash_hex: str, timestamp: float
) -> None:
    """Cache the per-frame inputs needed by `cmd_run` so re-runs skip preview loads."""
    import json

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "sharpness": float(sharpness),
        "phash_hex": phash_hex,
        "timestamp": float(timestamp),
        "computed_at": int(time.time()),
    }
    (METADATA_DIR / f"{sha}.json").write_text(json.dumps(payload), encoding="utf-8")


def load_frame_metadata(sha: str) -> dict | None:
    import json

    p = METADATA_DIR / f"{sha}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


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


def export_labels_csv(path: Path) -> int:
    """Write labels.db to a CSV at path. Returns count written."""
    import csv

    rows = all_labels()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["sha256", "score", "stem", "src_path", "ts"])
        for sha, score, stem, src, ts in rows:
            writer.writerow([sha, score, stem, src, ts])
    return len(rows)


def import_labels_csv(path: Path) -> tuple[int, int]:
    """Read CSV at path and INSERT-OR-REPLACE into labels. Returns (count_imported, count_skipped)."""
    import csv

    imported = 0
    skipped = 0
    with open(path, encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                sha = (row.get("sha256") or "").strip()
                score = int(row.get("score") or "")
                if not (SCORE_MIN <= score <= SCORE_MAX) or not sha:
                    skipped += 1
                    continue
                stem = (row.get("stem") or "").strip()
                src = (row.get("src_path") or "").strip()
                if not stem or not src:
                    skipped += 1
                    continue
            except (KeyError, ValueError, TypeError):
                skipped += 1
                continue
            add_label(sha, score, stem, src)
            imported += 1
    return imported, skipped
