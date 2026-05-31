"""On-disk state: cached CLIP embeddings, thumbnails, per-frame scores, taste head.

State lives at `~/.local/share/banger-pipeline/` on every OS — works fine on
Windows too (just creates the dir under the user's home).
"""

import hashlib
import os
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Per-sha lock for the metadata sidecar read-modify-write cycle. Two
# pipeline threads can otherwise interleave: A reads, B reads, A writes, B
# writes -> A's fields are silently dropped. SQLite is already serialized
# by SQLite itself; this lock is only for the JSON sidecars and the .npy
# embedding files.
_sha_locks_guard = threading.Lock()
_sha_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)


def _sha_lock(sha: str) -> threading.Lock:
    with _sha_locks_guard:
        return _sha_locks[sha]

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


# labels.db is the only non-reconstructible user data, and Flask serves with
# threaded=True — so connections come from many threads. We open one
# connection per caller (sqlite3 handles are not thread-safe to share) but run
# the schema setup / v1->v2 migration EXACTLY ONCE, behind a module-level lock,
# mirroring the _init_lock/_inited pattern in library.py. Steady-state opens
# then skip the PRAGMA table_info probe and the migration entirely.
#
# We key the "inited" flag on the DB path (not a bare bool) so that tests which
# monkeypatch LABELS_DB to a fresh tmp_path each get their schema built; the
# real app only ever sees one path so this stays a single init in production.
_init_lock = threading.Lock()
_inited_db: str | None = None


def _migrate_and_init(conn: sqlite3.Connection) -> None:
    """Create the labels table (and run the v1->v2 score migration) once.

    Wrapped in an explicit transaction so the rename/insert/drop dance is
    all-or-nothing — a crash mid-migration can't leave a half-converted DB.
    """
    cols = [r[1] for r in conn.execute("PRAGMA table_info(labels)")]
    conn.execute("BEGIN")
    try:
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
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def _conn() -> sqlite3.Connection:
    """Open a labels.db connection configured for concurrent, durable use.

    WAL lets one writer coexist with many readers (Flask is threaded);
    synchronous=NORMAL is the safe+fast pairing for WAL; busy_timeout makes a
    contended write wait rather than instantly raising "database is locked".
    check_same_thread=False is required because the caller's `with` block may
    run on a different thread than the one that opened it.
    """
    global _inited_db
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(LABELS_DB, check_same_thread=False)
    # Cheap per-connection PRAGMAs (journal_mode is persisted in the DB header,
    # but setting it is idempotent and harmless to repeat).
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")

    db_key = str(LABELS_DB)
    if _inited_db != db_key:
        with _init_lock:
            if _inited_db != db_key:
                _migrate_and_init(conn)
                _inited_db = db_key
    return conn


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` atomically: temp sibling + fsync + os.replace.

    A crash or a concurrent writer can never leave a truncated sidecar at the
    real name — readers see either the old file or the fully-written new one.
    The temp sibling lives in the same directory so os.replace is atomic.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fp:
            fp.write(data)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_embedding(sha: str, emb: np.ndarray) -> None:
    _ensure_dirs()
    import io

    # Serialize via np.save into a buffer, then publish atomically so a crash /
    # concurrent writer never leaves a truncated .npy at the real name.
    buf = io.BytesIO()
    np.save(buf, emb.astype(np.float32))
    with _sha_lock(sha):
        _atomic_write_bytes(EMBEDDINGS_DIR / f"{sha}.npy", buf.getvalue())


# Public alias for the shared contract: save_embedding == cache_embedding.
# Other wave-2 code refers to save_embedding; keep cache_embedding too.
def save_embedding(sha: str, emb: np.ndarray) -> None:
    cache_embedding(sha, emb)


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
    sha: str,
    sharpness: float,
    phash_hex: str,
    timestamp: float,
    face_count: int | None = None,
    face_sharpness: float | None = None,
    metrics: dict | None = None,
    eyes: dict | None = None,
    face_embeddings: list | None = None,
    face_detections: list | None = None,
) -> None:
    """Cache the per-frame inputs needed by `cmd_run` so re-runs skip preview loads.

    `metrics` is the cv2-only multi-dim dict from banger.metrics. `eyes` is the
    optional per-face EAR/blink dict from banger.eyes. Both are merged into
    the JSON so downstream consumers (report, XMP encoder, UI) can read them
    without recomputing.
    """
    import json

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "sharpness": float(sharpness),
        "phash_hex": phash_hex,
        "timestamp": float(timestamp),
        "computed_at": int(time.time()),
    }
    if face_count is not None:
        payload["face_count"] = int(face_count)
    if face_sharpness is not None:
        payload["face_sharpness"] = float(face_sharpness)
    if metrics:
        payload["metrics"] = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
    if eyes:
        payload["eyes"] = eyes
    if face_embeddings:
        payload["face_embeddings"] = face_embeddings
    if face_detections:
        payload["face_detections"] = face_detections
    with _sha_lock(sha):
        _atomic_write_bytes(METADATA_DIR / f"{sha}.json", json.dumps(payload).encode("utf-8"))


def update_frame_metadata(sha: str, **fields) -> None:
    """Merge fields into an existing metadata file. Used by lazy enrichers
    (captions, recomputed metrics) that produce data after the initial
    pipeline pass. No-op if the metadata file doesn't exist yet."""
    import json

    p = METADATA_DIR / f"{sha}.json"
    with _sha_lock(sha):
        if not p.exists():
            return
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        payload.update(fields)
        _atomic_write_bytes(p, json.dumps(payload).encode("utf-8"))


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


# Public alias: the canonical name has long been add_label, but `set_label`
# reads better at call sites and is referenced by the shared contract. Same
# signature/behavior (INSERT OR REPLACE), so it's a true synonym.
def set_label(sha: str, score: int, stem: str, src_path: str) -> None:
    add_label(sha, score, stem, src_path)


def integrity_check() -> bool:
    """Run SQLite's PRAGMA integrity_check on labels.db (and library.db if it
    exists). Returns True iff every DB reports "ok". Cheap to call; useful as a
    health probe before a backup or after a hard crash. Never raises on a
    missing/locked DB — it just reports False for that DB.
    """
    ok = True
    dbs = [LABELS_DB]
    # library.db lives in the same STATE_DIR and is the other persistent store.
    library_db = STATE_DIR / "library.db"
    if library_db.exists():
        dbs.append(library_db)
    for db in dbs:
        if not Path(db).exists():
            continue
        try:
            conn = sqlite3.connect(db, check_same_thread=False)
            try:
                rows = conn.execute("PRAGMA integrity_check").fetchall()
            finally:
                conn.close()
        except sqlite3.Error:
            ok = False
            continue
        if not (len(rows) == 1 and rows[0][0] == "ok"):
            ok = False
    return ok


def backup_labels(dest: Path | None = None) -> Path:
    """Dump labels.db to a timestamped CSV and return its path.

    labels.db is the only irreplaceable user data, but it's tiny, so a plain
    CSV snapshot is the most portable, future-proof backup (re-importable via
    import_labels_csv). Defaults to STATE_DIR/backups/labels-YYYYmmdd-HHMMSS.csv.
    The CSV itself is written atomically (export_labels_csv writes the whole
    file in one open()) so a partially-written backup is never published under
    a name a caller might trust.
    """
    if dest is None:
        backups_dir = STATE_DIR / "backups"
        backups_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        dest = backups_dir / f"labels-{stamp}.csv"
    else:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
    export_labels_csv(dest)
    return dest


def cache_stats() -> dict[str, dict]:
    """Return per-cache (count, total_bytes) for the on-disk caches we own."""
    out: dict[str, dict] = {}
    for name, dir_path, suffix in (
        ("embeddings", EMBEDDINGS_DIR, ".npy"),
        ("thumbs", THUMBS_DIR, ".jpg"),
        ("previews", PREVIEWS_DIR, ".jpg"),
        ("metadata", METADATA_DIR, ".json"),
    ):
        if dir_path.exists():
            files = list(dir_path.glob(f"*{suffix}"))
            out[name] = {
                "count": len(files),
                "bytes": sum(p.stat().st_size for p in files),
                "path": str(dir_path),
            }
        else:
            out[name] = {"count": 0, "bytes": 0, "path": str(dir_path)}
    return out


def clear_cache(kinds: list[str]) -> dict[str, int]:
    """Delete files in the named caches. Returns count removed per kind.

    Valid kinds: 'embeddings', 'thumbs', 'previews', 'metadata'. Labels are
    NOT touched — they're not a cache, they're authored input.
    """
    targets = {
        "embeddings": (EMBEDDINGS_DIR, "*.npy"),
        "thumbs": (THUMBS_DIR, "*.jpg"),
        "previews": (PREVIEWS_DIR, "*.jpg"),
        "metadata": (METADATA_DIR, "*.json"),
    }
    removed: dict[str, int] = {}
    for k in kinds:
        if k not in targets:
            raise ValueError(f"unknown cache kind: {k}")
        dir_path, glob = targets[k]
        if not dir_path.exists():
            removed[k] = 0
            continue
        n = 0
        for p in dir_path.glob(glob):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
        removed[k] = n
    return removed


def export_labels_csv(path: Path) -> int:
    """Write labels.db to a CSV at path. Returns count written.

    Written atomically (build in memory, fsync temp sibling, os.replace) so a
    crash mid-export — or a reader racing the writer — never sees a truncated
    backup at the real name.
    """
    import csv
    import io

    rows = all_labels()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["sha256", "score", "stem", "src_path", "ts"])
    for sha, score, stem, src, ts in rows:
        writer.writerow([sha, score, stem, src, ts])
    _atomic_write_bytes(Path(path), buf.getvalue().encode("utf-8"))
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
