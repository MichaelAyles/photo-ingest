"""Persistent multi-root photo library.

banger started as a single-folder culler: point at a directory, get top
picks, done. Now it's growing into a real browser. That needs an index
that survives across runs, spans multiple folders, knows about every
camera body's RAW dialect, and lets us filter the world by face / tag /
camera / date / rating without re-walking the filesystem every time.

This module is the storage layer. Everything user-facing (the Library
tab in the GUI) reads through it. The scoring / face / tag pipelines
keep writing to metadata/<sha>.json as before; this index just stitches
those files together with filesystem reality and lets the UI query.

Schema:
  roots(id, path, label, added_at, last_scanned)
    one row per watched folder. The id is referenced by frames.
  frames(sha PRIMARY KEY, root_id, rel_path, stem, kind, mtime,
         camera_model, taken_at, indexed_at)
    one row per photo file. camera_model + taken_at come from EXIF on
    first index, cached forever so listing 50k frames is one query.

Scanning is incremental: we hash a file only when its (rel_path, mtime)
changes vs the indexed row. New files get added, removed files get
deleted, untouched files are skipped. A full re-index of 1k frames is
~3-5 s.

Threading: SQLite is fine with one writer + many readers via WAL mode.
The scan job is a background thread; the API reads can run concurrently
without locking the writer out.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from banger import state
from banger.frames import discover_frames
from banger.preview import JPEG_SUFFIXES, RAW_SUFFIXES

log = logging.getLogger("banger.library")

LIBRARY_DB = state.STATE_DIR / "library.db"
SUPPORTED_SUFFIXES = JPEG_SUFFIXES | RAW_SUFFIXES


@dataclass
class Root:
    id: int
    path: str
    label: str
    added_at: int
    last_scanned: int | None


@dataclass
class ScanProgress:
    """Mutable progress record shared between the scan thread and the API."""
    root_id: int
    root_path: str
    phase: str = "starting"  # starting | discovering | hashing | exif | inserting | done | error
    discovered: int = 0
    indexed_new: int = 0
    indexed_updated: int = 0
    removed: int = 0
    total: int = 0
    started_at: float = 0.0
    finished_at: float | None = None
    error: str | None = None

    def to_view(self) -> dict:
        return {
            "root_id": self.root_id,
            "root_path": self.root_path,
            "phase": self.phase,
            "discovered": self.discovered,
            "indexed_new": self.indexed_new,
            "indexed_updated": self.indexed_updated,
            "removed": self.removed,
            "total": self.total,
            "elapsed": round((self.finished_at or time.monotonic()) - self.started_at, 1),
            "error": self.error,
        }


# Single shared SQLite handle isn't enough because sqlite3 connections are
# per-thread by default. Open one per caller, but configure WAL once.
_init_lock = threading.Lock()
_inited = False


def _conn() -> sqlite3.Connection:
    global _inited
    state.STATE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(LIBRARY_DB, check_same_thread=False)
    with _init_lock:
        if not _inited:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS roots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    label TEXT NOT NULL,
                    added_at INTEGER NOT NULL,
                    last_scanned INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS frames (
                    sha TEXT PRIMARY KEY,
                    root_id INTEGER NOT NULL,
                    rel_path TEXT NOT NULL,
                    stem TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    mtime INTEGER NOT NULL,
                    camera_model TEXT,
                    taken_at INTEGER,
                    indexed_at INTEGER NOT NULL,
                    FOREIGN KEY(root_id) REFERENCES roots(id) ON DELETE CASCADE
                )
                """
            )
            # Migrations: add geotag columns if a pre-geotag DB is upgraded.
            cols = [r[1] for r in conn.execute("PRAGMA table_info(frames)")]
            for col, ddl in (
                ("lat", "ALTER TABLE frames ADD COLUMN lat REAL"),
                ("lon", "ALTER TABLE frames ADD COLUMN lon REAL"),
                ("place_city", "ALTER TABLE frames ADD COLUMN place_city TEXT"),
                ("place_region", "ALTER TABLE frames ADD COLUMN place_region TEXT"),
                ("place_country", "ALTER TABLE frames ADD COLUMN place_country TEXT"),
            ):
                if col not in cols:
                    conn.execute(ddl)
            conn.execute("CREATE INDEX IF NOT EXISTS frames_root_idx ON frames(root_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS frames_taken_idx ON frames(taken_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS frames_camera_idx ON frames(camera_model)")
            conn.execute("CREATE INDEX IF NOT EXISTS frames_city_idx ON frames(place_city)")
            conn.commit()
            _inited = True
    return conn


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------

def add_root(path: Path, label: str | None = None) -> Root:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"not a folder: {path}")
    if label is None:
        label = path.name or str(path)
    with _conn() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO roots (path, label, added_at) VALUES (?, ?, ?)",
            (str(path), label, int(time.time())),
        )
        if cur.rowcount == 0:
            # Already exists; fetch it.
            row = conn.execute(
                "SELECT id, path, label, added_at, last_scanned FROM roots WHERE path=?",
                (str(path),),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id, path, label, added_at, last_scanned FROM roots WHERE id=?",
                (cur.lastrowid,),
            ).fetchone()
    return Root(*row)


def remove_root(root_id: int) -> bool:
    with _conn() as conn:
        cur = conn.execute("DELETE FROM roots WHERE id=?", (root_id,))
        return cur.rowcount > 0


def all_roots() -> list[Root]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, path, label, added_at, last_scanned FROM roots ORDER BY added_at"
        ).fetchall()
    return [Root(*r) for r in rows]


def get_root(root_id: int) -> Root | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT id, path, label, added_at, last_scanned FROM roots WHERE id=?",
            (root_id,),
        ).fetchone()
    return Root(*row) if row else None


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def _gps_to_decimal(coord, ref) -> float | None:
    """Convert EXIF (deg, min, sec) rationals plus N/S/E/W ref to decimal degrees."""
    if not coord or not ref:
        return None
    try:
        parts = []
        for v in coord:
            if hasattr(v, "numerator"):
                parts.append(v.numerator / v.denominator if v.denominator else 0.0)
            elif isinstance(v, tuple) and len(v) == 2:
                parts.append(v[0] / v[1] if v[1] else 0.0)
            else:
                parts.append(float(v))
        if len(parts) < 3:
            return None
        deg, mn, sec = parts[0], parts[1], parts[2]
        val = deg + mn / 60.0 + sec / 3600.0
        if isinstance(ref, bytes):
            ref = ref.decode("ascii", errors="replace")
        if str(ref).strip().upper() in ("S", "W"):
            val = -val
        return val
    except (TypeError, ValueError, ZeroDivisionError):
        return None


# reverse_geocoder loads ~110k cities into a kd-tree on first import (~1s).
# We lazy-import on first scan; if it's not installed we skip place names.
_geocoder = None
_geocoder_imported = False


def _maybe_reverse_geocode(lat: float, lon: float) -> tuple[str | None, str | None, str | None]:
    """Return (city, region, country) for the coordinate, or (None, None, None)."""
    global _geocoder, _geocoder_imported
    if not _geocoder_imported:
        try:
            import reverse_geocoder as rg  # noqa: PLC0415
            _geocoder = rg
        except ImportError:
            _geocoder = None
        _geocoder_imported = True
    if _geocoder is None:
        return None, None, None
    try:
        result = _geocoder.search([(lat, lon)], mode=1, verbose=False)
        if not result:
            return None, None, None
        r = result[0]
        return r.get("name"), r.get("admin1"), r.get("cc")
    except Exception as e:
        log.warning("reverse-geocode failed for (%s, %s): %s", lat, lon, e)
        return None, None, None


def _extract_camera_and_date(path: Path) -> tuple[
    str | None, int | None, float | None, float | None,
    str | None, str | None, str | None
]:
    """Pull camera model, capture timestamp, and reverse-geocoded place from EXIF.

    Returns (camera_model, taken_at, lat, lon, place_city, place_region, place_country).
    All values may be None. Cheap one-shot read; falls back gracefully on
    anything PIL can't parse (RAW, HEIC without pillow-heif, etc.).
    """
    if path.suffix.lower() not in (".jpg", ".jpeg"):
        return None, None, None, None, None, None, None
    try:
        from PIL import Image
        with Image.open(path) as im:
            exif = im.getexif() or {}
            # GPSInfo lives in its own sub-IFD (tag 0x8825).
            try:
                gps_ifd = exif.get_ifd(0x8825) or {}
            except Exception:
                gps_ifd = {}
    except Exception:
        return None, None, None, None, None, None, None

    model = exif.get(272)
    if isinstance(model, bytes):
        try:
            model = model.decode("utf-8", errors="replace").strip("\x00")
        except Exception:
            model = None
    if isinstance(model, str):
        model = model.strip() or None

    date_taken = None
    raw_date = exif.get(36867)
    if isinstance(raw_date, str):
        try:
            import datetime
            dt = datetime.datetime.strptime(raw_date, "%Y:%m:%d %H:%M:%S")
            date_taken = int(dt.timestamp())
        except (ValueError, OSError):
            date_taken = None

    lat = _gps_to_decimal(gps_ifd.get(2), gps_ifd.get(1))   # GPSLatitude / Ref
    lon = _gps_to_decimal(gps_ifd.get(4), gps_ifd.get(3))   # GPSLongitude / Ref
    city = region = country = None
    if lat is not None and lon is not None:
        city, region, country = _maybe_reverse_geocode(lat, lon)

    return model, date_taken, lat, lon, city, region, country


def scan_root(root_id: int, progress: ScanProgress | None = None) -> ScanProgress:
    """Incrementally index every photo under the root.

    Walks the filesystem, compares each (rel_path, mtime) to what's indexed.
    New / changed files get hashed and inserted. Indexed-but-missing files
    get dropped. EXIF (camera + date) is parsed once on first index.

    Pass `progress` to share state with another thread (the API); if None we
    make our own.
    """
    root = get_root(root_id)
    if root is None:
        raise ValueError(f"unknown root id {root_id}")

    if progress is None:
        progress = ScanProgress(
            root_id=root_id, root_path=root.path, started_at=time.monotonic()
        )
    else:
        progress.started_at = time.monotonic()
    progress.phase = "discovering"

    try:
        root_path = Path(root.path)
        frames = discover_frames(root_path, recursive=True)
        progress.discovered = len(frames)
        progress.total = len(frames)

        # Build (rel_path, mtime) -> frame map from disk, and pull what's indexed.
        disk_index: dict[str, tuple[Path, float, str, str]] = {}
        for f in frames:
            src = f.classify_path
            try:
                mtime = src.stat().st_mtime
            except OSError:
                continue
            rel = str(src.relative_to(root_path)).replace("\\", "/")
            disk_index[rel] = (src, mtime, f.stem, f.kind)

        with _conn() as conn:
            indexed = {
                rel_path: (sha, mtime)
                for sha, rel_path, mtime in conn.execute(
                    "SELECT sha, rel_path, mtime FROM frames WHERE root_id=?",
                    (root_id,),
                )
            }

        # Decide what needs rehashing.
        progress.phase = "hashing"
        to_hash: list[tuple[str, Path, float, str, str]] = []  # (rel, src, mtime, stem, kind)
        for rel, (src, mtime, stem, kind) in disk_index.items():
            prev = indexed.get(rel)
            if prev is None or abs(prev[1] - mtime) > 1.0:
                to_hash.append((rel, src, mtime, stem, kind))

        new_rows: list[tuple] = []
        if to_hash:
            with ThreadPoolExecutor(max_workers=8) as exe:
                shas = list(exe.map(lambda t: state.sha256_of(t[1]), to_hash))
            progress.phase = "exif"
            for (rel, src, mtime, stem, kind), sha in zip(to_hash, shas, strict=True):
                cam, dt, lat, lon, city, region, country = _extract_camera_and_date(src)
                new_rows.append((
                    sha, root_id, rel, stem, kind, int(mtime),
                    cam, dt, int(time.time()), lat, lon, city, region, country,
                ))
                if rel not in indexed:
                    progress.indexed_new += 1
                else:
                    progress.indexed_updated += 1

        # Removed = indexed but not on disk anymore.
        gone = [rel for rel in indexed if rel not in disk_index]

        progress.phase = "inserting"
        with _conn() as conn:
            if new_rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO frames "
                    "(sha, root_id, rel_path, stem, kind, mtime, "
                    " camera_model, taken_at, indexed_at, "
                    " lat, lon, place_city, place_region, place_country) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    new_rows,
                )
            if gone:
                conn.executemany(
                    "DELETE FROM frames WHERE root_id=? AND rel_path=?",
                    [(root_id, rel) for rel in gone],
                )
                progress.removed = len(gone)
            conn.execute(
                "UPDATE roots SET last_scanned=? WHERE id=?",
                (int(time.time()), root_id),
            )

        progress.phase = "done"
        progress.finished_at = time.monotonic()
        log.info(
            "scan %s: %d new, %d updated, %d removed in %.1fs",
            root.path, progress.indexed_new, progress.indexed_updated,
            progress.removed, progress.finished_at - progress.started_at,
        )
    except Exception as e:
        log.exception("scan failed for root %s", root.path)
        progress.phase = "error"
        progress.error = str(e)
        progress.finished_at = time.monotonic()
    return progress


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

@dataclass
class LibraryFrame:
    sha: str
    root_id: int
    root_label: str
    rel_path: str
    stem: str
    kind: str
    camera_model: str | None
    taken_at: int | None
    abs_path: str
    lat: float | None = None
    lon: float | None = None
    place_city: str | None = None
    place_region: str | None = None
    place_country: str | None = None

    def to_view(self) -> dict:
        place_parts = [self.place_city, self.place_region, self.place_country]
        place = ", ".join(p for p in place_parts if p) or None
        return {
            "sha": self.sha,
            "root_id": self.root_id,
            "root_label": self.root_label,
            "rel_path": self.rel_path,
            "stem": self.stem,
            "kind": self.kind,
            "camera": self.camera_model,
            "taken_at": self.taken_at,
            "abs_path": self.abs_path,
            "lat": self.lat,
            "lon": self.lon,
            "place": place,
            "place_city": self.place_city,
            "place_region": self.place_region,
            "place_country": self.place_country,
        }


def query_frames(
    root_id: int | None = None,
    subdir: str | None = None,
    camera: str | None = None,
    after: int | None = None,
    before: int | None = None,
    limit: int = 500,
    offset: int = 0,
) -> list[LibraryFrame]:
    """Return library frames matching the filters. SQL-side filtering only;
    face/tag/rating filtering happens in the API layer because those live
    outside this DB (in metadata/<sha>.json and labels.db)."""
    clauses = []
    params: list = []
    if root_id is not None:
        clauses.append("frames.root_id=?")
        params.append(root_id)
    if subdir is not None:
        # subdir filter: rel_path is "sub/stem.ext", so subdir match is prefix.
        like = subdir.rstrip("/") + "/%"
        clauses.append("frames.rel_path LIKE ?")
        params.append(like)
    if camera is not None:
        clauses.append("frames.camera_model=?")
        params.append(camera)
    if after is not None:
        clauses.append("frames.taken_at >= ?")
        params.append(after)
    if before is not None:
        clauses.append("frames.taken_at <= ?")
        params.append(before)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    sql = (
        "SELECT frames.sha, frames.root_id, roots.label, frames.rel_path, "
        "       frames.stem, frames.kind, frames.camera_model, frames.taken_at, "
        "       roots.path, frames.lat, frames.lon, frames.place_city, "
        "       frames.place_region, frames.place_country "
        "FROM frames LEFT JOIN roots ON frames.root_id = roots.id "
        f"{where} "
        "ORDER BY COALESCE(frames.taken_at, 0) DESC, frames.rel_path "
        "LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])
    with _conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    out: list[LibraryFrame] = []
    for (sha, rid, rlabel, rel_path, stem, kind, cam, dt, rpath,
         lat, lon, city, region, country) in rows:
        abs_path = str(Path(rpath) / rel_path) if rpath else rel_path
        out.append(LibraryFrame(
            sha=sha, root_id=rid, root_label=rlabel or "",
            rel_path=rel_path, stem=stem, kind=kind,
            camera_model=cam, taken_at=dt, abs_path=abs_path,
            lat=lat, lon=lon, place_city=city,
            place_region=region, place_country=country,
        ))
    return out


def count_frames(root_id: int | None = None) -> int:
    sql = "SELECT COUNT(*) FROM frames"
    params: list = []
    if root_id is not None:
        sql += " WHERE root_id=?"
        params.append(root_id)
    with _conn() as conn:
        return int(conn.execute(sql, params).fetchone()[0])


def cameras() -> list[tuple[str, int]]:
    """All distinct camera models seen, with their frame counts."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT camera_model, COUNT(*) FROM frames "
            "WHERE camera_model IS NOT NULL "
            "GROUP BY camera_model ORDER BY 2 DESC"
        ).fetchall()
    return [(m, int(c)) for m, c in rows]


def places() -> list[tuple[str, int]]:
    """All distinct reverse-geocoded cities seen, with frame counts."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT place_city, COUNT(*) FROM frames "
            "WHERE place_city IS NOT NULL "
            "GROUP BY place_city ORDER BY 2 DESC"
        ).fetchall()
    return [(p, int(c)) for p, c in rows]


def backfill_gps() -> int:
    """Walk every frame with no GPS data, re-read EXIF, persist any GPS+place
    we find. Lets a library indexed before geotag-aware scan catch up
    without re-hashing every file. Returns the count updated."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT frames.sha, roots.path, frames.rel_path "
            "FROM frames JOIN roots ON frames.root_id = roots.id "
            "WHERE frames.lat IS NULL"
        ).fetchall()
    n = 0
    for sha, root_path, rel in rows:
        src = Path(root_path) / rel
        if not src.exists():
            continue
        _, _, lat, lon, city, region, country = _extract_camera_and_date(src)
        if lat is None and lon is None:
            continue
        with _conn() as conn:
            conn.execute(
                "UPDATE frames SET lat=?, lon=?, place_city=?, "
                "place_region=?, place_country=? WHERE sha=?",
                (lat, lon, city, region, country, sha),
            )
        n += 1
    return n


def reverse_geocode_unreferenced() -> int:
    """Backfill place_city / region / country for frames that have lat/lon but
    no place data. Returns the count updated. Cheap to call after the
    reverse_geocoder package becomes available (the user installs it after
    a scan)."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT sha, lat, lon FROM frames "
            "WHERE lat IS NOT NULL AND lon IS NOT NULL AND place_city IS NULL"
        ).fetchall()
    n = 0
    for sha, lat, lon in rows:
        city, region, country = _maybe_reverse_geocode(lat, lon)
        if city is None:
            continue
        with _conn() as conn:
            conn.execute(
                "UPDATE frames SET place_city=?, place_region=?, place_country=? "
                "WHERE sha=?",
                (city, region, country, sha),
            )
        n += 1
    return n


def folder_tree(root_id: int) -> list[tuple[str, int]]:
    """Return distinct top-level subdirs under a root with their frame counts.

    For now: just one level. A real tree would build a nested dict; that can
    come later when a user actually has 4+ levels of nesting they need to
    navigate.
    """
    with _conn() as conn:
        rows = conn.execute(
            "SELECT rel_path FROM frames WHERE root_id=?", (root_id,)
        ).fetchall()
    counts: dict[str, int] = {}
    for (rel,) in rows:
        parts = rel.split("/", 1)
        sub = parts[0] if len(parts) > 1 else ""
        counts[sub] = counts.get(sub, 0) + 1
    return sorted(counts.items(), key=lambda kv: (kv[0] == "", kv[0]))


def lookup_path(sha: str) -> Path | None:
    """Resolve a sha to the absolute source path. Used by thumb/preview
    endpoints when the sha didn't come from an active scoring job."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT roots.path, frames.rel_path "
            "FROM frames JOIN roots ON frames.root_id = roots.id "
            "WHERE frames.sha=?",
            (sha,),
        ).fetchone()
    if not row:
        return None
    root_path, rel_path = row
    p = Path(root_path) / rel_path
    return p if p.exists() else None
