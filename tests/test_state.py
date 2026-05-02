"""State module: sha hashing, embedding cache, label DB, schema migration."""

import sqlite3

import numpy as np
import pytest


def test_sha256_is_content_addressed(make_jpeg, isolated_state):
    p1 = make_jpeg(name="a.JPG", sharpness="medium", seed=42)
    p2 = make_jpeg(name="b.JPG", sharpness="medium", seed=42)  # explicit same seed
    p3 = make_jpeg(name="c.JPG", sharpness="high", seed=42)
    assert isolated_state.sha256_of(p1) == isolated_state.sha256_of(p2)
    assert isolated_state.sha256_of(p1) != isolated_state.sha256_of(p3)


def test_embedding_round_trip(isolated_state):
    sha = "abc123"
    emb = np.arange(512, dtype=np.float32) / 512
    assert isolated_state.load_embedding(sha) is None
    isolated_state.cache_embedding(sha, emb)
    loaded = isolated_state.load_embedding(sha)
    assert loaded is not None
    np.testing.assert_array_equal(loaded, emb)


def test_label_round_trip(isolated_state):
    isolated_state.add_label("sha-1", 3, "DSC00001", "/some/path.JPG")
    assert isolated_state.get_label("sha-1") == 3
    assert isolated_state.get_label("missing") is None
    assert isolated_state.labels_dict() == {"sha-1": 3}


def test_label_replaces_existing_score(isolated_state):
    isolated_state.add_label("sha-1", 3, "DSC00001", "/p.JPG")
    isolated_state.add_label("sha-1", -2, "DSC00001", "/p.JPG")
    assert isolated_state.get_label("sha-1") == -2
    assert len(isolated_state.all_labels()) == 1


def test_label_rejects_out_of_range(isolated_state):
    with pytest.raises(ValueError):
        isolated_state.add_label("sha", 10, "stem", "/p")
    with pytest.raises(ValueError):
        isolated_state.add_label("sha", -10, "stem", "/p")


def test_schema_migration_v1_up_down_to_v2_score(tmp_path, monkeypatch):
    # Hand-build a v1 labels.db with the old `label TEXT` column, then poke the
    # state module to upgrade it on next connection.
    import banger.state as st

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(st, "STATE_DIR", state_dir)
    monkeypatch.setattr(st, "EMBEDDINGS_DIR", state_dir / "embeddings")
    monkeypatch.setattr(st, "THUMBS_DIR", state_dir / "thumbs")
    monkeypatch.setattr(st, "PREVIEWS_DIR", state_dir / "previews")
    monkeypatch.setattr(st, "LABELS_DB", state_dir / "labels.db")
    monkeypatch.setattr(st, "TASTE_HEAD", state_dir / "taste_head.joblib")

    with sqlite3.connect(st.LABELS_DB) as conn:
        conn.execute(
            """CREATE TABLE labels (
                sha256 TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                stem TEXT NOT NULL,
                src_path TEXT NOT NULL,
                ts INTEGER NOT NULL
            )"""
        )
        conn.execute(
            "INSERT INTO labels VALUES (?, ?, ?, ?, ?)",
            ("sha-up", "up", "favourite", "/a.JPG", 1000),
        )
        conn.execute(
            "INSERT INTO labels VALUES (?, ?, ?, ?, ?)",
            ("sha-down", "down", "reject", "/b.JPG", 1001),
        )

    rows = st.all_labels()  # triggers migration
    by_sha = {sha: score for sha, score, *_ in rows}
    assert by_sha == {"sha-up": 5, "sha-down": -5}


def test_thumbnail_and_preview_paths_use_state_dir(isolated_state):
    sha = "deadbeef"
    isolated_state.cache_thumbnail(sha, b"jpegbytes")
    isolated_state.cache_preview_jpeg(sha, b"biggerbytes")
    assert isolated_state.thumbnail_path(sha).read_bytes() == b"jpegbytes"
    assert isolated_state.preview_jpeg_path(sha).read_bytes() == b"biggerbytes"


def test_csv_round_trip(isolated_state, tmp_path):
    isolated_state.add_label("sha-a", 5, "DSC1", "/path/a.JPG")
    isolated_state.add_label("sha-b", -3, "DSC2", "/path/b.JPG")
    isolated_state.add_label("sha-c", 0, "DSC3", "/path/c.JPG")

    csv_path = tmp_path / "labels.csv"
    n_exp = isolated_state.export_labels_csv(csv_path)
    assert n_exp == 3
    assert csv_path.exists()

    # Wipe DB and re-import to confirm round-trip.
    import sqlite3

    with sqlite3.connect(isolated_state.LABELS_DB) as conn:
        conn.execute("DELETE FROM labels")
    assert isolated_state.all_labels() == []

    n_imp, n_skip = isolated_state.import_labels_csv(csv_path)
    assert n_imp == 3
    assert n_skip == 0
    restored = isolated_state.labels_dict()
    assert restored == {"sha-a": 5, "sha-b": -3, "sha-c": 0}


def test_frame_metadata_round_trip(isolated_state):
    sha = "deadbeef"
    assert isolated_state.load_frame_metadata(sha) is None
    isolated_state.cache_frame_metadata(sha, sharpness=234.5, phash_hex="abc123", timestamp=1700000000.0)
    loaded = isolated_state.load_frame_metadata(sha)
    assert loaded is not None
    assert loaded["sharpness"] == 234.5
    assert loaded["phash_hex"] == "abc123"
    assert loaded["timestamp"] == 1700000000.0
    assert "computed_at" in loaded


def test_frame_metadata_corrupt_returns_none(isolated_state, tmp_path):
    sha = "corrupt"
    # Write a malformed JSON file directly.
    isolated_state.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    (isolated_state.METADATA_DIR / f"{sha}.json").write_text("not json", encoding="utf-8")
    assert isolated_state.load_frame_metadata(sha) is None


def test_csv_import_skips_malformed_rows(isolated_state, tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "sha256,score,stem,src_path,ts\n"
        "good-sha,3,DSC1,/p,1000\n"
        "no-score-sha,not-a-number,DSC2,/p,1001\n"
        "out-of-range,99,DSC3,/p,1002\n"
        "missing-col,2,DSC4\n",
        encoding="utf-8",
    )
    imported, skipped = isolated_state.import_labels_csv(csv_path)
    assert imported == 1
    assert skipped == 3
    assert isolated_state.labels_dict() == {"good-sha": 3}
