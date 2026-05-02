"""Server endpoints: thumb / preview / label / clear / stats / index.

Patches out the CLIP path so the test never loads a real model. The label
endpoint will try to encode an embedding on first label; we redirect that
to a tiny no-op so the SQLite write is the only side effect we exercise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def app(tmp_path: Path, monkeypatch, make_jpeg, isolated_state):
    """Build a Flask app pointed at a tmp dir of synthetic JPEGs."""
    # Three frames in two subdirs so recursive walk has something to chew on.
    make_jpeg(name="A.JPG", subdir="trip1")
    make_jpeg(name="B.JPG", subdir="trip2", sharpness="high")
    make_jpeg(name="C.JPG")  # at root

    # Stub aesthetic.encode_image so /api/label POST doesn't try to load CLIP.
    import banger.aesthetic as ae
    monkeypatch.setattr(ae, "encode_image", lambda preview: np.zeros(8, dtype=np.float32))

    from banger import server

    serve = server.serve

    captured = {}

    def capture_app(input_dir, port=8000):
        # Replace `app.run(...)` with capturing the constructed Flask app.
        from flask import Flask

        original_run = Flask.run
        Flask.run = lambda self, **kwargs: captured.setdefault("app", self)
        try:
            serve(input_dir, port=port)
        finally:
            Flask.run = original_run

    capture_app(tmp_path)
    return captured["app"]


@pytest.fixture
def client(app):
    return app.test_client()


def test_index_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    # Three frames in initial render — A, B, C.
    assert "A" in body
    assert "B" in body
    assert "C" in body
    assert "trip1" in body
    assert "trip2" in body


def test_stats_endpoint_initial(client):
    res = client.get("/api/stats")
    assert res.status_code == 200
    data = res.get_json()
    assert data["total"] == 3
    assert data["labelled"] == 0


def test_label_then_clear_round_trip(client):
    # Pull the index to discover sha values rendered into the page.
    body = client.get("/").get_data(as_text=True)
    import re

    # Cards are JS-rendered; sha appears in the FRAMES JSON literal in the script.
    shas = re.findall(r'"sha":\s*"([0-9a-f]{64})"', body)
    assert len(shas) == 3

    target = shas[0]
    res = client.post("/api/label", json={"sha": target, "score": 4})
    assert res.status_code == 200
    assert res.get_json() == {"sha": target, "score": 4}

    stats = client.get("/api/stats").get_json()
    assert stats["labelled"] == 1

    res = client.delete(f"/api/label/{target}")
    assert res.status_code == 200
    stats = client.get("/api/stats").get_json()
    assert stats["labelled"] == 0


def test_label_rejects_out_of_range(client):
    body = client.get("/").get_data(as_text=True)
    import re

    # Cards are JS-rendered; sha appears in the FRAMES JSON literal in the script.
    shas = re.findall(r'"sha":\s*"([0-9a-f]{64})"', body)
    target = shas[0]
    res = client.post("/api/label", json={"sha": target, "score": 99})
    assert res.status_code == 400


def test_label_rejects_unknown_sha(client):
    res = client.post("/api/label", json={"sha": "0" * 64, "score": 0})
    assert res.status_code == 404


def test_thumb_404_on_unknown_sha(client):
    res = client.get("/api/thumb/" + "0" * 64)
    assert res.status_code == 404
