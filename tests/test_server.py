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


def test_unlabelled_uncertain_no_head(client):
    """With no taste head saved, endpoint still answers — just unsorted."""
    res = client.get("/api/unlabelled-uncertain")
    assert res.status_code == 200
    data = res.get_json()
    assert data["head_loaded"] is False
    assert data["total_unlabelled"] == 3
    assert data["scored_count"] == 0
    assert len(data["ordered"]) == 3


def test_unlabelled_uncertain_with_head(client, monkeypatch):
    """When the head exists, unlabelled frames come back ordered by |predicted|."""
    import re

    body = client.get("/").get_data(as_text=True)
    shas = re.findall(r'"sha":\s*"([0-9a-f]{64})"', body)
    assert len(shas) == 3

    # Cache fake embeddings so the endpoint has something to score.
    from banger import state, taste_head

    for sha in shas:
        state.cache_embedding(sha, np.zeros(8, dtype=np.float32))

    class FakeHead:
        def predict(self, X):
            # Deterministic per-sha "predicted" score driven by row index.
            # All-zero embedding maps to a constant in real Ridge; the test
            # patches `predict_score` directly so we control the values.
            return np.array([0.0] * X.shape[0])

    monkeypatch.setattr(taste_head, "load", lambda: FakeHead())
    # Assign each sha a different |predicted| score in registration order.
    scores = {shas[0]: 2.5, shas[1]: -0.1, shas[2]: 1.2}
    monkeypatch.setattr(taste_head, "predict_score", lambda model, emb: scores[shas[_lookup_sha(emb, shas)]])

    res = client.get("/api/unlabelled-uncertain")
    assert res.status_code == 200
    data = res.get_json()
    assert data["head_loaded"] is True
    assert data["scored_count"] == 3
    # Most uncertain = smallest |score| = shas[1] (|-0.1| = 0.1).
    assert data["ordered"][0]["sha"] == shas[1]
    assert abs(data["ordered"][0]["predicted"] + 0.1) < 1e-6


def _lookup_sha(_emb, shas):
    """Round-robin index for the mocked predict_score in the test above."""
    if not hasattr(_lookup_sha, "_i"):
        _lookup_sha._i = 0
    i = _lookup_sha._i
    _lookup_sha._i = (i + 1) % len(shas)
    return i
