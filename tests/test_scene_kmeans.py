"""Scene KMeans clusterer smoke tests."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _state_dir(tmp_path, monkeypatch):
    """Point banger.state at a temp dir so we never touch the user's real state."""
    from banger import state, scene_kmeans

    monkeypatch.setattr(state, "STATE_DIR", tmp_path)
    monkeypatch.setattr(state, "EMBEDDINGS_DIR", tmp_path / "embeddings")
    monkeypatch.setattr(state, "METADATA_DIR", tmp_path / "metadata")
    monkeypatch.setattr(state, "THUMBS_DIR", tmp_path / "thumbs")
    monkeypatch.setattr(state, "PREVIEWS_DIR", tmp_path / "previews")
    monkeypatch.setattr(state, "LABELS_DB", tmp_path / "labels.db")
    monkeypatch.setattr(scene_kmeans, "SCENE_MODEL_PATH", tmp_path / "scene_kmeans.joblib")
    monkeypatch.setattr(scene_kmeans, "SCENE_SUMMARY_PATH", tmp_path / "scene_kmeans.json")
    state.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    scene_kmeans.load.cache_clear()
    yield
    scene_kmeans.load.cache_clear()


def _fake_emb(rng, dim=512):
    e = rng.standard_normal(dim).astype(np.float32)
    return e / np.linalg.norm(e)


def test_fit_refuses_below_k():
    from banger import state, scene_kmeans

    rng = np.random.default_rng(0)
    for i in range(3):
        np.save(state.EMBEDDINGS_DIR / f"sha{i}.npy", _fake_emb(rng))
    assert scene_kmeans.fit(k=5) is None


def test_fit_produces_summary_and_persists(monkeypatch):
    from banger import state, scene_kmeans

    # Stub out the CLIP text-embedding call so tests don't load the model.
    import torch

    fake_text = torch.tensor(
        np.eye(7, 512).astype(np.float32),
    )
    monkeypatch.setattr("banger.scenes._encoded_prompts", lambda: fake_text)

    rng = np.random.default_rng(1)
    for i in range(30):
        np.save(state.EMBEDDINGS_DIR / f"sha{i:03d}.npy", _fake_emb(rng))

    sc = scene_kmeans.fit(k=5)
    assert sc is not None
    assert sc.k == 5
    assert len(sc.clusters) == 5
    assert all(c.preset.startswith("cluster_") for c in sc.clusters)
    assert sum(c.size for c in sc.clusters) == 30
    assert scene_kmeans.SCENE_MODEL_PATH.exists()
    assert scene_kmeans.SCENE_SUMMARY_PATH.exists()


def test_load_returns_none_when_no_model():
    from banger import scene_kmeans

    assert scene_kmeans.load() is None


def test_classify_embedding_returns_valid_cluster(monkeypatch):
    from banger import state, scene_kmeans
    import torch

    monkeypatch.setattr(
        "banger.scenes._encoded_prompts",
        lambda: torch.tensor(np.eye(7, 512).astype(np.float32)),
    )

    rng = np.random.default_rng(2)
    for i in range(25):
        np.save(state.EMBEDDINGS_DIR / f"sha{i:03d}.npy", _fake_emb(rng))

    sc = scene_kmeans.fit(k=5)
    assert sc is not None

    info = sc.classify_embedding(_fake_emb(rng))
    assert 0 <= info.cluster_id < 5
    assert info.preset == f"cluster_{info.cluster_id:02d}"
    assert len(info.nearest_prompts) == 3
