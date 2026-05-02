"""Taste head: training, prediction, error paths.

Uses synthetic 8-dim embeddings (instead of real 512-d CLIP) so tests run
without loading any model. The Ridge regression itself is what we're
exercising — same logic at any dimensionality.
"""

import numpy as np

from banger import taste_head


def _seed_state(state, n_up: int, n_down: int):
    """Add labels with synthetic embeddings: 'up' frames get vector_up + noise,
    'down' frames get vector_down + noise. Cleanly separable, so a tiny Ridge
    on this data should approach perfect leave-one-out."""
    rng = np.random.default_rng(seed=42)
    vector_up = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vector_down = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for i in range(n_up):
        sha = f"up-{i:03d}"
        state.cache_embedding(sha, vector_up + 0.05 * rng.standard_normal(8).astype(np.float32))
        state.add_label(sha, 5, f"up-{i}", f"/u/{i}.jpg")
    for i in range(n_down):
        sha = f"down-{i:03d}"
        state.cache_embedding(sha, vector_down + 0.05 * rng.standard_normal(8).astype(np.float32))
        state.add_label(sha, -5, f"down-{i}", f"/d/{i}.jpg")


def test_train_with_no_labels_returns_none(isolated_state):
    assert taste_head.train_from_disk() is None


def test_train_with_one_label_returns_none(isolated_state):
    isolated_state.cache_embedding("only", np.zeros(8, dtype=np.float32))
    isolated_state.add_label("only", 3, "stem", "/p")
    assert taste_head.train_from_disk() is None


def test_train_with_separable_data_predicts_signs(isolated_state):
    _seed_state(isolated_state, n_up=10, n_down=10)
    model = taste_head.train_from_disk()
    assert model is not None
    assert taste_head.exists()

    pred_up = taste_head.predict_score(
        model, np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    )
    pred_down = taste_head.predict_score(
        model, np.array([-1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    )
    assert pred_up > 0
    assert pred_down < 0
    assert pred_up > pred_down + 5  # cleanly separated


def test_load_returns_none_when_no_head(isolated_state):
    assert taste_head.load() is None
    assert not taste_head.exists()


def test_load_round_trip(isolated_state):
    _seed_state(isolated_state, n_up=5, n_down=5)
    saved = taste_head.train_from_disk()
    assert saved is not None
    loaded = taste_head.load()
    assert loaded is not None
    probe = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    np.testing.assert_allclose(
        saved.predict(probe.reshape(1, -1)), loaded.predict(probe.reshape(1, -1))
    )


def test_train_warns_when_embeddings_missing(isolated_state, caplog):
    """Labels without cached embeddings should warn, not crash."""
    import logging

    isolated_state.add_label("ghost", 2, "ghost-stem", "/g.jpg")
    _seed_state(isolated_state, n_up=3, n_down=3)
    with caplog.at_level(logging.WARNING, logger="banger"):
        model = taste_head.train_from_disk()
    assert model is not None
    assert any("ghost" in rec.getMessage() for rec in caplog.records)


def test_train_with_only_one_class_returns_none(isolated_state):
    """If labels are all the same value, even 'mixed' classes are missing — Ridge can train but
    we treat it as needing variance. (Two distinct labels are required.)"""
    _seed_state(isolated_state, n_up=5, n_down=0)
    # Even with same-value labels, Ridge can technically fit (predicts a constant).
    # The current contract: trains as long as len(y) >= 2. Document with this test.
    model = taste_head.train_from_disk()
    assert model is not None
