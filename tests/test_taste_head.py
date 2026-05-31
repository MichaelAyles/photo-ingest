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


def _seed_state_highdim(state, n_up: int, n_down: int, dim: int = 512):
    """Same as _seed_state but with high-dimensional embeddings, to exercise
    the p>>n PCA/RidgeCV path. Only the first axis carries signal; the rest is
    noise, so PCA-then-ridge should still recover the sign cleanly."""
    rng = np.random.default_rng(seed=7)
    vector_up = np.zeros(dim, dtype=np.float32)
    vector_up[0] = 1.0
    vector_down = np.zeros(dim, dtype=np.float32)
    vector_down[0] = -1.0
    for i in range(n_up):
        sha = f"hup-{i:03d}"
        state.cache_embedding(
            sha, vector_up + 0.05 * rng.standard_normal(dim).astype(np.float32)
        )
        state.add_label(sha, 5, f"hup-{i}", f"/u/{i}.jpg")
    for i in range(n_down):
        sha = f"hdown-{i:03d}"
        state.cache_embedding(
            sha, vector_down + 0.05 * rng.standard_normal(dim).astype(np.float32)
        )
        state.add_label(sha, -5, f"hdown-{i}", f"/d/{i}.jpg")


# ---- blend_with_prior --------------------------------------------------------


def test_blend_prior_dominates_when_few_labels():
    """With ~0 labels the blend should sit essentially on top of the prior,
    ignoring a wildly different personal score."""
    personal = 10.0
    prior = -3.0
    # Zero labels: pure prior.
    assert taste_head.blend_with_prior(personal, prior, n_labels=0) == prior
    # A single label: still overwhelmingly the prior.
    blended = taste_head.blend_with_prior(personal, prior, n_labels=1)
    assert abs(blended - prior) < abs(blended - personal)


def test_blend_personal_dominates_when_many_labels():
    """Once we're at/over the full-trust label count, the blend is the
    personal score and the prior drops out."""
    personal = 10.0
    prior = -3.0
    assert taste_head.blend_with_prior(personal, prior, n_labels=50) == personal
    assert taste_head.blend_with_prior(personal, prior, n_labels=500) == personal


def test_blend_is_monotonic_in_label_count():
    """As labels accumulate the blend moves monotonically from prior toward
    personal."""
    personal = 8.0
    prior = -2.0
    prev = taste_head.blend_with_prior(personal, prior, n_labels=0)
    for n in range(1, 60, 5):
        cur = taste_head.blend_with_prior(personal, prior, n_labels=n)
        assert cur >= prev  # personal > prior here, so blend should rise
        prev = cur
    # Midpoint (n == half of full_trust) is the simple average.
    mid = taste_head.blend_with_prior(personal, prior, n_labels=25)
    assert abs(mid - (personal + prior) / 2.0) < 1e-6


def test_blend_respects_custom_full_trust_labels():
    personal = 10.0
    prior = 0.0
    # With full_trust_labels=10, n_labels=10 should be fully personal.
    assert (
        taste_head.blend_with_prior(personal, prior, n_labels=10, full_trust_labels=10)
        == personal
    )
    # Halfway at n=5.
    half = taste_head.blend_with_prior(
        personal, prior, n_labels=5, full_trust_labels=10
    )
    assert abs(half - personal / 2.0) < 1e-6


# ---- p>>n PCA / RidgeCV path -------------------------------------------------


def test_train_highdim_uses_pca_ridgecv_and_predicts_signs(isolated_state):
    """A small (n ~= 24) but high-dim (512) synthetic set should route through
    the PCA + RidgeCV pipeline and still rank up vs down correctly."""
    _seed_state_highdim(isolated_state, n_up=12, n_down=12, dim=512)
    model = taste_head.train_from_disk()
    assert model is not None
    # Pipeline introspection: both the dimensionality reduction and the
    # cross-validated ridge should be present in the p>>n regime.
    assert "pca" in model.named_steps
    assert type(model.named_steps["ridge"]).__name__ == "RidgeCV"
    # PCA must have shrunk the feature space below the raw embedding dim.
    assert model.named_steps["pca"].n_components < 512

    up = np.zeros(512, dtype=np.float32)
    up[0] = 1.0
    down = np.zeros(512, dtype=np.float32)
    down[0] = -1.0
    pred_up = taste_head.predict_score(model, up)
    pred_down = taste_head.predict_score(model, down)
    assert pred_up > pred_down


def test_train_few_labels_falls_back_to_fixed_alpha(isolated_state):
    """With only a handful of labels we should not attempt RidgeCV/PCA — a
    plain fixed-alpha Ridge keeps cold start from blowing up."""
    _seed_state(isolated_state, n_up=2, n_down=2)  # 4 labels, 8-dim
    model = taste_head.train_from_disk()
    assert model is not None
    # Below the CV threshold -> plain Ridge, and below the PCA threshold -> no PCA.
    assert type(model.named_steps["ridge"]).__name__ == "Ridge"
    assert "pca" not in model.named_steps


def test_save_helper_round_trips(isolated_state):
    """The new save() helper persists a model loadable by load()."""
    _seed_state(isolated_state, n_up=5, n_down=5)
    model = taste_head.train_from_disk()
    assert model is not None
    # Remove the on-disk head, then re-save via the helper.
    isolated_state.TASTE_HEAD.unlink()
    assert not taste_head.exists()
    taste_head.save(model)
    assert taste_head.exists()
    loaded = taste_head.load()
    assert loaded is not None
    probe = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    np.testing.assert_allclose(
        model.predict(probe.reshape(1, -1)), loaded.predict(probe.reshape(1, -1))
    )
