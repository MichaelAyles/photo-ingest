"""MMR top-N selection: pure relevance vs balanced vs pure diversity."""

import numpy as np

from banger.select import normalise_scores, select_diverse_top_n


def _emb(*coords: float) -> np.ndarray:
    """Make an L2-normalised embedding from explicit coords."""
    a = np.array(coords, dtype=np.float32)
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def test_normalise_scores_basic():
    assert normalise_scores([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


def test_normalise_scores_all_equal():
    assert normalise_scores([3.0, 3.0, 3.0]) == [0.5, 0.5, 0.5]


def test_normalise_scores_empty():
    assert normalise_scores([]) == []


def test_lambda_one_is_pure_top_k():
    items = [
        ("a", 1.0, _emb(1, 0)),
        ("b", 5.0, _emb(1, 0)),  # same direction as 'a'
        ("c", 3.0, _emb(0, 1)),  # different direction
    ]
    selected = select_diverse_top_n(items, n=2, diversity_lambda=1.0)
    keys = [s[0] for s in selected]
    # Pure top-K ignores similarity: top 2 by score are 'b' (5) and 'c' (3).
    assert keys == ["b", "c"]


def test_lambda_below_one_prefers_diverse():
    items = [
        ("similar_high", 5.0, _emb(1, 0)),
        ("similar_med", 4.0, _emb(0.99, 0.14)),  # near-identical to first
        ("different_low", 3.0, _emb(0, 1)),  # orthogonal
    ]
    # Plain top-2 would pick the two similar ones.
    plain = select_diverse_top_n(items, n=2, diversity_lambda=1.0)
    assert [s[0] for s in plain] == ["similar_high", "similar_med"]
    # MMR should swap in the orthogonal candidate even at small diversity weight.
    diverse = select_diverse_top_n(items, n=2, diversity_lambda=0.5)
    assert [s[0] for s in diverse] == ["similar_high", "different_low"]


def test_lambda_zero_picks_most_different_after_first():
    items = [
        ("a", 5.0, _emb(1, 0)),
        ("b", 5.0, _emb(0.95, 0.31)),  # close to a
        ("c", 5.0, _emb(0, 1)),  # orthogonal
    ]
    selected = select_diverse_top_n(items, n=2, diversity_lambda=0.0)
    assert [s[0] for s in selected] == ["a", "c"]


def test_n_larger_than_pool_returns_all():
    items = [
        ("x", 1.0, _emb(1, 0)),
        ("y", 2.0, _emb(0, 1)),
    ]
    selected = select_diverse_top_n(items, n=5, diversity_lambda=0.5)
    assert len(selected) == 2


def test_empty_pool():
    assert select_diverse_top_n([], n=10) == []


def test_n_zero_returns_empty():
    items = [("x", 1.0, _emb(1, 0))]
    assert select_diverse_top_n(items, n=0) == []


def test_first_pick_is_always_top_score():
    items = [
        ("low", 1.0, _emb(1, 0)),
        ("high", 9.0, _emb(0.95, 0.31)),
        ("mid", 5.0, _emb(0, 1)),
    ]
    for lam in (0.0, 0.3, 0.5, 0.7, 1.0):
        out = select_diverse_top_n(items, n=1, diversity_lambda=lam)
        assert out[0][0] == "high", f"first pick should be top-score at lambda={lam}"


def test_preserves_original_score_value():
    """Normalisation is internal — output payloads carry the unchanged score."""
    items = [
        ("a", 17.5, _emb(1, 0)),
        ("b", -3.2, _emb(0, 1)),
    ]
    out = select_diverse_top_n(items, n=2, diversity_lambda=0.5)
    scores = sorted(s[1] for s in out)
    assert scores == [-3.2, 17.5]
