"""Aesthetic scoring: pure-logic tests with a stubbed CLIP load.

`_load()` is monkey-patched to return a synthetic (model, processor, device,
text_emb, n_pos) tuple so tests don't need to download or run the real CLIP
model. The math under test is `score_from_embedding`, which is just a
matmul against the prompt embeddings + a positive/negative average.
"""

import numpy as np
import torch

from banger import aesthetic
from banger.aesthetic import (
    NEGATIVE_PROMPTS,
    POSITIVE_PROMPTS,
    score_from_embedding,
)


def _stub_load(text_emb: torch.Tensor):
    """Returns a closure that aesthetic._load can be monkey-patched to."""
    return lambda: (None, None, "cpu", text_emb, len(POSITIVE_PROMPTS))


def _build_text_emb(boost_positives: bool = False, boost_negatives: bool = False) -> torch.Tensor:
    """Six unit-norm rows (3 positives + 3 negatives), optionally boosted along axis 0."""
    n = len(POSITIVE_PROMPTS) + len(NEGATIVE_PROMPTS)
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((n, 4)).astype(np.float32)
    if boost_positives:
        for i in range(len(POSITIVE_PROMPTS)):
            arr[i] = np.array([5, 0, 0, 0], dtype=np.float32)
    if boost_negatives:
        for i in range(len(POSITIVE_PROMPTS), n):
            arr[i] = np.array([5, 0, 0, 0], dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return torch.from_numpy(arr).float()


def test_score_breakdown_includes_every_prompt(monkeypatch):
    text_emb = _build_text_emb()
    monkeypatch.setattr(aesthetic, "_load", _stub_load(text_emb))
    img_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    score, breakdown = score_from_embedding(img_emb)
    assert set(breakdown.keys()) == set(POSITIVE_PROMPTS + NEGATIVE_PROMPTS)
    assert isinstance(score, float)


def test_image_aligned_with_positives_scores_high(monkeypatch):
    text_emb = _build_text_emb(boost_positives=True)
    monkeypatch.setattr(aesthetic, "_load", _stub_load(text_emb))
    img_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    high_score, _ = score_from_embedding(img_emb)
    assert high_score > 0


def test_image_aligned_with_negatives_scores_low(monkeypatch):
    text_emb = _build_text_emb(boost_negatives=True)
    monkeypatch.setattr(aesthetic, "_load", _stub_load(text_emb))
    img_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    low_score, _ = score_from_embedding(img_emb)
    assert low_score < 0


def test_three_positives_three_negatives():
    """The current prompt set is balanced; an unbalanced set would skew the score."""
    assert len(POSITIVE_PROMPTS) == 3
    assert len(NEGATIVE_PROMPTS) == 3


def test_no_blur_in_negatives_documents_decision():
    """Blur and focus are deliberately omitted from negatives — sharpness gate
    handles real blur, and CLIP can't tell intentional bokeh from accidental
    blur. If a future change re-adds 'blurry' it should fail this so the
    decision gets re-discussed."""
    blocked = ("blurry", "out of focus", "out-of-focus")
    for prompt in NEGATIVE_PROMPTS:
        for word in blocked:
            assert word not in prompt.lower(), (
                f"Negative prompt re-introduces '{word}': {prompt!r}"
            )


def test_score_scaling_constant_documented(monkeypatch):
    """Score = (mean_pos - mean_neg) * 50 — calibration roughly to [-2, 3]
    in observed CLIP cosine territory. Tested by aligning the image embedding
    with positives and verifying the magnitude of the resulting score."""
    text_emb = _build_text_emb(boost_positives=True)
    monkeypatch.setattr(aesthetic, "_load", _stub_load(text_emb))
    img_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    score, _ = score_from_embedding(img_emb)
    # When all positives are aligned with image (cos≈1) and negatives are
    # random unit vectors (cos roughly 0..0.5), mean(pos)-mean(neg) is
    # large and (* 50) puts the score well above 5.
    assert score > 5
