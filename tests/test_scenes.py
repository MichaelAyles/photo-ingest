"""Scene classification: pure-logic tests with synthetic embeddings (no CLIP load)."""

import numpy as np
import torch

from banger.scenes import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_PRESET,
    DEFAULT_PROMPT,
    PROMPT_TO_PRESET,
    SCENE_PROMPTS,
    classify_with_emb,
)


def _build_text_emb(special_prompt: str | None = None, special_strength: float = 1.0) -> torch.Tensor:
    """Make a (n_prompts, 4) text-embedding tensor.

    Each row is a basis-like vector. Optionally boost one prompt's row so
    its cosine similarity to a probe image emb is highest.
    """
    n = len(SCENE_PROMPTS)
    rng = np.random.default_rng(seed=0)
    arr = rng.standard_normal((n, 4)).astype(np.float32)
    if special_prompt is not None:
        idx = SCENE_PROMPTS.index(special_prompt)
        arr[idx] = arr[idx] * special_strength + np.array([10, 0, 0, 0], dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / norms
    return torch.from_numpy(arr).float()


def test_picks_highest_similarity_prompt():
    target = "a vibrant warm landscape at golden hour"
    text_emb = _build_text_emb(special_prompt=target)
    image_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # aligned to row 0 axis
    out = classify_with_emb(image_emb, text_emb, confidence_threshold=0.0)
    assert out.prompt == target
    assert out.preset == PROMPT_TO_PRESET[target]
    assert not out.fell_back
    assert out.score == out.top_score


def test_falls_back_to_default_below_threshold():
    # Random-ish unit text embeddings; a probe nearly orthogonal to all of them.
    text_emb = _build_text_emb()
    image_emb = np.array([0.001, 0.001, 0.001, 0.001], dtype=np.float32)
    image_emb /= np.linalg.norm(image_emb)
    out = classify_with_emb(image_emb, text_emb, confidence_threshold=0.99)
    assert out.fell_back
    assert out.prompt == DEFAULT_PROMPT
    assert out.preset == DEFAULT_PRESET


def test_breakdown_includes_every_prompt():
    text_emb = _build_text_emb()
    image_emb = np.zeros(4, dtype=np.float32)
    image_emb[0] = 1.0
    out = classify_with_emb(image_emb, text_emb, confidence_threshold=0.0)
    assert set(out.breakdown.keys()) == set(SCENE_PROMPTS)


def test_every_prompt_has_a_preset_mapping():
    for p in SCENE_PROMPTS:
        assert p in PROMPT_TO_PRESET
        # Preset name is filesystem-safe (lowercase, underscores).
        assert all(c.isalnum() or c == "_" for c in PROMPT_TO_PRESET[p])


def test_default_threshold_in_calibrated_range():
    # CLAUDE.md guessed 0.25; empirically tuned downward against the user's
    # 1000+ frame sample. Anything outside this band should look suspicious.
    assert 0.15 <= CONFIDENCE_THRESHOLD <= 0.30


def test_per_prompt_threshold_can_override_global():
    """A prompt-specific threshold should fire when the global threshold wouldn't."""
    target = "a vibrant warm landscape at golden hour"
    text_emb = _build_text_emb(special_prompt=target)
    # Image embedding partially aligned with the boosted axis; cos sim ≈ 0.7.
    image_emb = np.array([0.7, 0.7, 0.0, 0.0], dtype=np.float32)
    image_emb /= np.linalg.norm(image_emb)

    # Global threshold = 0.0 → the prompt fires.
    out_loose = classify_with_emb(image_emb, text_emb, confidence_threshold=0.0)
    assert out_loose.prompt == target
    assert not out_loose.fell_back

    # Per-prompt threshold above the actual sim → that prompt is rejected,
    # and the call falls back to the default.
    overrides = {target: 0.95}
    out_strict = classify_with_emb(
        image_emb, text_emb, confidence_threshold=0.0, per_prompt_thresholds=overrides
    )
    assert out_strict.fell_back
    assert out_strict.prompt == DEFAULT_PROMPT
