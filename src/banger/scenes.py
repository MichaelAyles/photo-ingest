"""CLIP zero-shot classification of an image into one of seven treatment prompts.

The pipeline picks one of these prompts per surviving frame; each prompt
maps 1:1 to a darktable .xmp preset that's been hand-tuned offline. The
prompts and the mapping are documented in CLAUDE.md § Stage 7 / Stage 8;
keep them in sync with any preset changes in `presets/`.

Confidence falls back to a safe default ("crisp daylight outdoor portrait")
when the top similarity is below CONFIDENCE_THRESHOLD — better to slap a
neutral preset on a weird scene than to apply something inappropriate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from banger.aesthetic import _load as _load_clip_internals

CONFIDENCE_THRESHOLD = 0.20  # CLAUDE.md guessed 0.25; empirically 97% fell back at that level on real data

# Per-prompt thresholds (step 19): start uniform at the global default, override
# specific prompts when empirical tuning suggests they need to be tighter or
# looser. Looking up via .get(prompt, CONFIDENCE_THRESHOLD) so a missing entry
# falls back to the global value rather than crashing.
PER_PROMPT_THRESHOLDS: dict[str, float] = {
    # No overrides yet — populate as evidence accumulates that, e.g.,
    # "a moody high-contrast black and white photograph" needs +0.02 to avoid
    # firing on dim daytime shots, etc.
}

SCENE_PROMPTS = [
    "a moody high-contrast black and white photograph",
    "a vibrant warm landscape at golden hour",
    "a cinematic low-light scene with deep shadows",
    "a crisp daylight outdoor portrait",
    "a punchy high-saturation street scene",
    "a soft pastel atmospheric photograph",
    "a documentary-style flat tonal photograph",
]

PROMPT_TO_PRESET = {
    "a moody high-contrast black and white photograph": "bw_moody",
    "a vibrant warm landscape at golden hour": "golden_landscape",
    "a cinematic low-light scene with deep shadows": "cinematic_lowlight",
    "a crisp daylight outdoor portrait": "crisp_daylight",
    "a punchy high-saturation street scene": "street_punch",
    "a soft pastel atmospheric photograph": "pastel_soft",
    "a documentary-style flat tonal photograph": "documentary_flat",
}

DEFAULT_PROMPT = "a crisp daylight outdoor portrait"
DEFAULT_PRESET = PROMPT_TO_PRESET[DEFAULT_PROMPT]


@dataclass
class SceneMatch:
    prompt: str  # the prompt actually used (default if fell_back)
    preset: str  # short preset name, e.g. 'bw_moody'
    score: float  # cosine similarity to `prompt`
    top_score: float  # max similarity across SCENE_PROMPTS
    breakdown: dict[str, float]
    fell_back: bool  # True if top_score was below CONFIDENCE_THRESHOLD


@lru_cache(maxsize=1)
def _encoded_prompts() -> torch.Tensor:
    """Cache the CLIP text embeddings for SCENE_PROMPTS — one CLIP forward total."""
    # `device` is chosen by banger.aesthetic._pick_device (cuda → mps → cpu,
    # honoring BANGER_AESTHETIC_DEVICE). Reusing the shared CLIP load keeps
    # scene classification on the same accelerator as aesthetic scoring.
    model, processor, device, _, _ = _load_clip_internals()
    with torch.inference_mode():
        text_inputs = processor(text=SCENE_PROMPTS, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_emb = model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb


def classify_with_emb(
    image_emb: np.ndarray,
    text_emb: torch.Tensor,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    per_prompt_thresholds: dict[str, float] | None = None,
) -> SceneMatch:
    """Pure logic over a precomputed text-embedding tensor; no model load.

    The threshold check is per-prompt — `per_prompt_thresholds` overrides
    `confidence_threshold` for any prompt with an entry. A prompt fails only
    when its similarity is below ITS specific threshold, so different
    treatments can have different stringency.
    """
    overrides = per_prompt_thresholds or PER_PROMPT_THRESHOLDS
    img_t = torch.from_numpy(image_emb).to(text_emb.device).to(text_emb.dtype)
    sims = (img_t @ text_emb.T).cpu().tolist()
    breakdown = dict(zip(SCENE_PROMPTS, sims, strict=True))
    top_prompt = max(breakdown, key=breakdown.get)
    top_score = breakdown[top_prompt]
    threshold_for_top = overrides.get(top_prompt, confidence_threshold)
    if top_score < threshold_for_top:
        return SceneMatch(
            prompt=DEFAULT_PROMPT,
            preset=DEFAULT_PRESET,
            score=breakdown[DEFAULT_PROMPT],
            top_score=top_score,
            breakdown=breakdown,
            fell_back=True,
        )
    return SceneMatch(
        prompt=top_prompt,
        preset=PROMPT_TO_PRESET[top_prompt],
        score=top_score,
        top_score=top_score,
        breakdown=breakdown,
        fell_back=False,
    )


def classify(image_emb: np.ndarray) -> SceneMatch:
    return classify_with_emb(image_emb, _encoded_prompts())
