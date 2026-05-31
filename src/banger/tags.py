"""CLIP zero-shot tagging against a curated vocabulary.

The detail overlay used to show a BLIP-generated sentence. Sentences look
nice in isolation but are noisy ("a man and a woman taking a self camera
self") and don't slice or filter well. Tag chips do both jobs: they're
short, they read fast at a glance, and downstream features (group-by-tag
in the gallery, filter "show me all the dogs", etc) drop out for free.

How it works:
- A ~180-entry vocabulary across subjects, scenes, activities, lighting,
  aesthetics, and weather. Each phrase gets prefixed with "a photo of"
  so CLIP's text encoder sees the kind of prompt it was trained on.
- All tag phrases are encoded once via CLIP at module load (the same
  CLIP model used for aesthetic scoring; no second model, no extra dep).
- For a frame, we already have the image embedding cached. Take cosine
  sim against the tag matrix, threshold at TAG_MIN_SIM, return top-N.

Threshold rationale: CLIP cosine sims for real photos against natural
phrases sit roughly in [0.18, 0.30]. A "this tag clearly applies"
boundary is ~0.22 for ViT-B/32. We expose TAG_MIN_SIM so it's tunable.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

# Categories are just for organisation; the model sees them as one flat list.
SUBJECTS = [
    "person", "people", "group of people", "child", "baby", "family",
    "man", "woman", "couple", "selfie",
    "dog", "cat", "horse", "cow", "sheep", "bird", "fish",
    "lion", "tiger", "elephant", "monkey", "giraffe", "zebra", "wolf", "bear", "deer",
    "flowers", "tree", "leaves", "rock", "sand", "ice", "snow",
    "car", "bicycle", "motorcycle", "boat", "train", "airplane", "bus",
    "food", "drink", "coffee", "wine", "cake", "pizza",
]

SCENES = [
    "mountain", "hill", "valley", "forest", "woods", "meadow",
    "beach", "ocean", "lake", "river", "waterfall", "desert", "cave",
    "city", "street", "alley", "rooftop", "skyscraper",
    "village", "town", "harbour", "bridge",
    "park", "garden", "trail", "path", "road",
    "zoo", "wildlife park", "farm",
    "indoor", "outdoor", "restaurant", "cafe", "bar", "kitchen",
    "bedroom", "living room", "office", "stadium", "concert", "museum",
]

ACTIVITIES = [
    "hiking", "walking", "running", "cycling",
    "swimming", "surfing", "skiing", "snowboarding", "climbing",
    "eating", "drinking", "cooking", "shopping",
    "working", "reading", "playing",
    "dancing", "singing", "performing",
    "kissing", "hugging", "laughing", "smiling",
    "portrait shot", "candid shot", "action shot",
]

LIGHTING = [
    "sunset", "sunrise", "golden hour", "blue hour", "night", "daytime",
    "bright sunlight", "overcast", "shaded", "indoor lighting",
    "backlit subject", "silhouette", "high contrast", "low contrast",
]

AESTHETIC = [
    "close-up", "macro", "wide-angle landscape", "telephoto compression",
    "shallow depth of field", "deep focus",
    "minimalist composition", "busy composition",
    "black and white", "monochrome", "vibrant colours", "muted colours",
    "high angle", "low angle", "aerial view", "drone shot",
]

WEATHER = [
    "snowy", "rainy", "foggy", "misty", "cloudy", "sunny", "stormy", "windy",
]

TAG_VOCAB: list[str] = SUBJECTS + SCENES + ACTIVITIES + LIGHTING + AESTHETIC + WEATHER

TAG_PROMPT_TEMPLATE = "a photo of {}"

# A tag has to clear this cosine to make it into the result. ~0.22 is empirically
# the "this clearly applies" line for CLIP ViT-B/32 against natural phrasing.
TAG_MIN_SIM = 0.22

# How many tags per image. Even at the threshold, frames with rich content
# can produce 20+; we cap so the chips fit on screen.
TAG_TOP_N = 10


@lru_cache(maxsize=1)
def _encoded_tags() -> tuple[torch.Tensor, list[str]]:
    """CLIP-encode every TAG_VOCAB entry once. Returns (NxD tensor, labels)."""
    from banger.aesthetic import _load as _load_clip_internals

    # `device` here is whatever banger.aesthetic._pick_device chose
    # (cuda → mps → cpu, honoring BANGER_AESTHETIC_DEVICE). We reuse the
    # one CLIP load rather than picking a device independently so tagging
    # always runs on the same accelerator as aesthetic scoring.
    model, processor, device, _, _ = _load_clip_internals()
    prompts = [TAG_PROMPT_TEMPLATE.format(t) for t in TAG_VOCAB]
    with torch.inference_mode():
        inputs = processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb, list(TAG_VOCAB)


def tag_from_embedding(
    image_emb: np.ndarray,
    top_n: int = TAG_TOP_N,
    min_sim: float = TAG_MIN_SIM,
) -> list[tuple[str, float]]:
    """Return up to `top_n` (tag, cosine_sim) pairs above `min_sim`, sorted desc."""
    text_emb, labels = _encoded_tags()
    img = torch.from_numpy(image_emb).to(text_emb.device).to(text_emb.dtype)
    sims = (img @ text_emb.T).cpu().tolist()
    ranked = sorted(zip(labels, sims, strict=True), key=lambda kv: -kv[1])
    out: list[tuple[str, float]] = []
    for label, sim in ranked:
        if sim < min_sim:
            break
        out.append((label, float(sim)))
        if len(out) >= top_n:
            break
    # Guarantee at least the top 3 even if they're below threshold, so the
    # panel never shows an empty tag block on weird images.
    if not out:
        out = [(label, float(sim)) for label, sim in ranked[:3]]
    return out
