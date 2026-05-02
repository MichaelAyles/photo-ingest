"""Prompt-based aesthetic scoring with CLIP ViT-B/32.

The original plan called for aesthetic-predictor-v2.5 (SigLIP-so400m). Its
3.3 GB safetensors file segfaults torch.UntypedStorage.__getitem__ on this
Windows install (the mmap-slice bug bites somewhere above ~2 GB; smaller
safetensors files load fine). The plan explicitly allows a CLIP-prompt
fallback for v0; this swaps SigLIP for CLIP ViT-B/32 (~600 MB) and scores
each frame as the difference between average similarity to a small set of
"good photo" prompts and a small set of "bad photo" prompts.
"""

import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from banger.preview import load_preview

CLIP_MODEL = "openai/clip-vit-base-patch32"

POSITIVE_PROMPTS = [
    "a beautiful, well-composed photograph",
    "a striking professional photograph with strong subject and clean composition",
    "a sharp, vibrant, technically excellent photograph",
]
NEGATIVE_PROMPTS = [
    "a blurry, poorly framed photograph",
    "a boring, cluttered, low-quality snapshot",
    "an out-of-focus, badly exposed, amateur photo",
]


@lru_cache(maxsize=1)
def _load():
    from transformers import CLIPModel, CLIPProcessor

    device = os.environ.get("BANGER_AESTHETIC_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    with torch.inference_mode():
        text_inputs = processor(
            text=POSITIVE_PROMPTS + NEGATIVE_PROMPTS, return_tensors="pt", padding=True
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_emb = model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    return model, processor, device, text_emb, len(POSITIVE_PROMPTS)


def score_from_preview(preview: np.ndarray) -> tuple[float, dict[str, float]]:
    """Return (score, per-prompt cosine similarity dict)."""
    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    model, processor, device, text_emb, n_pos = _load()

    image_inputs = processor(images=pil, return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

    with torch.inference_mode():
        img_emb = model.get_image_features(**image_inputs)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sims = (img_emb @ text_emb.T).squeeze(0).cpu().tolist()

    all_prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
    breakdown = dict(zip(all_prompts, sims))
    pos = sum(sims[:n_pos]) / n_pos
    neg = sum(sims[n_pos:]) / (len(sims) - n_pos)
    # Cosine sims for typical image-text pairs sit in [0.15, 0.35]. The pos-neg
    # gap typically lands in [-0.05, 0.10]. Scale by 50 so the printable score
    # has the rough magnitude of an aesthetic rating; only ranking matters.
    return (pos - neg) * 50.0, breakdown


def score(path: Path) -> tuple[float, dict[str, float]]:
    return score_from_preview(load_preview(path))
