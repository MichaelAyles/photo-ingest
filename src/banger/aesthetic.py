"""Prompt-based aesthetic scoring with CLIP ViT-B/32.

The original plan called for aesthetic-predictor-v2.5 (SigLIP-so400m). Its
3.3 GB safetensors file segfaults torch.UntypedStorage.__getitem__ on this
Windows install (the mmap-slice bug bites somewhere above ~2 GB; smaller
safetensors files load fine). The plan explicitly allows a CLIP-prompt
fallback for v0; this swaps SigLIP for CLIP ViT-B/32 (~600 MB) and scores
each frame as the difference between average similarity to a small set of
"good photo" prompts and a small set of "bad photo" prompts.

Generic prompts can't capture personal taste. Once `banger label` has
collected enough frames and `banger train` has produced a taste head,
cmd_run uses that instead — see banger.taste_head.
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
    "a striking photograph with the subject in clear focus and a beautifully soft background",
    "a well-composed photograph with intentional depth of field and a strong subject",
    "a beautiful, engaging photograph with clean storytelling and strong subject placement",
]
# Note: prompts about blur/focus are deliberately omitted. The sharpness gate
# (Laplacian variance) already filters real motion blur and missed focus, and
# CLIP cannot distinguish intentional bokeh from accidental blur — so a "blurry"
# negative double-counts the sharpness gate AND punishes shallow-DOF portraits.
NEGATIVE_PROMPTS = [
    "a forgettable, awkwardly composed snapshot",
    "a cluttered photograph with distracting elements and no clear subject",
    "a badly framed photo where the subject is cut off or oddly placed",
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


def encode_image(preview: np.ndarray) -> np.ndarray:
    """Return the L2-normalized CLIP image embedding (1D float32, 512 dims)."""
    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    model, processor, device, _, _ = _load()
    image_inputs = processor(images=pil, return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    with torch.inference_mode():
        img_emb = model.get_image_features(**image_inputs)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    return img_emb.squeeze(0).cpu().numpy().astype(np.float32)


def score_from_embedding(emb: np.ndarray) -> tuple[float, dict[str, float]]:
    _, _, _, text_emb, n_pos = _load()
    img_t = torch.from_numpy(emb).to(text_emb.device).to(text_emb.dtype)
    sims = (img_t @ text_emb.T).cpu().tolist()
    all_prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
    breakdown = dict(zip(all_prompts, sims, strict=True))
    pos = sum(sims[:n_pos]) / n_pos
    neg = sum(sims[n_pos:]) / (len(sims) - n_pos)
    # Cosine sims sit in [0.15, 0.35]; pos-neg gap typically [-0.05, 0.10].
    # Scale by 50 so the score reads roughly like an aesthetic rating.
    return (pos - neg) * 50.0, breakdown


def score_from_preview(preview: np.ndarray) -> tuple[float, dict[str, float]]:
    return score_from_embedding(encode_image(preview))


def score(path: Path) -> tuple[float, dict[str, float]]:
    return score_from_preview(load_preview(path))
