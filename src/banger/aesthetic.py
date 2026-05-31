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


def _pick_device() -> str:
    """Choose the torch device for CLIP work.

    Honors BANGER_AESTHETIC_DEVICE if set; otherwise prefers CUDA, then
    Apple Silicon (MPS), then CPU. When an explicit override asks for an
    unavailable accelerator we degrade to CPU rather than crash. On CPU we
    let torch use every core, since CLIP forwards are the throughput floor.
    """
    env = os.environ.get("BANGER_AESTHETIC_DEVICE")
    if env:
        device = env
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        elif device == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    return device


@lru_cache(maxsize=1)
def _load():
    from transformers import CLIPModel, CLIPProcessor

    device = _pick_device()

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


def _to_pil(item) -> Image.Image:
    """Coerce a batch item to a PIL.Image.

    Accepts a pathlib.Path / str (opened from disk), an existing PIL.Image
    (returned as-is, converted to RGB), or a BGR numpy array as produced by
    cv2/load_preview (converted to RGB). This is the union the SHARED
    CONTRACT promises for encode_images_batch.
    """
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, (str, os.PathLike)):
        with Image.open(item) as im:
            return im.convert("RGB")
    # Assume a numpy array in BGR order (the cv2/load_preview convention).
    rgb = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def encode_images_batch(images: list, batch_size: int = 32) -> np.ndarray:
    """L2-normalized CLIP image embeddings for many inputs, in input order.

    `images` is a list whose items may be PIL.Image, pathlib.Path (or str
    path), or BGR numpy arrays. Runs the CLIP processor + get_image_features
    in batches of `batch_size`, one forward per batch under no_grad, and
    returns a float32 array of shape (N, D) where each row is L2-normalized.
    Returns an empty (0, 0) array for an empty input.
    """
    if not images:
        return np.empty((0, 0), dtype=np.float32)

    model, processor, device, _, _ = _load()
    chunks: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batch = [_to_pil(it) for it in images[start : start + batch_size]]
        image_inputs = processor(images=batch, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        with torch.no_grad():
            emb = model.get_image_features(**image_inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        chunks.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def encode_image(preview: np.ndarray) -> np.ndarray:
    """Return the L2-normalized CLIP image embedding (1D float32, 512 dims)."""
    return encode_images_batch([preview])[0]


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
