"""BLIP image captioning for the detail overlay.

Generates a short free-form description ("two people hiking on a
mountain trail at sunset") that answers the "what is in this photo"
question alongside the CLIP-aesthetic and metric breakdown. Different
job from CLIP: CLIP gives us an embedding for retrieval and a similarity
score against prompts, but doesn't generate natural language. BLIP's
encoder-decoder architecture is built for captioning, so the output is
fluent rather than a top-K of prompt cosines.

We use the base model (Salesforce/blip-image-captioning-base, ~250 MB)
rather than blip-2 because base runs comfortably on the user's 4 GB GPU
alongside CLIP, generates in ~1-2s per image on CUDA, and the quality
gap with blip-2-flan-t5-xl is small for "rough description" use.

Lazy by design: the model loads on first use, not on GUI boot, because
the typical user opens the GUI to score photos and only sometimes
clicks into the detail overlay. The first overlay click after a fresh
start pays ~10-15s of model load, subsequent ones are fast.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import cv2
import numpy as np
import torch
from PIL import Image

log = logging.getLogger("banger.caption")

BLIP_MODEL = "Salesforce/blip-image-captioning-base"


@lru_cache(maxsize=1)
def _load():
    """Load BLIP base on first call. Picks GPU when available, falls back to CPU."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = os.environ.get("BANGER_CAPTION_DEVICE")
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("loading BLIP base on %s (first call, ~10-15s)", device)
    proc = BlipProcessor.from_pretrained(BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device).eval()
    return proc, model, device


def caption(preview_bgr: np.ndarray, max_length: int = 30, num_beams: int = 4) -> str:
    """Return a single short caption for the BGR preview, or '' on failure."""
    if preview_bgr is None or preview_bgr.size == 0:
        return ""
    try:
        proc, model, device = _load()
    except Exception as e:
        log.warning("BLIP load failed: %s", e)
        return ""

    rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    try:
        inputs = proc(images=pil, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )
        text = proc.decode(out[0], skip_special_tokens=True).strip()
        return text
    except Exception as e:
        log.warning("BLIP generate failed: %s", e)
        return ""


def caption_available() -> bool:
    """Cheap "do we have the model files on disk?" check.

    transformers caches under ~/.cache/huggingface/. We don't try to actually
    load the model here, just verify the snapshot exists. Used by the GUI to
    show whether the first caption call will incur a download.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    cached = try_to_load_from_cache(BLIP_MODEL, "config.json")
    return cached is not None and cached is not False
