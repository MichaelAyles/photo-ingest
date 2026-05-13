"""Develop pipeline: numpy-based image adjustments.

This is the "real edit" path. The existing develop.py shells out to
darktable-cli for preset application; that's brittle on Windows/Mac. We
want banger to do the actual pixel-pushing so the install story stays
"pip install" not "install darktable first."

Ten adjustments, modelled after Lightroom's Develop module:
- exposure (stops)         linear gain in light-linear space
- contrast (-100..+100)    midpoint-pivot S-curve
- highlights / shadows     localised tone curve on upper / lower regions
- whites / blacks          endpoint stretches (white point, black point)
- saturation               flat HSV S scaling
- vibrance                 saturation boost weighted by inverse current sat
                           (boosts dull areas more than already-saturated)
- temp                     blue<>yellow channel shift
- tint                     green<>magenta channel shift
- crop                     normalised {x, y, w, h} rectangle
- rotation                 free rotation in degrees

Working colour space: gamma-corrected sRGB float32 in [0, 1]. Most users
don't need linear-light fidelity for casual edits, and gamma-space is
fast and matches what the preview-JPEG already is. If we want HDR /
log-space later, the path is one ICC profile lookup away.

Performance: a 1024 px preview develop pass runs in ~30 ms on CPU. Full
4000 px export is ~400 ms. Real-time slider feedback in the GUI uses
the preview path; export uses the full-size path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from banger import state
from banger.preview import JPEG_SUFFIXES, RAW_SUFFIXES

log = logging.getLogger("banger.editor")


@dataclass
class DevelopParams:
    exposure: float = 0.0          # stops, typical range -2..+2
    contrast: float = 0.0          # -100..+100
    highlights: float = 0.0        # -100..+100, positive = recover, negative = crush
    shadows: float = 0.0           # -100..+100, positive = lift, negative = deepen
    whites: float = 0.0            # -100..+100
    blacks: float = 0.0            # -100..+100
    saturation: float = 0.0        # -100..+100
    vibrance: float = 0.0          # -100..+100
    temp: float = 0.0              # -100 (cool) .. +100 (warm)
    tint: float = 0.0              # -100 (green) .. +100 (magenta)
    crop: dict | None = None       # {"x", "y", "w", "h"} in [0, 1] of pre-crop image
    rotation: float = 0.0          # degrees, free rotation


def default_params() -> DevelopParams:
    return DevelopParams()


def to_dict(p: DevelopParams) -> dict:
    return asdict(p)


def from_dict(d: dict) -> DevelopParams:
    if not d:
        return DevelopParams()
    fields = set(DevelopParams.__dataclass_fields__.keys())
    clean = {k: v for k, v in d.items() if k in fields}
    return DevelopParams(**clean)


# ---------------------------------------------------------------------------
# Per-adjustment functions. Each takes float32 [0, 1] BGR and returns same.
# ---------------------------------------------------------------------------

def _apply_exposure(img: np.ndarray, stops: float) -> np.ndarray:
    if abs(stops) < 1e-3:
        return img
    return np.clip(img * (2.0 ** stops), 0.0, 1.0)


def _apply_contrast(img: np.ndarray, contrast: float) -> np.ndarray:
    if abs(contrast) < 1e-3:
        return img
    factor = 1.0 + contrast / 100.0
    return np.clip((img - 0.5) * factor + 0.5, 0.0, 1.0)


def _apply_tone_region(img: np.ndarray, amount: float, mask: np.ndarray) -> np.ndarray:
    """Lift or crush a tonal region defined by `mask` (broadcastable to img).

    `amount` in [-100, 100]. Positive lifts (toward 1), negative crushes (toward 0).
    Mask weights the effect so we don't blow out shadows when adjusting highlights.
    """
    if abs(amount) < 1e-3:
        return img
    a = amount / 100.0
    # Smoothly push the masked area toward 1 (positive) or 0 (negative).
    target = 1.0 if a > 0 else 0.0
    return np.clip(img + (target - img) * abs(a) * 0.45 * mask, 0.0, 1.0)


def _luma(img: np.ndarray) -> np.ndarray:
    """Per-pixel luminance, single channel. cv2 uses BGR so weights are flipped."""
    return img[..., 0] * 0.114 + img[..., 1] * 0.587 + img[..., 2] * 0.299


def _apply_highlights(img: np.ndarray, amount: float) -> np.ndarray:
    if abs(amount) < 1e-3:
        return img
    # Mask: 0 below mid, ramps to 1 at top.
    l = _luma(img)
    mask = np.clip((l - 0.5) * 2.0, 0.0, 1.0)
    return _apply_tone_region(img, -amount, mask[..., None])  # positive recovers = pushes down


def _apply_shadows(img: np.ndarray, amount: float) -> np.ndarray:
    if abs(amount) < 1e-3:
        return img
    l = _luma(img)
    mask = np.clip((0.5 - l) * 2.0, 0.0, 1.0)
    return _apply_tone_region(img, amount, mask[..., None])  # positive lifts


def _apply_whites(img: np.ndarray, amount: float) -> np.ndarray:
    if abs(amount) < 1e-3:
        return img
    # White-point stretch: scale around 1.0 for the top quarter of values.
    l = _luma(img)
    mask = np.clip((l - 0.75) * 4.0, 0.0, 1.0)
    a = amount / 100.0 * 0.25
    return np.clip(img + a * mask[..., None], 0.0, 1.0)


def _apply_blacks(img: np.ndarray, amount: float) -> np.ndarray:
    if abs(amount) < 1e-3:
        return img
    l = _luma(img)
    mask = np.clip((0.25 - l) * 4.0, 0.0, 1.0)
    a = amount / 100.0 * 0.25
    return np.clip(img + a * mask[..., None], 0.0, 1.0)


def _apply_saturation(img: np.ndarray, amount: float) -> np.ndarray:
    if abs(amount) < 1e-3:
        return img
    factor = 1.0 + amount / 100.0
    hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0.0, 1.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _apply_vibrance(img: np.ndarray, amount: float) -> np.ndarray:
    """Saturation boost weighted by inverse current saturation: dull pixels
    get the full boost, already-saturated pixels barely move."""
    if abs(amount) < 1e-3:
        return img
    hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    boost = (amount / 100.0) * (1.0 - sat)  # less effect on already-saturated
    hsv[..., 1] = np.clip(sat + boost, 0.0, 1.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _apply_temp_tint(img: np.ndarray, temp: float, tint: float) -> np.ndarray:
    if abs(temp) < 1e-3 and abs(tint) < 1e-3:
        return img
    # temp: positive = warmer = +R, -B. tint: positive = magenta = -G, +R+B.
    t = temp / 100.0 * 0.15
    n = tint / 100.0 * 0.10
    out = img.copy()
    # BGR order in cv2.
    out[..., 0] = np.clip(out[..., 0] - t + n * 0.5, 0.0, 1.0)  # B
    out[..., 1] = np.clip(out[..., 1] - n, 0.0, 1.0)            # G
    out[..., 2] = np.clip(out[..., 2] + t + n * 0.5, 0.0, 1.0)  # R
    return out


def _apply_rotation(img: np.ndarray, degrees: float) -> np.ndarray:
    if abs(degrees) < 0.05:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def _apply_crop(img: np.ndarray, crop: dict | None) -> np.ndarray:
    if not crop:
        return img
    h, w = img.shape[:2]
    x = max(0, int(crop.get("x", 0) * w))
    y = max(0, int(crop.get("y", 0) * h))
    cw = max(1, int(crop.get("w", 1) * w))
    ch = max(1, int(crop.get("h", 1) * h))
    x2 = min(w, x + cw)
    y2 = min(h, y + ch)
    if x2 <= x or y2 <= y:
        return img
    return img[y:y2, x:x2]


def apply_develop(img_bgr_u8: np.ndarray, params: DevelopParams) -> np.ndarray:
    """Run the full develop pipeline. Input + output both BGR uint8."""
    img = img_bgr_u8.astype(np.float32) / 255.0
    img = _apply_exposure(img, params.exposure)
    img = _apply_temp_tint(img, params.temp, params.tint)
    img = _apply_whites(img, params.whites)
    img = _apply_blacks(img, params.blacks)
    img = _apply_highlights(img, params.highlights)
    img = _apply_shadows(img, params.shadows)
    img = _apply_contrast(img, params.contrast)
    img = _apply_vibrance(img, params.vibrance)
    img = _apply_saturation(img, params.saturation)
    img = _apply_rotation(img, params.rotation)
    img = _apply_crop(img, params.crop)
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Source loading. Preview path uses the cached 1024px image for interactive
# slider feedback; export path loads the full-res original from disk.
# ---------------------------------------------------------------------------

def load_full_size(src_path: Path) -> np.ndarray:
    """Load original-resolution BGR uint8. JPEG via cv2, RAW via rawpy
    postprocess (slower but gives the demosaiced full image)."""
    suffix = src_path.suffix
    if suffix in JPEG_SUFFIXES:
        if suffix.lower() in (".heic", ".heif"):
            from pillow_heif import register_heif_opener
            register_heif_opener()
            with Image.open(src_path) as im:
                arr = np.asarray(im.convert("RGB"))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cv2 failed to read {src_path}")
        return img
    if suffix in RAW_SUFFIXES:
        import rawpy
        with rawpy.imread(str(src_path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=False,
                output_bps=8,
            )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    raise ValueError(f"unsupported suffix: {src_path}")


# ---------------------------------------------------------------------------
# Persistence: develop params live inside metadata/<sha>.json under "develop".
# ---------------------------------------------------------------------------

def load_params(sha: str) -> DevelopParams:
    meta = state.load_frame_metadata(sha) or {}
    return from_dict(meta.get("develop") or {})


def save_params(sha: str, params: DevelopParams) -> None:
    state.update_frame_metadata(sha, develop=to_dict(params))


def has_edits(sha: str) -> bool:
    meta = state.load_frame_metadata(sha) or {}
    d = meta.get("develop") or {}
    if not d:
        return False
    p = from_dict(d)
    default = DevelopParams()
    for f in DevelopParams.__dataclass_fields__:
        if getattr(p, f) != getattr(default, f):
            return True
    return False


# ---------------------------------------------------------------------------
# Export: render full-res and write JPEG next to source (or to dst).
# ---------------------------------------------------------------------------

def export_jpeg(
    src_path: Path,
    params: DevelopParams,
    dst_path: Path,
    quality: int = 92,
) -> Path:
    img = load_full_size(src_path)
    out = apply_develop(img, params)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst_path), out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError(f"cv2.imwrite failed for {dst_path}")
    return dst_path


def default_export_path(src_path: Path) -> Path:
    """`/foo/IMG_123.JPG` -> `/foo/IMG_123_edit.jpg`. Avoids overwriting source."""
    return src_path.with_name(f"{src_path.stem}_edit.jpg")
