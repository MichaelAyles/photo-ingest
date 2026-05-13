from pathlib import Path

import cv2
import numpy as np

PREVIEW_LONG_EDGE = 1024

# rawpy / libraw handles all of these. The bytes-level reader is the same,
# so adding extensions is enough to bring multi-camera support online without
# per-vendor branching. Add new ones here as cameras get tested.
_RAW_BASES = (
    "arw",   # Sony
    "cr2", "cr3",  # Canon
    "nef", "nrw",  # Nikon
    "raf",         # Fuji
    "pef",         # Pentax
    "orf",         # Olympus
    "rw2",         # Panasonic
    "dng",         # Apple ProRAW, Adobe, Pixel, generic
    "srw",         # Samsung
    "x3f",         # Sigma
    "3fr",         # Hasselblad
    "iiq",         # Phase One
    "rwl",         # Leica
    "gpr",         # GoPro
)
RAW_SUFFIXES = {f".{b}" for b in _RAW_BASES} | {f".{b.upper()}" for b in _RAW_BASES}
JPEG_SUFFIXES = {".jpg", ".jpeg", ".JPG", ".JPEG", ".heic", ".HEIC", ".heif", ".HEIF"}
SUPPORTED_SUFFIXES = RAW_SUFFIXES | JPEG_SUFFIXES


def load_preview(path: Path) -> np.ndarray:
    """Return a BGR uint8 array, long edge resized to PREVIEW_LONG_EDGE."""
    suffix = path.suffix
    if suffix in JPEG_SUFFIXES:
        if suffix.lower() in (".heic", ".heif"):
            img = _read_heic(path)
        else:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cv2 failed to read {path}")
    elif suffix in RAW_SUFFIXES:
        img = _extract_raw_preview(path)
    else:
        raise ValueError(f"unsupported suffix: {path}")
    return _resize_long_edge(img, PREVIEW_LONG_EDGE)


def _read_heic(path: Path) -> np.ndarray:
    """HEIC / HEIF support via pillow-heif. The decoder is optional; if the
    user hasn't installed it we surface a clear error rather than crashing
    on cv2.imread (which doesn't support HEIC)."""
    try:
        from pillow_heif import register_heif_opener
        from PIL import Image
    except ImportError as e:
        raise ValueError(
            f"HEIC file {path} requires pillow-heif (pip install pillow-heif)"
        ) from e
    register_heif_opener()
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    # PIL gives RGB; cv2 expects BGR.
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _extract_raw_preview(path: Path) -> np.ndarray:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        try:
            thumb = raw.extract_thumb()
        except rawpy.LibRawNoThumbnailError as e:
            raise ValueError(f"no embedded thumbnail in {path}") from e
    if thumb.format != rawpy.ThumbFormat.JPEG:
        raise ValueError(f"unexpected thumb format {thumb.format} in {path}")
    arr = np.frombuffer(thumb.data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"failed to decode embedded JPEG in {path}")
    return img


def _resize_long_edge(img: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1.0:
        return img
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
