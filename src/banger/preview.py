from pathlib import Path

import cv2
import numpy as np

PREVIEW_LONG_EDGE = 1024

RAW_SUFFIXES = {".arw", ".ARW"}
JPEG_SUFFIXES = {".jpg", ".jpeg", ".JPG", ".JPEG"}
SUPPORTED_SUFFIXES = RAW_SUFFIXES | JPEG_SUFFIXES


def load_preview(path: Path) -> np.ndarray:
    """Return a BGR uint8 array, long edge resized to PREVIEW_LONG_EDGE."""
    suffix = path.suffix
    if suffix in JPEG_SUFFIXES:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cv2 failed to read {path}")
    elif suffix in RAW_SUFFIXES:
        img = _extract_raw_preview(path)
    else:
        raise ValueError(f"unsupported suffix: {path}")
    return _resize_long_edge(img, PREVIEW_LONG_EDGE)


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
