"""Face detection + face-region sharpness for the v1 face-aware gate.

Uses the Haar cascade that ships with opencv-python — no extra dep, no
download. Haar is fast (~5-15 ms per 1024 px preview), good enough for
a "frame contains a face?" yes/no, and conservative on false-positives.

The pipeline rule (plan step 18): for any surviving frame that contains
a face, also require that the SHARPEST face in the frame passes a
face-specific Laplacian-variance threshold. Catches the photographer's
classic missed-focus portrait — sharp background, soft face — that the
global sharpness gate alone lets through.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

# Haar variance is computed on a face crop (typically 100-300 px square),
# which has much less detail than a full-frame Laplacian. Calibrate
# separately: 50 is a starting guess and can be tightened from real data.
FACE_SHARPNESS_THRESHOLD = 50.0
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE_RATIO = 0.05  # face must be at least 5% of the long edge


@lru_cache(maxsize=1)
def _detector() -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        raise RuntimeError(f"failed to load haar cascade at {path}")
    return clf


def detect_faces(preview_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return a list of (x, y, w, h) face bounding boxes. Empty if none found."""
    if preview_bgr.size == 0:
        return []
    gray = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    min_dim = max(20, int(round(min(h, w) * HAAR_MIN_SIZE_RATIO)))
    boxes = _detector().detectMultiScale(
        gray,
        scaleFactor=HAAR_SCALE_FACTOR,
        minNeighbors=HAAR_MIN_NEIGHBORS,
        minSize=(min_dim, min_dim),
    )
    return [tuple(int(v) for v in box) for box in boxes]


def face_sharpness(preview_bgr: np.ndarray, face_box: tuple[int, int, int, int]) -> float:
    """Laplacian variance on a face crop. Higher = sharper face."""
    x, y, w, h = face_box
    H, W = preview_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = preview_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def best_face_sharpness(preview_bgr: np.ndarray) -> tuple[int, float]:
    """Return (face_count, sharpest face's Laplacian variance).

    face_count == 0 means no face was detected; the second value is 0 in that
    case. The pipeline's face-aware gate should only kick in when face_count
    > 0; otherwise the global sharpness gate stands.
    """
    boxes = detect_faces(preview_bgr)
    if not boxes:
        return 0, 0.0
    sharpest = max(face_sharpness(preview_bgr, box) for box in boxes)
    return len(boxes), sharpest
