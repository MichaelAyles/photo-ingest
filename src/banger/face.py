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


# insightface buffalo_l returns 5-point keypoints (kps) per face, ordered
# [left_eye, right_eye, nose, left_mouth, right_mouth]. Rows 0 and 1 are the
# two eye centres — exactly what eyes.eye_region_sharpness wants.
_INSIGHTFACE_EYE_KPS_IDX = [0, 1]


def _box_area(box: tuple[int, int, int, int]) -> int:
    _, _, w, h = box
    return max(0, w) * max(0, h)


def best_face_eye_sharpness(image: np.ndarray) -> float | None:
    """Sharpness measured specifically in the EYE region of the most prominent face.

    This is the "are the subject's eyes tack-sharp" check Aftershoot is known
    for: a portrait can clear a whole-frame or whole-face blur gate while the
    eyes themselves are soft (focus landed on the cheek, ear, or background).
    Restricting the focus measure to the eye region surfaces that miss.

    Detection cascade, best signal first:
      1. insightface (if installed): pick the highest-confidence face and use
         its 5-point kps eye centres, measuring Laplacian variance in a padded
         box around the eyes via eyes.eye_region_sharpness. Best because the
         landmarks are precise and robust to pose.
      2. mediapipe FaceMesh (if installed): pick the largest face mesh and feed
         its LEFT/RIGHT eye landmark points to eyes.eye_region_sharpness.
      3. Haar fallback (always available): no landmarks, so measure the whole
         largest-face bbox Laplacian via the existing face_sharpness. This is
         the coarsest tier but degrades gracefully with zero extra deps.

    Args:
        image: a BGR np image (a preview frame).

    Returns:
        The eye-region (or, in the Haar tier, whole-face) Laplacian variance as
        a float when a face is found, else None. None means "no face detected,
        so the eye-sharpness gate has nothing to say" — distinct from 0.0,
        which means a face was found but its eye region carried no detail.
    """
    if image is None or image.size == 0:
        return None

    # Tier 1: insightface kps (most precise eye localisation).
    try:
        from banger import face_id as _face_id
    except ImportError:
        _face_id = None
    if _face_id is not None and _face_id.insightface_available():
        app = _face_id._app()
        if app is not None:
            try:
                faces = app.get(image)
            except Exception:  # noqa: BLE001 — detector hiccup → fall through to next tier
                faces = []
            best = None
            best_score = -1.0
            for f in faces:
                kps = getattr(f, "kps", None)
                if kps is None:
                    continue
                score = float(getattr(f, "det_score", 0.0))
                if score > best_score:
                    best_score = score
                    best = kps
            if best is not None:
                from banger import eyes as _eyes

                eye_pts = np.asarray(best, dtype=np.float32)[_INSIGHTFACE_EYE_KPS_IDX]
                return _eyes.eye_region_sharpness(image, eye_pts)

    # Tier 2: mediapipe FaceMesh eye landmarks (largest face).
    try:
        from banger import eyes as _eyes
    except ImportError:
        _eyes = None
    if _eyes is not None and _eyes.mediapipe_available():
        fm = _eyes._facemesh()
        if fm is not None:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            result = fm.process(rgb)
            if result.multi_face_landmarks:
                # Pick the face whose eye-landmark bbox is largest (closest /
                # most prominent subject), matching the Haar "biggest face" rule.
                best_pts = None
                best_span = -1.0
                eye_idx = _eyes.LEFT_EYE_IDX + _eyes.RIGHT_EYE_IDX
                for fl in result.multi_face_landmarks:
                    pts = np.array(
                        [[fl.landmark[i].x * w, fl.landmark[i].y * h] for i in eye_idx],
                        dtype=np.float32,
                    )
                    span = float(
                        (pts[:, 0].max() - pts[:, 0].min())
                        * (pts[:, 1].max() - pts[:, 1].min())
                    )
                    if span > best_span:
                        best_span = span
                        best_pts = pts
                if best_pts is not None:
                    return _eyes.eye_region_sharpness(image, best_pts)

    # Tier 3: Haar fallback — no landmarks, so use the largest face's bbox.
    boxes = detect_faces(image)
    if not boxes:
        return None
    biggest = max(boxes, key=_box_area)
    return face_sharpness(image, biggest)
