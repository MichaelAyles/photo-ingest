"""Closed-eye / blink detection via mediapipe FaceMesh.

Aftershoot leans on this hard; Haar (banger's face detector) has no
landmarks so we can't compute Eye Aspect Ratio from it alone. mediapipe
FaceMesh ships a 468-point mesh with stable eye-corner / eyelid indices,
runs CPU-only in ~30-60 ms per 1024 px preview, and only loads if the
user asks for it (--eye-gate). Missing mediapipe = the gate is a no-op
rather than an error, matching how --face-gate degrades.

EAR = (vertical eyelid gap) / (horizontal eye width). Open eyes sit
around 0.25-0.30, closed eyes drop near 0.10. Threshold 0.21 is the
balanced point facet's insightface path uses; tightened slightly here
because mediapipe's eyelid landmarks track subtle squints more
aggressively than insightface's 106-point set.

Frame aggregation: facet uses ANY-blink-fails. We do min(per-face EAR)
so a single closed-eye subject is enough to gate the frame, which is the
behaviour the user asked for in plan step alongside --face-gate.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

log = logging.getLogger("banger")

EYE_AR_THRESHOLD = 0.21

# mediapipe FaceMesh landmark indices for the six points that define EAR
# on each eye (outer corner, inner corner, top1, top2, bottom1, bottom2).
LEFT_EYE_IDX = [33, 133, 159, 158, 145, 153]
RIGHT_EYE_IDX = [362, 263, 386, 385, 374, 380]

# Padding (as a fraction of the eye-region bounding box's longer edge) added
# around the eye landmarks before measuring Laplacian variance. A little
# context — eyelashes, brow, catchlights — is where the "tack-sharp eyes"
# signal actually lives, so we don't crop to the bare landmark hull.
EYE_CROP_PAD_RATIO = 0.4


@lru_cache(maxsize=1)
def _facemesh():
    """Return a process-wide cached mediapipe FaceMesh, or None if uninstalled.

    Cached with lru_cache because building a FaceMesh constructs a TFLite graph
    (~50-150 ms), which is far too expensive to pay per frame. The old code
    rebuilt and closed one inside analyse_eyes' inner loop; that graph
    construction dominated the per-frame cost. We now build once and reuse.

    static_image_mode=True means each process() call is independent (no temporal
    tracking state leaks between frames), so a single shared instance is safe
    across unrelated previews. We intentionally never call .close() — the
    instance lives for the life of the process and is reclaimed at exit.
    """
    try:
        import mediapipe as mp
    except ImportError:
        return None
    # mediapipe >= 0.10.x dropped the legacy `mp.solutions` API (it has only
    # `.tasks` now), so accessing mp.solutions.face_mesh raises AttributeError
    # rather than ImportError. Guard the whole build so a solutions-less or
    # otherwise-broken mediapipe degrades to a no-op gate instead of crashing.
    try:
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )
    except Exception as e:  # AttributeError (no .solutions), or graph-build failure
        log.warning("mediapipe FaceMesh unavailable (%s) — eye gate is a no-op", e)
        return None


def _ear(landmarks: np.ndarray, idx: list[int]) -> float:
    p_outer, p_inner, p_top1, p_top2, p_bot1, p_bot2 = (landmarks[i] for i in idx)
    v1 = float(np.linalg.norm(p_top1 - p_bot1))
    v2 = float(np.linalg.norm(p_top2 - p_bot2))
    h = float(np.linalg.norm(p_outer - p_inner))
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.3


def analyse_eyes(preview_bgr: np.ndarray) -> dict:
    """Return {face_count, ear_min, ear_mean, any_blink, per_face: [...]}.

    No-faces or no-mediapipe both return face_count=0 and ear_min=1.0 so the
    caller's `ear_min < threshold` check naturally lets the frame through.
    """
    fm = _facemesh()
    if fm is None:
        return {"face_count": 0, "ear_min": 1.0, "ear_mean": 1.0, "any_blink": 0, "per_face": []}
    import cv2

    rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
    h, w = preview_bgr.shape[:2]
    # Do NOT close fm here: it's the lru_cache'd shared instance. Closing it
    # would free the TFLite graph and force a costly rebuild on the next frame.
    result = fm.process(rgb)

    if not result.multi_face_landmarks:
        return {"face_count": 0, "ear_min": 1.0, "ear_mean": 1.0, "any_blink": 0, "per_face": []}

    ears: list[float] = []
    per_face = []
    for face in result.multi_face_landmarks:
        pts = np.array([[lm.x * w, lm.y * h] for lm in face.landmark], dtype=np.float32)
        ear_l = _ear(pts, LEFT_EYE_IDX)
        ear_r = _ear(pts, RIGHT_EYE_IDX)
        avg = (ear_l + ear_r) / 2.0
        ears.append(avg)
        per_face.append({"ear_l": round(ear_l, 4), "ear_r": round(ear_r, 4), "ear": round(avg, 4)})

    ear_min = float(min(ears))
    ear_mean = float(sum(ears) / len(ears))
    any_blink = 1 if ear_min < EYE_AR_THRESHOLD else 0
    return {
        "face_count": len(ears),
        "ear_min": round(ear_min, 4),
        "ear_mean": round(ear_mean, 4),
        "any_blink": any_blink,
        "per_face": per_face,
    }


def eye_region_sharpness(image: np.ndarray, landmarks) -> float:
    """Laplacian variance inside a padded box around the eye landmarks.

    This is the focus measure behind Aftershoot's "are the subject's eyes
    tack-sharp" check: a portrait can clear a whole-face or whole-frame blur
    gate while the eyes themselves are soft (focus landed on the cheek/ear).
    Restricting the Laplacian to the eye region surfaces exactly that miss.

    Args:
        image: a BGR (or any single-/3-channel) np image, typically the
            preview the landmarks were detected on. Pixel coordinates in
            `landmarks` are interpreted in this image's frame.
        landmarks: eye landmark coordinates as an (N, 2) array-like of
            absolute (x, y) pixel positions. Accepts either the mediapipe
            FaceMesh eye points (e.g. the six LEFT_EYE_IDX/RIGHT_EYE_IDX
            points, or both eyes' points concatenated) or insightface's
            5-point kps (left-eye, right-eye, ... — pass the eye rows).

    Returns:
        The Laplacian variance over the padded eye crop (higher = sharper).
        Returns 0.0 — never None — when landmarks are missing/empty or the
        derived crop is degenerate (zero area / out of bounds), so callers can
        treat it as "no signal / not sharp" uniformly alongside the other
        Laplacian measures in this package.
    """
    if image is None or getattr(image, "size", 0) == 0:
        return 0.0

    try:
        pts = np.asarray(landmarks, dtype=np.float32)
    except (TypeError, ValueError):
        return 0.0
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return 0.0
    # Keep only x, y in case 3D (mediapipe) landmarks were passed in.
    pts = pts[:, :2]
    if not np.all(np.isfinite(pts)):
        return 0.0

    H, W = image.shape[:2]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    # Pad by a fraction of the box's longer edge; floor the pad so even a
    # single-point or perfectly-horizontal landmark set yields a real crop.
    box_w = x_max - x_min
    box_h = y_max - y_min
    pad = EYE_CROP_PAD_RATIO * max(box_w, box_h)
    pad = max(pad, 4.0)

    x0 = int(np.floor(x_min - pad))
    y0 = int(np.floor(y_min - pad))
    x1 = int(np.ceil(x_max + pad))
    y1 = int(np.ceil(y_max + pad))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0

    import cv2

    crop = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mediapipe_available() -> bool:
    """True only if mediapipe AND the legacy solutions FaceMesh API we use exist.

    mediapipe >= 0.10.x ships without `mp.solutions`, so a bare import check
    would report it available and then crash in _facemesh(). Probe the actual
    attribute path we depend on so the eye gate no-ops cleanly on those builds.
    """
    try:
        import mediapipe as mp
    except ImportError:
        return False
    return hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
