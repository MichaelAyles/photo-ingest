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

import numpy as np

log = logging.getLogger("banger")

EYE_AR_THRESHOLD = 0.21

# mediapipe FaceMesh landmark indices for the six points that define EAR
# on each eye (outer corner, inner corner, top1, top2, bottom1, bottom2).
LEFT_EYE_IDX = [33, 133, 159, 158, 145, 153]
RIGHT_EYE_IDX = [362, 263, 386, 385, 374, 380]


def _facemesh():
    """Return a configured mediapipe FaceMesh, or None if mediapipe isn't installed."""
    try:
        import mediapipe as mp
    except ImportError:
        return None
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    )


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
    result = fm.process(rgb)
    fm.close()

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


def mediapipe_available() -> bool:
    try:
        import mediapipe  # noqa: F401
        return True
    except ImportError:
        return False
