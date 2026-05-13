"""Multi-dimensional image quality metrics, OpenCV only.

Facet computes nine scoring dims per frame. Banger v0 only had two of them
(Laplacian sharpness, CLIP aesthetic via the taste head) plus an opt-in
face-sharpness gate. This module covers the other seven, all from pure
OpenCV on a single 1024 px preview, so they're cheap, deterministic, and
don't pull in a new dep.

Everything here is one-shot over a single BGR preview. We compute gray
and HSV once, then reuse them across the metrics that need them. The
return is a flat dict that gets merged into metadata/<sha>.json so the
labelling UI and the XMP step can surface it.

Design notes:
- All scores are normalised to 0..10 to match facet's convention so a
  benchmark comparison can speak the same scale.
- Exposure handles silhouettes (intentional backlit) so we don't punish
  artistic high-contrast shots as "clipped". Lifted from facet's
  histogram-based scorer; the constants are theirs, lightly retuned.
- Composition uses Canny + thirds proximity. Facet has a SAMP-Net path
  too but it 404s on download and the rule-based fallback is what their
  default 4 GB profile uses anyway.
- No blink/EAR here. Haar has no landmarks, so EAR needs mediapipe or
  insightface. That ships in eyes.py behind --eye-gate.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_all(preview_bgr: np.ndarray) -> dict[str, float]:
    """Compute every cv2-only quality dim in one pass.

    Keys returned:
      color_harmony   0..10, HSV entropy
      exposure        0..10, histogram-based with shadow/highlight/silhouette
      contrast        0..10, percentile + RMS
      noise           raw sigma (Immerkaer), lower = cleaner
      dynamic_range   stops (log2 of p98/p2)
      composition     0..10, rule-of-thirds proximity of detected subject
      leading_lines   0..10, Hough lines with diagonal bias
      mean_luminance  0..1
      shadow_clipped  0/1
      highlight_clipped 0/1
      is_silhouette   0/1 (don't penalise as clipping if 1)
      is_monochrome   0/1
    """
    if preview_bgr is None or preview_bgr.size == 0:
        return {}

    gray = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2HSV)

    out: dict[str, float] = {}
    out.update(_color_harmony(hsv))
    out.update(_exposure(gray))
    out.update(_contrast(gray))
    out["noise"] = _noise(gray)
    out["dynamic_range"] = _dynamic_range(gray)
    out.update(_composition(preview_bgr, gray))
    out["leading_lines"] = _leading_lines(gray)
    out["is_monochrome"] = _is_monochrome(hsv)
    return out


def _color_harmony(hsv: np.ndarray) -> dict[str, float]:
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    total = hist.sum()
    if total <= 0:
        return {"color_harmony": 0.0}
    probs = hist / total
    nonzero = probs > 0
    entropy = float(-np.sum(probs[nonzero] * np.log2(probs[nonzero])))
    # max entropy for 180x256 bins ~ log2(46080) ≈ 15.5 bits.
    return {"color_harmony": min(10.0, entropy * 10.0 / 15.5)}


def _exposure(gray: np.ndarray) -> dict[str, float]:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total <= 0:
        return {
            "exposure": 5.0, "mean_luminance": 0.5,
            "shadow_clipped": 0, "highlight_clipped": 0, "is_silhouette": 0,
        }
    h = hist / total
    bins = np.arange(256)
    mean_val = float(np.sum(bins * h))
    mean_lum = mean_val / 255.0
    spread = float(np.sqrt(np.sum(((bins - mean_val) ** 2) * h)))

    shadow_mass = float(h[:30].sum())
    highlight_mass = float(h[225:].sum())
    shadow_clipped = 1 if shadow_mass > 0.15 else 0
    highlight_clipped = 1 if highlight_mass > 0.10 else 0

    # Silhouette: heavy shadow plus significant highlight = intentional backlit
    lower_third = float(h[:85].sum())
    upper_third = float(h[170:].sum())
    is_silhouette = 1 if (lower_third > 0.35 and upper_third > 0.25) else 0

    lum_penalty = abs(mean_lum - 0.5) * 8.0
    spread_bonus = min(4.0, spread / 20.0)
    clip_penalty = 0.0 if is_silhouette else (shadow_mass * 4.0 + highlight_mass * 5.0)
    exposure = max(0.0, min(10.0, 7.0 - lum_penalty + spread_bonus - clip_penalty))

    return {
        "exposure": round(exposure, 2),
        "mean_luminance": round(mean_lum, 4),
        "shadow_clipped": shadow_clipped,
        "highlight_clipped": highlight_clipped,
        "is_silhouette": is_silhouette,
    }


def _contrast(gray: np.ndarray) -> dict[str, float]:
    g = gray.astype(np.float64)
    p5, p95 = np.percentile(g, [5, 95])
    percentile_contrast = float((p95 - p5) / 255.0)
    rms = float(np.std(g) / 255.0)
    # Mirror facet's mix: percentile contributes up to ~4, rms up to ~6.
    score = float(min(10.0, percentile_contrast * 5.0 + rms * 20.0))
    return {"contrast": round(score, 2)}


def _noise(gray: np.ndarray) -> float:
    """Immerkaer noise estimate. Lower = cleaner. ~0-5 clean, 5-15 moderate, 15+ noisy."""
    g = gray.astype(np.float64)
    h, w = g.shape
    if h < 3 or w < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    s = float(np.sum(np.abs(cv2.filter2D(g, -1, kernel))))
    sigma = s * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2))
    return round(float(sigma), 2)


def _dynamic_range(gray: np.ndarray) -> float:
    p2 = float(np.percentile(gray, 2))
    p98 = float(np.percentile(gray, 98))
    p2 = max(p2, 1.0)
    return round(float(np.log2(max(p98, 1.0) / p2)), 2)


def _composition(preview_bgr: np.ndarray, gray: np.ndarray) -> dict[str, float]:
    """Adaptive-Canny subject region, score by rule-of-thirds + power point proximity."""
    h, w = gray.shape
    median_val = float(np.median(gray))
    lo = int(max(0, 0.5 * median_val))
    hi = int(min(255, 1.5 * median_val))
    edges = cv2.Canny(gray, lo, hi)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (h * w) * 0.0001
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not contours:
        # No detectable subject. Assume "decent" but flag uncertain.
        return {"composition": 5.0, "composition_subject_detected": 0}

    # Pick contour scoring well on area and proximity to thirds intersections.
    thirds_x = [w / 3, 2 * w / 3]
    thirds_y = [h / 3, 2 * h / 3]
    best = None
    best_score = -1.0
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        area_score = cv2.contourArea(c) / (h * w)
        dx = min(abs(cx - t) for t in thirds_x) / w
        dy = min(abs(cy - t) for t in thirds_y) / h
        thirds_bonus = max(0.0, 1.0 - (dx + dy))
        s = area_score * (1.0 + thirds_bonus)
        if s > best_score:
            best_score = s
            best = c
    if best is None:
        return {"composition": 5.0, "composition_subject_detected": 0}

    x, y, bw, bh = cv2.boundingRect(best)
    cx = (x + bw / 2) / w
    cy = (y + bh / 2) / h
    thirds = [1 / 3, 2 / 3]
    power_points = [(px, py) for px in thirds for py in thirds]
    min_pp = min(float(np.hypot(cx - px, cy - py)) for px, py in power_points)
    pp_score = max(0.0, 10.0 - min_pp * 25.0)
    line_dx = min(abs(cx - t) for t in thirds)
    line_dy = min(abs(cy - t) for t in thirds)
    line_score = max(0.0, 10.0 - (line_dx + line_dy) * 15.0)
    centre_score = max(0.0, 10.0 - (abs(cx - 0.5) + abs(cy - 0.5)) * 10.0)
    # Weight power points heavier than mere thirds proximity, fall back to centre
    # composition when that's stronger (centred portraits are fine).
    weighted = (pp_score * 2.0 + line_score * 1.0) / 3.0
    return {
        "composition": round(max(weighted, centre_score), 2),
        "composition_subject_detected": 1,
    }


def _leading_lines(gray: np.ndarray) -> float:
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    min_len = int(min(h, w) * 0.15)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=min_len, maxLineGap=20)
    if lines is None:
        return 0.0
    diagonal = float(np.hypot(h, w))
    total = 0.0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = 90.0 if (x2 - x1) == 0 else abs(float(np.degrees(np.arctan((y2 - y1) / (x2 - x1)))))
        bonus = 1.5 if 15 <= angle <= 75 else 1.0
        total += (length / diagonal) * 10.0 * bonus
    avg = total / max(1, len(lines))
    return round(min(10.0, avg * 2.0), 2)


def _is_monochrome(hsv: np.ndarray, threshold: float = 0.1) -> int:
    mean_sat = float(np.mean(hsv[:, :, 1]) / 255.0)
    return 1 if mean_sat < threshold else 0


# Names of metric keys that read as 0..10 quality scores; used by the
# combiner. Order is informative for the report and the XMP encoder.
QUALITY_DIMS = (
    "exposure",
    "contrast",
    "color_harmony",
    "composition",
    "leading_lines",
)


def combined_score(metrics: dict[str, float], aesthetic: float | None = None) -> float:
    """Average the 0..10 quality dims, blend with the taste head if present.

    The taste head is in roughly the [-5, +5] range; we shift+scale to 0..10
    and average with the cv2 dims at equal weight. When no head is available
    we just return the cv2 average so the XMP encoder always has something.
    """
    parts = []
    for k in QUALITY_DIMS:
        if k in metrics:
            parts.append(float(metrics[k]))
    cv_avg = sum(parts) / len(parts) if parts else 5.0
    if aesthetic is None:
        return cv_avg
    aest_0_10 = max(0.0, min(10.0, (aesthetic + 5.0)))
    return (cv_avg + aest_0_10) / 2.0
