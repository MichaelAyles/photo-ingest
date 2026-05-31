from pathlib import Path

import cv2
import numpy as np

from banger.preview import load_preview

CONFIG = {
    "threshold": 100.0,
}


def sharpness_from_preview(preview: np.ndarray) -> float:
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sharpness(path: Path) -> float:
    return sharpness_from_preview(load_preview(path))


def normalized_sharpness(image: np.ndarray) -> float:
    """Contrast-normalized Laplacian variance — a scene/resolution-invariant focus measure.

    AUDIT NOTE: the existing `sharpness` / `sharpness_from_preview` measure is a
    raw Laplacian variance compared against an ABSOLUTE threshold (CONFIG
    ["threshold"] == 100). That absolute cutoff is the gap the pipeline audit
    flagged: Laplacian variance scales with both image contrast and resolution,
    so a low-contrast (foggy, backlit, flat-light) but perfectly-focused frame
    scores low and is wrongly culled, while a high-contrast busy scene clears
    the bar even when soft. The threshold therefore has to be re-tuned per
    scene/camera/preview size, which doesn't generalize.

    This function divides the Laplacian variance by the image's intensity
    variance (the (std)^2 of the grayscale), which removes the first-order
    contrast dependence and yields a roughly unitless ratio. That makes a
    single gate threshold far more portable across exposures, lenses, and
    preview resolutions. Higher = sharper relative to the scene's own contrast.

    Returns 0.0 for empty/degenerate or perfectly flat images (zero contrast),
    matching the convention used elsewhere in this package.
    """
    if image is None or getattr(image, "size", 0) == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float64)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Intensity variance as the contrast normalizer; guard the flat-image case.
    intensity_var = float(gray.var())
    if intensity_var <= 1e-6:
        return 0.0
    return lap_var / intensity_var
