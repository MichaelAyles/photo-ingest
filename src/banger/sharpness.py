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
