from pathlib import Path

import cv2

from banger.preview import load_preview

CONFIG = {
    "threshold": 100.0,
}


def sharpness(path: Path) -> float:
    preview = load_preview(path)
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
