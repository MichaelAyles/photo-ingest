"""Burst / duplicate detection via perceptual hash + EXIF timestamp.

For a sequence of frames, cluster any pair that satisfies BOTH:
  - perceptual-hash Hamming distance <= HAMMING_DIST_DEFAULT
  - EXIF (or mtime fallback) timestamps within TIME_WINDOW_DEFAULT seconds

Within each cluster, the caller decides which frame is "best" — usually
by aesthetic score. The pipeline calls this after aesthetic scoring so
the score is already cached per frame.

The clustering is greedy and time-ordered: a frame joins the most recent
cluster whose representative still falls within the time window and is
within Hamming distance. This is O(n) for typical bursts and avoids the
n^2 of full pairwise comparison while still catching the case "twelve
nearly identical frames in two seconds."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

HASH_SIZE = 8  # 64-bit phash
HAMMING_DIST_DEFAULT = 6
TIME_WINDOW_DEFAULT = 3.0  # seconds

EXIF_DATETIME_ORIGINAL_TAG = 36867  # PIL constant; see ExifTags


@dataclass
class ClusterItem:
    """One frame's input to clustering: identity plus the bits needed to cluster + rank."""

    key: str  # opaque caller identifier (sha, stem, etc.) — not interpreted here
    phash: imagehash.ImageHash
    timestamp: float  # seconds since epoch
    score: float  # higher = better, used to pick best within a cluster


@dataclass
class Cluster:
    items: list[ClusterItem] = field(default_factory=list)

    @property
    def best(self) -> ClusterItem:
        return max(self.items, key=lambda i: i.score)

    def __len__(self) -> int:
        return len(self.items)


def phash_from_preview(preview_bgr: np.ndarray) -> imagehash.ImageHash:
    """imagehash needs a PIL Image; convert from cv2's BGR uint8."""
    rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb), hash_size=HASH_SIZE)


def exif_timestamp(path: Path) -> float | None:
    """Return DateTimeOriginal as a unix timestamp, or None if not present/parseable."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
        dto = exif.get(EXIF_DATETIME_ORIGINAL_TAG)
        if not dto:
            return None
        # EXIF format: 'YYYY:MM:DD HH:MM:SS'.
        return datetime.strptime(dto, "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:
        return None


def fallback_timestamp(path: Path) -> float:
    return path.stat().st_mtime


def best_timestamp(path: Path) -> float:
    """EXIF when available, file mtime otherwise. Always returns something."""
    ts = exif_timestamp(path)
    return ts if ts is not None else fallback_timestamp(path)


def cluster_bursts(
    items: list[ClusterItem],
    hamming_dist: int = HAMMING_DIST_DEFAULT,
    time_window: float = TIME_WINDOW_DEFAULT,
) -> list[Cluster]:
    if not items:
        return []
    ordered = sorted(items, key=lambda it: it.timestamp)
    clusters: list[Cluster] = []
    for it in ordered:
        placed = False
        # Walk newest-first; once we leave the time window we can stop.
        for cluster in reversed(clusters):
            rep = cluster.items[0]
            if it.timestamp - rep.timestamp > time_window:
                break
            if (it.phash - rep.phash) <= hamming_dist:
                cluster.items.append(it)
                placed = True
                break
        if not placed:
            clusters.append(Cluster(items=[it]))
    return clusters


def keep_best_per_cluster(clusters: list[Cluster]) -> list[ClusterItem]:
    return [c.best for c in clusters]
