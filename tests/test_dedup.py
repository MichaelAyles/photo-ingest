"""Burst dedup: clustering rules + best-per-cluster selection."""


import imagehash
import numpy as np
import pytest

from banger.dedup import (
    Cluster,
    ClusterItem,
    cluster_bursts,
    exif_timestamp,
    fallback_timestamp,
    keep_best_per_cluster,
    phash_from_preview,
)


def _hash_from_int(bits: int, hash_size: int = 8) -> imagehash.ImageHash:
    """Build a synthetic ImageHash from a 64-bit int — saves making real images."""
    n = hash_size * hash_size  # 64 bits for hash_size=8
    arr = np.array([(bits >> i) & 1 for i in range(n)], dtype=bool).reshape(hash_size, hash_size)
    return imagehash.ImageHash(arr)


def test_identical_close_in_time_form_one_cluster():
    h = _hash_from_int(0xDEADBEEF_DEADBEEF)
    items = [
        ClusterItem("a", h, 1000.0, score=1.0),
        ClusterItem("b", h, 1000.5, score=2.0),
        ClusterItem("c", h, 1001.0, score=0.5),
    ]
    clusters = cluster_bursts(items)
    assert len(clusters) == 1
    assert {it.key for it in clusters[0].items} == {"a", "b", "c"}


def test_identical_far_apart_in_time_split():
    h = _hash_from_int(0xDEADBEEF)
    items = [
        ClusterItem("a", h, 1000.0, score=1.0),
        ClusterItem("b", h, 2000.0, score=1.0),
    ]
    clusters = cluster_bursts(items, time_window=3.0)
    assert len(clusters) == 2


def test_different_hashes_close_in_time_split():
    h1 = _hash_from_int(0x0)
    h2 = _hash_from_int(0xFFFFFFFF_FFFFFFFF)  # max hamming distance from h1 = 64
    items = [
        ClusterItem("a", h1, 1000.0, score=1.0),
        ClusterItem("b", h2, 1000.5, score=1.0),
    ]
    clusters = cluster_bursts(items, hamming_dist=6)
    assert len(clusters) == 2


def test_keep_best_picks_highest_score():
    h = _hash_from_int(0x1)
    items = [
        ClusterItem("low", h, 1000.0, score=1.0),
        ClusterItem("mid", h, 1000.5, score=3.5),
        ClusterItem("high", h, 1001.0, score=4.0),
    ]
    clusters = cluster_bursts(items)
    assert len(clusters) == 1
    best = keep_best_per_cluster(clusters)
    assert [it.key for it in best] == ["high"]


def test_singletons_pass_through_unchanged():
    items = [
        ClusterItem("a", _hash_from_int(0x1), 1000.0, score=1.0),
        ClusterItem("b", _hash_from_int(0xFFFF_FFFF_FFFF_FFFF), 2000.0, score=2.0),
        ClusterItem("c", _hash_from_int(0xAAAA_AAAA_AAAA_AAAA), 3000.0, score=3.0),
    ]
    clusters = cluster_bursts(items)
    assert len(clusters) == 3
    assert sum(len(c) for c in clusters) == 3


def test_empty_input_returns_empty():
    assert cluster_bursts([]) == []
    assert keep_best_per_cluster([]) == []


def test_cluster_best_is_max_score():
    h = _hash_from_int(0x1)
    c = Cluster(
        items=[
            ClusterItem("x", h, 0.0, score=1.0),
            ClusterItem("y", h, 0.0, score=5.0),
            ClusterItem("z", h, 0.0, score=3.0),
        ]
    )
    assert c.best.key == "y"


def test_phash_from_preview_returns_imagehash(make_jpeg):
    from banger.preview import load_preview

    p = make_jpeg(sharpness="high")
    h = phash_from_preview(load_preview(p))
    assert isinstance(h, imagehash.ImageHash)


def test_phash_distance_zero_for_same_image(make_jpeg):
    from banger.preview import load_preview

    p = make_jpeg(sharpness="high")
    arr = load_preview(p)
    h1 = phash_from_preview(arr)
    h2 = phash_from_preview(arr)
    assert (h1 - h2) == 0


def test_phash_distance_nonzero_for_different_images(make_jpeg):
    from banger.preview import load_preview

    a = make_jpeg(name="a.JPG", sharpness="low")
    b = make_jpeg(name="b.JPG", sharpness="high")
    h_a = phash_from_preview(load_preview(a))
    h_b = phash_from_preview(load_preview(b))
    assert (h_a - h_b) > 0


def test_exif_timestamp_falls_back_to_mtime_for_synthetic_jpeg(make_jpeg):
    p = make_jpeg()
    # Synthetic JPEG won't have EXIF DateTimeOriginal.
    assert exif_timestamp(p) is None
    assert isinstance(fallback_timestamp(p), float)


@pytest.mark.parametrize(
    "hamming, expected_clusters",
    [
        (0, 2),  # tight: even tiny hamming distance splits
        (32, 1),  # loose: anything within 32 of 64 bits clusters
    ],
)
def test_hamming_threshold_controls_split(hamming, expected_clusters):
    # h1 and h2 differ in ~half their bits.
    h1 = _hash_from_int(0x0)
    h2 = _hash_from_int(0xAAAA_AAAA_AAAA_AAAA)
    actual_dist = h1 - h2
    items = [
        ClusterItem("a", h1, 1000.0, score=1.0),
        ClusterItem("b", h2, 1000.5, score=1.0),
    ]
    clusters = cluster_bursts(items, hamming_dist=hamming, time_window=10.0)
    if hamming >= actual_dist:
        assert len(clusters) == 1
    else:
        assert len(clusters) == 2


def test_disabled_returns_every_frame_as_singleton():
    # Three identical frames close in time would normally collapse to one
    # cluster; with enabled=False each stays in its own singleton.
    h = _hash_from_int(0xDEADBEEF_DEADBEEF)
    items = [
        ClusterItem("a", h, 1000.0, score=1.0),
        ClusterItem("b", h, 1000.5, score=2.0),
        ClusterItem("c", h, 1001.0, score=0.5),
    ]
    clusters = cluster_bursts(items, enabled=False)
    assert len(clusters) == 3
    assert all(len(c) == 1 for c in clusters)
    # All frames survive best-per-cluster selection when dedup is off.
    kept = {it.key for it in keep_best_per_cluster(clusters)}
    assert kept == {"a", "b", "c"}


def test_tighter_hamming_yields_more_groups():
    # Two pairs: within each pair the hashes differ by a few bits.
    # The default hamming (6) merges each pair; a stricter threshold splits them.
    h_a = _hash_from_int(0x0)
    h_a2 = _hash_from_int(0b1111)  # 4 bits from h_a
    h_b = _hash_from_int(0xFFFF_FFFF_FFFF_FFFF)
    h_b2 = _hash_from_int(0x0FFF_FFFF_FFFF_FFFF)  # 4 bits from h_b
    items = [
        ClusterItem("a", h_a, 1000.0, score=1.0),
        ClusterItem("a2", h_a2, 1000.2, score=1.0),
        ClusterItem("b", h_b, 1000.4, score=1.0),
        ClusterItem("b2", h_b2, 1000.6, score=1.0),
    ]
    # Default threshold (6) tolerates the 4-bit gap -> two groups.
    loose = cluster_bursts(items, time_window=10.0)
    # Stricter threshold (0) won't tolerate any gap -> four groups.
    tight = cluster_bursts(items, hamming_dist=0, time_window=10.0)
    assert len(loose) == 2
    assert len(tight) == 4
    assert len(tight) > len(loose)


def test_wider_time_window_groups_frames_default_would_split():
    # Two identical frames 5s apart: the default 3s window splits them,
    # but a 10s window treats them as one burst.
    h = _hash_from_int(0xDEADBEEF_DEADBEEF)
    items = [
        ClusterItem("a", h, 1000.0, score=1.0),
        ClusterItem("b", h, 1005.0, score=2.0),
    ]
    default = cluster_bursts(items)  # time_window=3.0
    wide = cluster_bursts(items, time_window=10.0)
    assert len(default) == 2
    assert len(wide) == 1
    assert {it.key for it in wide[0].items} == {"a", "b"}
