"""Face identity clustering + face-diverse selection smoke tests.

Bypasses insightface (which would download ~280 MB on the test machine)
by feeding hand-crafted embeddings directly into cluster_faces and the
selector. The dependency boundary between extract_face_embeddings and
cluster_faces is the natural seam for this.
"""

import numpy as np

from banger.face_id import (
    cluster_faces,
    decode_from_cache,
    encode_for_cache,
    person_count,
)
from banger.frames import Frame
from banger.report import Row
from banger.select import select_faces_top_n


def _emb(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _jitter(base: np.ndarray, seed: int, mag: float = 0.01) -> np.ndarray:
    # Note: in N-dim the noise vector has norm ~sqrt(N)*mag, so even mag=0.05
    # in 512-dim moves the resulting unit vector by cosine-distance ~0.33,
    # well past our 0.5 eps when sums of noise add up across components.
    # 0.01 keeps the cosine distance < 0.05, comfortably inside the cluster.
    rng = np.random.default_rng(seed)
    v = base + rng.standard_normal(base.shape).astype(np.float32) * mag
    return v / np.linalg.norm(v)


def test_cluster_faces_empty_input():
    assert cluster_faces([]) == []
    assert cluster_faces([[]]) == [set()]


def test_cluster_faces_groups_similar_embeddings():
    person_a = _emb(0)
    person_b = _emb(1)
    per_frame = [
        [_jitter(person_a, 10)],          # frame 0: person A
        [_jitter(person_a, 11)],          # frame 1: also A
        [_jitter(person_b, 20)],          # frame 2: B
        [_jitter(person_a, 12), _jitter(person_b, 21)],  # frame 3: A and B
    ]
    clusters = cluster_faces(per_frame)
    # Two people total across the batch.
    assert person_count(clusters) == 2
    # Frames 0,1 share a cluster; 2 has only the other; 3 has both.
    assert clusters[0] == clusters[1]
    assert clusters[0] != clusters[2]
    assert clusters[3] == clusters[0] | clusters[2]


def test_encode_decode_roundtrip():
    e1, e2 = _emb(99), _emb(100)
    encoded = encode_for_cache([e1, e2])
    decoded = decode_from_cache(encoded)
    assert len(decoded) == 2
    assert np.allclose(decoded[0], e1, atol=1e-6)


def test_decode_tolerates_missing():
    assert decode_from_cache(None) == []
    assert decode_from_cache([]) == []
    assert decode_from_cache(["not-a-list"]) == []


def _row(tmp_path, stem: str) -> Row:
    src = tmp_path / f"{stem}.JPG"
    src.write_bytes(b"")
    return Row(
        frame=Frame(stem=stem, subdir="", jpeg=src, raw=None),
        sharpness=400.0, aesthetic=0.0, aesthetic_breakdown=None,
        aesthetic_source="head", thumb_b64="",
    )


def test_select_faces_top_n_picks_one_per_person(tmp_path):
    person_a = _emb(0)
    person_b = _emb(1)
    person_c = _emb(2)

    rows = [_row(tmp_path, f"DSC{i:05d}") for i in range(6)]
    fake_emb = lambda: _emb(99 + 0)  # CLIP embedding, value doesn't matter for face strategy
    items = [(rows[i], float(10 - i), fake_emb()) for i in range(6)]
    face_embs = [
        [_jitter(person_a, 100)],          # 0: A, best score (10)
        [_jitter(person_a, 101)],          # 1: A
        [_jitter(person_b, 200)],          # 2: B (score 8)
        [_jitter(person_b, 201)],          # 3: B
        [_jitter(person_c, 300)],          # 4: C (score 6)
        [],                                # 5: no face
    ]
    chosen = select_faces_top_n(items, face_embs_per_item=face_embs, n=3)
    stems = [r.frame.stem for r, _, _ in chosen]
    # Best frame per person, sorted by score descending.
    assert stems == ["DSC00000", "DSC00002", "DSC00004"]


def test_select_faces_top_n_fills_with_kmeans_when_few_people(tmp_path):
    person_a = _emb(0)
    rows = [_row(tmp_path, f"DSC{i:05d}") for i in range(5)]
    # Vary CLIP embeddings so kmeans has something to cluster.
    items = [(rows[i], float(10 - i), _emb(900 + i)) for i in range(5)]
    face_embs = [
        [_jitter(person_a, 100)],   # 0: A (score 10) → chosen as person rep
        [],                         # 1
        [],                         # 2
        [],                         # 3
        [],                         # 4
    ]
    chosen = select_faces_top_n(items, face_embs_per_item=face_embs, n=4)
    assert len(chosen) == 4
    stems = [r.frame.stem for r, _, _ in chosen]
    assert "DSC00000" in stems  # the person rep
    # Remaining 3 fillers come from the kmeans path over the remaining 4 items.


def test_select_faces_top_n_no_faces_falls_back_to_kmeans(tmp_path):
    rows = [_row(tmp_path, f"DSC{i:05d}") for i in range(5)]
    items = [(rows[i], float(10 - i), _emb(900 + i)) for i in range(5)]
    chosen = select_faces_top_n(items, face_embs_per_item=[[] for _ in rows], n=3)
    assert len(chosen) == 3


def test_select_faces_top_n_rejects_mismatched_lengths(tmp_path):
    rows = [_row(tmp_path, "A")]
    items = [(rows[0], 1.0, _emb(0))]
    try:
        select_faces_top_n(items, face_embs_per_item=[], n=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
