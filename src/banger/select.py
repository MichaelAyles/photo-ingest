"""Top-N selection with MMR diversity over CLIP embeddings.

Plain top-K by aesthetic score finds the densest cluster of "good" frames —
on a real dataset that's a long run of near-identical portraits from one
good-weather day. MMR (maximal marginal relevance) trades some relevance
for visual diversity: each pick maximises (lambda * normalised_score) -
((1 - lambda) * max_cosine_similarity_to_already_picked), so each chosen
frame is forced to be different from its predecessors.

Lambda = 1.0 reproduces plain top-K (no diversity penalty).
Lambda = 0.5 weights relevance and diversity equally — sensible default
for a 10-frame highlight reel.
Lambda = 0.0 picks the most-different frames regardless of score
(useful for "give me a sample of the dataset" rather than "best of").
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np

T = TypeVar("T")


def normalise_scores(scores: list[float]) -> list[float]:
    """Min-max scale to [0, 1]; degenerate (all equal) → 0.5 across the board."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def select_diverse_top_n(
    items: list[tuple[T, float, np.ndarray]],
    n: int,
    diversity_lambda: float = 0.5,
) -> list[tuple[T, float, np.ndarray]]:
    """Pick `n` items from `items` using MMR.

    items: list of (payload, score, embedding). Embeddings should be
        L2-normalised so dot product = cosine similarity. Order doesn't
        matter — the selection is greedy on MMR, not on input order.
    n: max selections.
    diversity_lambda: 0..1; 1.0 = pure top-N by score.

    Returns the selected items in the order they were chosen (= rank).
    """
    if not items or n <= 0:
        return []
    if diversity_lambda >= 1.0:
        # Plain top-N by score; preserve original payload + embedding refs.
        return sorted(items, key=lambda it: -it[1])[:n]

    scores = [s for _, s, _ in items]
    norm = normalise_scores(scores)
    pool: list[tuple[T, float, np.ndarray, float]] = [
        (payload, score, emb, ns) for (payload, score, emb), ns in zip(items, norm, strict=True)
    ]

    # First pick is always the highest-relevance frame.
    pool.sort(key=lambda it: -it[3])
    selected = [pool.pop(0)]

    while len(selected) < n and pool:
        best_mmr = -float("inf")
        best_idx = 0
        for i, (_, _, emb, ns) in enumerate(pool):
            max_sim = max(float(emb @ s_emb) for _, _, s_emb, _ in selected)
            mmr = diversity_lambda * ns - (1.0 - diversity_lambda) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
        selected.append(pool.pop(best_idx))

    return [(payload, score, emb) for payload, score, emb, _ in selected]


def select_top_k(
    items: list[tuple[T, float, np.ndarray]], n: int
) -> list[tuple[T, float, np.ndarray]]:
    """Plain sort-by-score-desc, take first N. No diversity awareness."""
    return sorted(items, key=lambda it: -it[1])[:n]


def select_faces_top_n(
    items: list[tuple[T, float, np.ndarray]],
    face_embs_per_item: list[list[np.ndarray]],
    n: int,
    random_state: int = 0,
) -> list[tuple[T, float, np.ndarray]]:
    """Identity-diverse selection: one good shot per detected person, the
    Aftershoot trick. Frames without faces fall back to kmeans-on-CLIP for
    the remaining slots so a folder of landscapes still gets visual variety.

    The argument shape mirrors select_kmeans_top_n with an extra parallel
    list of per-item face embeddings (already extracted upstream via
    banger.face_id). Returns at most `n` items in chosen order.
    """
    if not items or n <= 0:
        return []
    if len(face_embs_per_item) != len(items):
        raise ValueError("face_embs_per_item must be the same length as items")

    from banger.face_id import cluster_faces

    per_frame = cluster_faces(face_embs_per_item)
    # Find each person cluster's best-scoring frame (the "rep" for that person).
    best_for_person: dict[int, int] = {}  # person_id -> index into items
    for idx, persons in enumerate(per_frame):
        score = items[idx][1]
        for pid in persons:
            cur = best_for_person.get(pid)
            if cur is None or items[cur][1] < score:
                best_for_person[pid] = idx

    # Dedupe: one frame might be the best for multiple people (group photo).
    # That frame still gets picked once; we lose no information because the
    # reps below it for those other people will be lower-scoring.
    chosen_idx: list[int] = []
    seen: set[int] = set()
    # Sort person reps by frame score descending so the strongest face cluster
    # leads.
    for pid in sorted(best_for_person, key=lambda p: -items[best_for_person[p]][1]):
        idx = best_for_person[pid]
        if idx in seen:
            continue
        chosen_idx.append(idx)
        seen.add(idx)
        if len(chosen_idx) >= n:
            break

    if len(chosen_idx) >= n:
        return [items[i] for i in chosen_idx]

    # Fill remaining slots with kmeans-on-CLIP picks from the leftover pool.
    remaining = [items[i] for i in range(len(items)) if i not in seen]
    needed = n - len(chosen_idx)
    fillers = select_kmeans_top_n(remaining, n=needed, random_state=random_state)
    return [items[i] for i in chosen_idx] + fillers


def select_kmeans_top_n(
    items: list[tuple[T, float, np.ndarray]],
    n: int,
    random_state: int = 0,
) -> list[tuple[T, float, np.ndarray]]:
    """Strong-diversity selection: cluster candidates into N visual groups via
    k-means on their embeddings, take the highest-scoring frame from each.

    Guarantees N visually distinct picks (one per cluster) when len(items) > n.
    Useful when the goal is portfolio variety — the wedding-shoot case where
    "100 excellent shots of the bride and groom" should yield ONE picked
    portrait alongside picks from every other category.
    """
    if not items or n <= 0:
        return []
    if len(items) <= n:
        return select_top_k(items, n)

    from sklearn.cluster import KMeans

    embs = np.stack([e for _, _, e in items])
    n_clusters = min(n, len(items))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(embs)

    by_cluster: dict[int, tuple[T, float, np.ndarray]] = {}
    for label, item in zip(labels.tolist(), items, strict=True):
        if label not in by_cluster or item[1] > by_cluster[label][1]:
            by_cluster[label] = item

    return sorted(by_cluster.values(), key=lambda it: -it[1])
