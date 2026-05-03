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
