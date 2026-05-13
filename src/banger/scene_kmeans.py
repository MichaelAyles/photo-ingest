"""KMeans-based scene router over cached CLIP embeddings.

The original scenes.py picks the closest of seven hand-written prompts.
That worked OK in v0 but in practice 97% of frames fell back at the
0.25 threshold and 90% fell back at 0.20, CLIP text-image sims for these
prompts cluster too tightly to discriminate confidently. Replacing the
prompt-vs-image cosine routine with KMeans over the image embeddings
themselves sidesteps that ambiguity: clusters drop out of the actual
data, and the user authors a darktable preset per cluster after seeing
the cluster gallery.

Fit step: run `banger scenes fit -k 5` after a typical batch has been
processed. We walk the embeddings cache, fit KMeans, persist the model
plus a cluster summary (count, nearest-neighbour prompt) to
state.SCENE_MODEL. cmd_run then auto-loads it. With no model on disk
the pipeline falls back to scenes.classify (prompt-based) so nothing
breaks before you run the fit step.

Each cluster maps to a preset slot named cluster_NN. Authoring is then
"build a darktable XMP for cluster_03 that suits the photos in that
cluster's gallery", concrete and grounded in your data, where the
prompt approach asked you to guess in advance.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from banger import state
from banger.scenes import SCENE_PROMPTS

log = logging.getLogger("banger")

SCENE_MODEL_PATH = state.STATE_DIR / "scene_kmeans.joblib"
SCENE_SUMMARY_PATH = state.STATE_DIR / "scene_kmeans.json"


@dataclass
class ClusterInfo:
    cluster_id: int
    preset: str  # e.g. "cluster_03"
    size: int  # number of training embeddings in this cluster
    centroid: np.ndarray  # L2-normalised, shape (emb_dim,)
    nearest_prompts: list[tuple[str, float]]  # top-3 SCENE_PROMPTS by cosine sim


@dataclass
class SceneClusters:
    k: int
    clusters: list[ClusterInfo]
    model: object  # sklearn KMeans, kept for fast predict()

    def classify_embedding(self, emb: np.ndarray) -> ClusterInfo:
        cid = int(self.model.predict(emb.reshape(1, -1))[0])
        return self.clusters[cid]


def _load_cached_embeddings() -> tuple[list[str], np.ndarray]:
    """Return (sha-list, NxD float32 matrix) from state.EMBEDDINGS_DIR."""
    if not state.EMBEDDINGS_DIR.exists():
        return [], np.zeros((0, 0), dtype=np.float32)
    shas: list[str] = []
    embs: list[np.ndarray] = []
    for p in sorted(state.EMBEDDINGS_DIR.glob("*.npy")):
        try:
            e = np.load(p)
        except (OSError, ValueError):
            continue
        if e.ndim != 1:
            continue
        embs.append(e.astype(np.float32))
        shas.append(p.stem)
    if not embs:
        return [], np.zeros((0, 0), dtype=np.float32)
    return shas, np.stack(embs)


def _nearest_prompts(centroid: np.ndarray, text_emb, top_n: int = 3) -> list[tuple[str, float]]:
    """Return the top_n SCENE_PROMPTS by cosine sim to this centroid (centroid pre-normalised)."""
    import torch

    c = torch.from_numpy(centroid).to(text_emb.device).to(text_emb.dtype)
    sims = (c @ text_emb.T).cpu().tolist()
    ranked = sorted(zip(SCENE_PROMPTS, sims, strict=True), key=lambda kv: -kv[1])
    return [(p, float(s)) for p, s in ranked[:top_n]]


def fit(k: int = 5, random_state: int = 0) -> SceneClusters | None:
    """Cluster all cached embeddings with KMeans. Persist the model + summary."""
    from sklearn.cluster import KMeans

    shas, X = _load_cached_embeddings()
    if X.shape[0] < k:
        log.error(
            "need at least k=%d cached embeddings to fit; have %d. "
            "Run `banger run` over a folder first.",
            k, X.shape[0],
        )
        return None

    log.info("fitting KMeans(k=%d) on %d cached embeddings (dim=%d)", k, X.shape[0], X.shape[1])
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)

    # Normalise centroids so cosine sim against text_emb is meaningful.
    centroids = km.cluster_centers_.astype(np.float32)
    centroids = centroids / np.maximum(
        np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8
    )

    from banger.scenes import _encoded_prompts

    text_emb = _encoded_prompts()

    clusters: list[ClusterInfo] = []
    for cid in range(k):
        members = int(np.sum(labels == cid))
        info = ClusterInfo(
            cluster_id=cid,
            preset=f"cluster_{cid:02d}",
            size=members,
            centroid=centroids[cid],
            nearest_prompts=_nearest_prompts(centroids[cid], text_emb),
        )
        clusters.append(info)

    # Persist.
    import joblib

    state.STATE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, SCENE_MODEL_PATH)
    SCENE_SUMMARY_PATH.write_text(
        json.dumps(
            {
                "k": k,
                "trained_on": int(X.shape[0]),
                "clusters": [
                    {
                        "cluster_id": c.cluster_id,
                        "preset": c.preset,
                        "size": c.size,
                        "nearest_prompts": [{"prompt": p, "sim": s} for p, s in c.nearest_prompts],
                    }
                    for c in clusters
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Clear the lazy load cache.
    load.cache_clear()

    log.info("trained on %d embeddings, saved to %s", X.shape[0], SCENE_MODEL_PATH)
    for c in clusters:
        top = c.nearest_prompts[0]
        log.info(
            "  %s (n=%d): top prompt = %r (sim %.3f)",
            c.preset, c.size, top[0], top[1],
        )
    return SceneClusters(k=k, clusters=clusters, model=km)


@lru_cache(maxsize=1)
def load() -> SceneClusters | None:
    """Lazy-load the persisted KMeans + summary; returns None if not fit yet."""
    if not SCENE_MODEL_PATH.exists() or not SCENE_SUMMARY_PATH.exists():
        return None
    import joblib

    try:
        km = joblib.load(SCENE_MODEL_PATH)
        summary = json.loads(SCENE_SUMMARY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as e:
        log.warning("scene kmeans load failed: %s", e)
        return None

    centroids = km.cluster_centers_.astype(np.float32)
    centroids = centroids / np.maximum(
        np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8
    )
    clusters: list[ClusterInfo] = []
    for entry in summary["clusters"]:
        cid = entry["cluster_id"]
        nps = [(np_["prompt"], float(np_["sim"])) for np_ in entry["nearest_prompts"]]
        clusters.append(
            ClusterInfo(
                cluster_id=cid,
                preset=entry["preset"],
                size=entry["size"],
                centroid=centroids[cid],
                nearest_prompts=nps,
            )
        )
    return SceneClusters(k=summary["k"], clusters=clusters, model=km)


def exists() -> bool:
    return SCENE_MODEL_PATH.exists()


def print_summary() -> int:
    """CLI helper: print the persisted summary, or a hint to fit."""
    if not SCENE_SUMMARY_PATH.exists():
        print("(no scene-kmeans model, run `banger scenes fit -k 5`)")
        return 0
    summary = json.loads(SCENE_SUMMARY_PATH.read_text(encoding="utf-8"))
    print(f"KMeans(k={summary['k']}) trained on {summary['trained_on']} embeddings")
    for c in summary["clusters"]:
        print(f"  {c['preset']}  n={c['size']:>4}")
        for np_ in c["nearest_prompts"]:
            print(f"      {np_['sim']:+.3f}  {np_['prompt']}")
    return 0
