"""Personal taste head: ridge regression on cached CLIP embeddings.

Trained from scores written by `banger label` or `banger ui`. Replaces
prompt-based aesthetic scoring in cmd_run when the head exists on disk.
Output is a continuous score, calibrated against your own -5..+5 labels.
"""

import logging

import joblib
import numpy as np

from banger import state

log = logging.getLogger("banger")


def exists() -> bool:
    return state.TASTE_HEAD.exists()


def load():
    return joblib.load(state.TASTE_HEAD) if exists() else None


def predict_score(model, embedding: np.ndarray) -> float:
    """Return predicted score (roughly in the user's labelling range)."""
    return float(model.predict(embedding.reshape(1, -1))[0])


def train_from_disk():
    from sklearn.linear_model import Ridge

    rows = state.all_labels()
    if not rows:
        log.error("no labels yet — use the UI (`banger ui <dir>`) or CLI to score frames first")
        return None

    X: list[np.ndarray] = []
    y: list[float] = []
    missing: list[tuple[str, str]] = []
    for sha, score, stem, _src, _ts in rows:
        emb = state.load_embedding(sha)
        if emb is None:
            missing.append((stem, sha))
            continue
        X.append(emb)
        y.append(float(score))

    if missing:
        log.warning(
            "%d labelled frames have no cached embedding "
            "(re-run `banger run` over their source dirs first):",
            len(missing),
        )
        for stem, sha in missing[:5]:
            log.warning("  %s (%s)", stem, sha[:12])

    if len(y) < 2:
        log.error("need at least 2 labelled frames to train; got %d", len(y))
        return None

    X_arr = np.stack(X)
    y_arr = np.array(y)

    log.info(
        "training Ridge on %d labels (range %.1f..%.1f, mean %.2f, embedding dim=%d)",
        len(y_arr),
        float(y_arr.min()),
        float(y_arr.max()),
        float(y_arr.mean()),
        X_arr.shape[1],
    )

    model = Ridge(alpha=1.0)
    model.fit(X_arr, y_arr)

    if 4 <= len(y_arr) <= 200:
        from sklearn.model_selection import LeaveOneOut, cross_val_score

        # MAE on leave-one-out — interpretable in score units.
        scores = cross_val_score(
            model, X_arr, y_arr, cv=LeaveOneOut(), scoring="neg_mean_absolute_error"
        )
        log.info(
            "leave-one-out MAE: %.2f (in score units; chance ≈ %.2f)",
            -float(scores.mean()),
            float(np.abs(y_arr - y_arr.mean()).mean()),
        )

    state.STATE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, state.TASTE_HEAD)
    log.info("saved taste head: %s", state.TASTE_HEAD)
    return model
