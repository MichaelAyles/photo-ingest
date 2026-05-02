"""Personal taste head: logistic regression on top of cached CLIP embeddings.

Trained from `banger label` data. Replaces prompt-based aesthetic scoring
in cmd_run when the head exists on disk.
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
    """Map P(up) to a 0-10 score so it stacks with the existing report scale."""
    prob_up = model.predict_proba(embedding.reshape(1, -1))[0, 1]
    return float(prob_up * 10.0)


def train_from_disk():
    from sklearn.linear_model import LogisticRegression

    rows = state.all_labels()
    if not rows:
        log.error("no labels yet — run `banger label <dir> up|down <stem>...` first")
        return None

    X: list[np.ndarray] = []
    y: list[int] = []
    missing: list[tuple[str, str]] = []
    for sha, label, stem, _src, _ts in rows:
        emb = state.load_embedding(sha)
        if emb is None:
            missing.append((stem, sha))
            continue
        X.append(emb)
        y.append(1 if label == "up" else 0)

    if missing:
        log.warning(
            "%d labelled frames have no cached embedding "
            "(run `banger run` over the source dir first):",
            len(missing),
        )
        for stem, sha in missing[:5]:
            log.warning("  %s (%s)", stem, sha[:12])

    if len(set(y)) < 2:
        log.error(
            "need both up and down labels to train; got %d up, %d down",
            sum(y),
            len(y) - sum(y),
        )
        return None

    X_arr = np.stack(X)
    y_arr = np.array(y)

    log.info(
        "training on %d up + %d down labels (embedding dim=%d)",
        int(y_arr.sum()),
        int(len(y_arr) - y_arr.sum()),
        X_arr.shape[1],
    )

    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(X_arr, y_arr)

    if 4 <= len(y_arr) <= 200:
        from sklearn.model_selection import LeaveOneOut, cross_val_score

        scores = cross_val_score(model, X_arr, y_arr, cv=LeaveOneOut())
        log.info(
            "leave-one-out accuracy: %.2f over %d folds (chance = %.2f)",
            float(scores.mean()),
            len(scores),
            float(max(y_arr.mean(), 1 - y_arr.mean())),
        )

    state.STATE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, state.TASTE_HEAD)
    log.info("saved taste head: %s", state.TASTE_HEAD)
    return model
