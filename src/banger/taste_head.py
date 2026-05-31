"""Personal taste head: ridge regression on cached CLIP embeddings.

Trained from scores written by `banger label` or `banger ui`. Replaces
prompt-based aesthetic scoring in cmd_run when the head exists on disk.
Output is a continuous score, calibrated against your own -5..+5 labels.

The CLIP embeddings are high-dimensional (512-d) relative to the number of
labels a fresh user has produced (often a few dozen) — the classic p>>n
regime where a plain Ridge overfits and cross-validated alpha selection is
fragile. To handle this we:

  * standardize, then reduce dimensionality with PCA (to roughly
    min(50, n_samples-1, n_features)) once there are enough samples, so the
    downstream regressor sees far fewer features than labels; and
  * pick the ridge penalty with RidgeCV over a small alpha grid instead of a
    single hard-coded alpha — falling back to a sane fixed alpha when there
    are too few labels for cross-validation to be meaningful.

Everything is wrapped in an sklearn Pipeline so joblib save/load and the
`model.predict(X)` call site stay unchanged.

Cold start: a brand-new head trained on a handful of labels is unreliable.
`blend_with_prior` lets callers fold in a generic aesthetic prior (e.g.
`aesthetic.score_from_embedding`), weighting toward the prior when there are
few labels and toward the personal head as the label count grows. We accept
the prior as a plain number rather than importing aesthetic.py, to avoid a
cross-module / circular dependency.
"""

import logging

import joblib
import numpy as np

from banger import state

log = logging.getLogger("banger")

# Below this many labels we don't trust RidgeCV's cross-validation; use a
# fixed, fairly strong penalty instead.
_MIN_LABELS_FOR_CV = 6
# Only bother with PCA once we have enough samples that reducing dimensionality
# actually buys us something (and PCA has >1 component to keep).
_MIN_LABELS_FOR_PCA = 8
# Target number of PCA components, capped further by n_samples-1 / n_features.
_PCA_TARGET_COMPONENTS = 50
# Alpha grid for RidgeCV — spans weak to very strong regularization, which is
# what the p>>n regime wants.
_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
# Fixed penalty used in the very-few-labels fallback (stronger than the old
# alpha=1.0 default, since with a tiny n we want to lean on regularization).
_FALLBACK_ALPHA = 10.0
# Number of labels at which the personal head is considered fully trusted for
# blending purposes (blend weight saturates to 1.0 here).
_BLEND_FULL_TRUST_LABELS = 50


def exists() -> bool:
    return state.TASTE_HEAD.exists()


def load():
    return joblib.load(state.TASTE_HEAD) if exists() else None


def predict_score(model, embedding: np.ndarray) -> float:
    """Return predicted score (roughly in the user's labelling range)."""
    return float(model.predict(embedding.reshape(1, -1))[0])


def blend_with_prior(
    personal_score: float,
    prior_score: float,
    n_labels: int,
    *,
    full_trust_labels: int = _BLEND_FULL_TRUST_LABELS,
) -> float:
    """Blend a personal-head prediction with a generic prior.

    Cold-start hook: when the personal taste head was trained on only a few
    labels (or there is no head and the caller passes the prior as
    ``personal_score``), its predictions are noisy. We linearly ramp the
    weight given to the personal score from 0 at ``n_labels == 0`` up to 1.0
    at ``n_labels >= full_trust_labels``, weighting the remainder toward the
    generic ``prior_score``::

        w_personal = clamp(n_labels / full_trust_labels, 0, 1)
        result     = w_personal * personal_score + (1 - w_personal) * prior_score

    The ``prior_score`` is supplied numerically by the caller (e.g. from
    ``aesthetic.score_from_embedding``) so this module never imports the
    aesthetic model. With many labels the result is essentially the personal
    score; with none it is the prior.
    """
    if full_trust_labels <= 0:
        w_personal = 1.0
    else:
        w_personal = max(0.0, min(1.0, n_labels / float(full_trust_labels)))
    return float(w_personal * personal_score + (1.0 - w_personal) * prior_score)


def _build_pipeline(n_samples: int, n_features: int):
    """Construct a robust regression Pipeline sized for the available data.

    p>>n handling: standardize -> (optional) PCA -> Ridge/RidgeCV. With few
    labels we skip PCA and use a fixed strong alpha; with more labels we reduce
    dimensionality and cross-validate the penalty over a small grid.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [("scale", StandardScaler())]

    # PCA only helps once we have a few samples; keep it strictly below
    # n_samples (PCA needs n_components <= min(n_samples, n_features)) and
    # leave a degree of freedom for the regressor by using n_samples - 1.
    use_pca = n_samples >= _MIN_LABELS_FOR_PCA
    if use_pca:
        n_components = min(_PCA_TARGET_COMPONENTS, n_samples - 1, n_features)
        n_components = max(1, n_components)
        # Only worth it if we're actually shrinking the feature space.
        if n_components < n_features:
            steps.append(("pca", PCA(n_components=n_components, random_state=0)))

    # RidgeCV needs enough samples for its internal cross-validation to mean
    # anything; below that, fall back to a fixed, strongish alpha.
    if n_samples >= _MIN_LABELS_FOR_CV:
        # cv=None lets RidgeCV use efficient leave-one-out generalized CV.
        regressor = RidgeCV(alphas=_ALPHA_GRID)
    else:
        regressor = Ridge(alpha=_FALLBACK_ALPHA)

    steps.append(("ridge", regressor))
    return Pipeline(steps)


def train_from_disk():
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
    n_samples, n_features = X_arr.shape

    model = _build_pipeline(n_samples, n_features)

    # Describe the regime we're actually training in (helps debug cold start).
    regressor = model.named_steps["ridge"]
    reg_name = type(regressor).__name__
    has_pca = "pca" in model.named_steps
    pca_desc = (
        f", PCA->{model.named_steps['pca'].n_components} dims" if has_pca else ""
    )
    log.info(
        "training %s on %d labels (range %.1f..%.1f, mean %.2f, embedding dim=%d%s)",
        reg_name,
        len(y_arr),
        float(y_arr.min()),
        float(y_arr.max()),
        float(y_arr.mean()),
        n_features,
        pca_desc,
    )

    model.fit(X_arr, y_arr)

    # Report the cross-validated alpha that RidgeCV settled on, when available.
    chosen_alpha = getattr(model.named_steps["ridge"], "alpha_", None)
    if chosen_alpha is not None:
        log.info("RidgeCV selected alpha=%.3g", float(chosen_alpha))

    if 4 <= len(y_arr) <= 200:
        from sklearn.model_selection import LeaveOneOut, cross_val_score

        # MAE on leave-one-out — interpretable in score units. Re-fit a fresh
        # pipeline per fold via cross_val_score so PCA/RidgeCV are honest.
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


def save(model, path=None) -> None:
    """Persist a trained taste-head model to disk via joblib.

    Mirrors what ``train_from_disk`` does internally; exposed so callers can
    save a model they trained or post-processed themselves. Defaults to the
    canonical ``state.TASTE_HEAD`` location.
    """
    target = path if path is not None else state.TASTE_HEAD
    state.STATE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)
