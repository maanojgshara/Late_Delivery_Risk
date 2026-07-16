"""Model construction, cross-validated baselines, tuning and calibration.

Builds the three class-weighted candidates (logreg / random_forest / hist_gbm) each paired
with its model-type preprocessor, scores them with leakage-free OOF cross-validation, runs a
randomized hyperparameter search (per-candidate progress bar) and calibrates the tuned models
with a cost-aware decision threshold.
"""
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, ParameterSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (average_precision_score, roc_auc_score, recall_score,
                             precision_score, f1_score, balanced_accuracy_score, brier_score_loss)

from .utils import tqdm
from .preprocess import build_tree_preprocessor, build_linear_preprocessor

SCORING_PRIMARY = "average_precision"


def build_candidate_models(cfg):
    """Create the three candidate estimators with class weights (imbalance action #1).

    Each entry pairs an estimator with its preprocessor kind ('linear' or 'tree').
    Parameters: cfg supplies random_state.
    Returns: dict {name: (kind, estimator)} for logreg / random_forest / hist_gbm.
    """
    rs = cfg["random_state"]
    return {
        "logreg":        ("linear", LogisticRegression(max_iter=2000, C=1.0,
                                       class_weight="balanced", random_state=rs)),
        "random_forest": ("tree",   RandomForestClassifier(n_estimators=300,
                                       class_weight="balanced_subsample", n_jobs=-1, random_state=rs)),
        "hist_gbm":      ("tree",   HistGradientBoostingClassifier(learning_rate=0.1,
                                       max_iter=300, class_weight="balanced", random_state=rs)),
    }


def make_model_pipeline(kind, est, cfg, multicol):
    """Wrap an estimator with its model-type-appropriate preprocessor into one Pipeline.

    Parameters: kind is 'linear'/'tree'; est is the estimator; cfg is the config;
    multicol holds 'linear_numeric'/'tree_numeric' from the multicollinearity step.
    Returns: an unfitted Pipeline (preprocessor -> classifier with a fresh estimator clone).
    """
    if kind == "linear":
        pre = build_linear_preprocessor(cfg, multicol["linear_numeric"])
    else:
        pre = build_tree_preprocessor(cfg, multicol["tree_numeric"])
    return Pipeline([("pre", pre), ("clf", clone(est))])


def oof_probabilities(pipe, X, y, cv, desc="CV"):
    """Out-of-fold P(late) via an explicit stratified-CV loop with a 0-100% progress bar.

    Each fold trains a clone of the pipeline and predicts its held-out rows (leakage-free).
    Parameters: pipe is the model pipeline; X/y the training data; cv the splitter; desc the bar label.
    Returns: a 1-D numpy array of OOF probabilities aligned with y.
    """
    oof = np.zeros(len(y), dtype=float); yv = np.asarray(y)
    for tr, va in tqdm(list(cv.split(X, yv)), total=cv.get_n_splits(), desc=desc, leave=True):
        m = clone(pipe).fit(X.iloc[tr], yv[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def metrics_at(y, proba, thr=0.5):
    """Compute the imbalance-aware metric suite at a probability threshold.
    Parameters: y is labels; proba is P(late); thr is the decision threshold.
    Returns: a dict of PR-AUC, ROC-AUC, recall, precision, F1, balanced accuracy and Brier.
    """
    pred = (proba >= thr).astype(int)
    return {"PR_AUC": average_precision_score(y, proba), "ROC_AUC": roc_auc_score(y, proba),
            "recall": recall_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred), "balanced_acc": balanced_accuracy_score(y, pred),
            "brier": brier_score_loss(y, proba)}


def param_distributions(name, cfg):
    """Return the randomized-search grid for a model from the config.
    Parameters: name is the model key; cfg holds models[name]['param_grid'].
    Returns: a dict of {pipeline_param: list_of_values}.
    """
    return cfg["models"][name]["param_grid"]


def run_search(name, kind, est, X_train, y_train, cfg, multicol):
    """Randomized hyperparameter search as an explicit candidate loop (reliable 0-100% bar).

    Samples search_n_iter configs, scores each by mean CV average-precision, keeps the original
    typed param dicts, and refits the best on the full training set.
    Parameters: name/kind/est identify the model; X_train/y_train the data; cfg the config;
    multicol the multicollinearity result.
    Returns: (best_pipeline_refit, best_params, best_score, trials_dataframe).
    """
    sampler = list(ParameterSampler(param_distributions(name, cfg),
                                    n_iter=cfg["search_n_iter"], random_state=cfg["random_state"]))
    scored = []
    for params in tqdm(sampler, total=len(sampler), desc=f"{name} tuning", leave=True):
        pipe = make_model_pipeline(kind, est, cfg, multicol).set_params(**params)
        score = cross_val_score(pipe, X_train, y_train, scoring=SCORING_PRIMARY,
                                cv=cfg["search_cv"], n_jobs=-1).mean()
        scored.append((params, float(score)))
    # pick from the ORIGINAL sampled dicts -> preserves int/None/str types (a DataFrame would coerce e.g. max_depth 30 -> 30.0)
    best_params, best_score = max(scored, key=lambda ps: ps[1])
    trials_df = (pd.DataFrame([{**p, "mean_cv_PR_AUC": s} for p, s in scored])
                 .sort_values("mean_cv_PR_AUC", ascending=False).reset_index(drop=True))
    best_pipe = make_model_pipeline(kind, est, cfg, multicol).set_params(**best_params).fit(X_train, y_train)
    return best_pipe, best_params, best_score, trials_df


def choose_threshold_cost(y_true, proba, fn_fp_ratio):
    """Pick the probability threshold minimising expected cost (FN weighted fn_fp_ratio x FP).
    Parameters: y_true is labels; proba is P(late); fn_fp_ratio is the FN:FP cost ratio.
    Returns: the cost-minimising threshold (float).
    """
    grid = np.linspace(0.01, 0.99, 99); best_t, best_c = 0.5, np.inf; y_true = np.asarray(y_true)
    for t in grid:
        pred = (proba >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum()); fp = int(((pred == 1) & (y_true == 0)).sum())
        # weight FN ~5x FP: missing a late delivery is costlier than a false alarm
        cost = fn_fp_ratio * fn + fp
        if cost < best_c:
            best_c, best_t = cost, t
    return float(best_t)


def calibrate_prefit(base_fitted, X_cal, y_cal, method):
    """Calibrate an already-fitted estimator on a held-out slice, robust across sklearn versions.

    Uses FrozenEstimator (sklearn>=1.6) when available, otherwise the legacy cv='prefit' API.
    Parameters: base_fitted is the fitted estimator; X_cal/y_cal the calibration slice; method the
    calibration method ('isotonic'/'sigmoid').
    Returns: a fitted CalibratedClassifierCV.
    """
    try:
        # sklearn >=1.6: freeze the fitted model so CalibratedClassifierCV calibrates WITHOUT refitting
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(FrozenEstimator(base_fitted), method=method).fit(X_cal, y_cal)
    except Exception:
        return CalibratedClassifierCV(base_fitted, method=method, cv="prefit").fit(X_cal, y_cal)
