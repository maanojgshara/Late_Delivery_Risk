"""End-to-end ML Pipeline.

Runs the full flow: 
    ingest (SQLite) -> clean -> engineer -> split -> multicollinearity ->
    train 3 calibrated models -> tune -> calibrate + threshold -> evaluate -> interpret (importance,
    SHAP) -> error & bias analysis -> select model -> risk x impact scoring. 

Configured via config.yaml with CLI overrides. 

"""
import argparse
import time
import matplotlib
matplotlib.use("Agg")                      # headless backend for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve

from . import feature_engineer as F
from .config import load_config
from .utils import Artifacts, label_bars
from .data_load import load_tables
from .preprocess import clean_data, compute_multicollinearity
from . import train as T
from . import evaluate as E
from .score import risk_impact_scores

SEG_COLS = ["branch", "parcel_category", "delivery_priority", "promised_service_tier"]


def parse_args():
    """Parse CLI arguments that override config.yaml for quick experimentation.
    Parameters: none (reads sys.argv).
    Returns: an argparse.Namespace of overrides.
    """
    p = argparse.ArgumentParser(description="MoveEasy late-delivery risk pipeline")
    p.add_argument("--config", default=None, help="path to config YAML")
    p.add_argument("--search-n-iter", type=int, default=None, help="randomized-search candidates")
    p.add_argument("--cv-folds", type=int, default=None, help="StratifiedKFold folds")
    p.add_argument("--test-size", type=float, default=None, help="hold-out test fraction")
    p.add_argument("--random-state", type=int, default=None, help="global seed")
    p.add_argument("--models", default=None, help="comma list subset, e.g. logreg,hist_gbm")
    return p.parse_args()


def _split(X, y, cfg, art):
    """Create the stratified hold-out split + StratifiedKFold and save the split report/figure.
    Parameters: X/y full data; cfg config; art saver.
    Returns: (X_train, X_test, y_train, y_test, cv).
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=cfg["random_state"])
    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["random_state"])
    rep = pd.DataFrame({"subset": ["full", "train", "test"],
                        "n_rows": [len(y), len(y_tr), len(y_te)],
                        "positive_rate": [round(y.mean(), 4), round(y_tr.mean(), 4), round(y_te.mean(), 4)]})
    art.logger.info(f"Split -> train={len(y_tr):,} test={len(y_te):,} | "
                    f"positive_rate={y_tr.mean():.4f}")
    art.table(rep, "split_report", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    pd.Series({"on-time": int((y_tr == 0).sum()), "late": int((y_tr == 1).sum())}).plot(
        kind="bar", ax=ax[0], color=["#4C72B0", "#C44E52"]); ax[0].set_title("Train class counts")
    ax[1].bar(rep["subset"], rep["positive_rate"], color="#4C72B0"); ax[1].set_ylim(0, 0.15)
    ax[1].set_title("Positive rate by subset")
    label_bars(ax[0], fmt="{:.0f}"); label_bars(ax[1], fmt="{:.3f}")
    plt.tight_layout(); art.fig(fig, "class_balance"); plt.close(fig)
    return X_tr, X_te, y_tr, y_te, cv


def _multicollinearity(X_train, cfg, art):
    """Run the multicollinearity diagnosis and save VIF/Spearman tables + figures.
    Parameters: X_train training features; cfg; art saver.
    Returns: the multicollinearity result dict.
    """
    mc = compute_multicollinearity(X_train, cfg)
    art.table(mc["vif"].rename("VIF"), "vif")
    art.table(mc["spearman"].round(3), "spearman_corr")
    if not mc["pairs"].empty:
        art.table(mc["pairs"], "high_corr_pairs", index=False)
    art.logger.info(f"Multicollinearity -> LINEAR_DROP={mc['linear_drop']} | "
                    f"linear keeps {len(mc['linear_numeric'])}, tree keeps {len(mc['tree_numeric'])}")
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(mc["spearman"], annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Spearman correlation - training numeric features")
    plt.tight_layout(); art.fig(fig, "spearman_heatmap"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    mc["vif"].sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.axvline(cfg["vif_threshold"], ls="--", color="red"); ax.set_title("VIF by numeric feature (train)")
    label_bars(ax, fmt="{:.1f}", horizontal=True)
    plt.tight_layout(); art.fig(fig, "vif_bar"); plt.close(fig)
    return mc


def _cv_baseline(candidates, X_train, y_train, cv, cfg, multicol, art):
    """Score candidates with OOF cross-validation; save baseline metrics, PR-AUC bar, confusion.
    Parameters: candidates dict; X_train/y_train; cv; cfg; multicol; art.
    Returns: dict of OOF probabilities per model.
    """
    rows, oof = [], {}
    for name, (kind, est) in candidates.items():
        pipe = T.make_model_pipeline(kind, est, cfg, multicol)
        oof[name] = T.oof_probabilities(pipe, X_train, y_train, cv, desc=f"{name} CV")
        m = T.metrics_at(y_train, oof[name], 0.5); m["model"] = name; rows.append(m)
        art.logger.info(f"[CV baseline] {name}: PR-AUC={m['PR_AUC']:.4f} recall={m['recall']:.3f}")
    baseline = pd.DataFrame(rows).set_index("model").round(4)
    art.table(baseline, "cv_baseline_metrics")
    fig, ax = plt.subplots(figsize=(7, 4))
    baseline["PR_AUC"].plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52", "#55A868"][:len(baseline)])
    ax.set_title("CV OOF PR-AUC by model"); label_bars(ax, fmt="{:.3f}")
    plt.tight_layout(); art.fig(fig, "cv_pr_auc"); plt.close(fig)
    return oof, baseline


def _tune(candidates, X_train, y_train, cfg, multicol, art):
    """Tune every candidate, save trial histories/best params/models and a summary bar.
    Parameters: candidates; X_train/y_train; cfg; multicol; art.
    Returns: dict {name: tuned_pipeline}.
    """
    tuned, rows = {}, []
    for name, (kind, est) in candidates.items():
        best_pipe, best_params, best_score, trials = T.run_search(name, kind, est, X_train, y_train, cfg, multicol)
        tuned[name] = best_pipe
        art.model(best_pipe, f"tuned_{name}", subdir=name)
        art.table(trials, f"trials_{name}", index=False, subdir=name)
        art.run({"model": name, "best_params": best_params, "best_cv_pr_auc": best_score},
                f"best_{name}", subdir=name)
        rows.append({"model": name, "best_cv_PR_AUC": round(best_score, 4), **best_params})
        art.logger.info(f"[tuned] {name}: PR-AUC={best_score:.4f} params={best_params}")
    tuning = pd.DataFrame(rows).set_index("model")
    art.table(tuning, "tuning_summary")
    fig, ax = plt.subplots(figsize=(7, 4))
    tuning["best_cv_PR_AUC"].plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52", "#55A868"][:len(tuning)])
    ax.set_title("Best CV PR-AUC after tuning"); label_bars(ax, fmt="{:.3f}")
    plt.tight_layout(); art.fig(fig, "tuned_pr_auc"); plt.close(fig)
    return tuned, tuning


def _calibrate(tuned, X_train, y_train, cfg, art):
    """Calibrate tuned models on a held-out slice; pick cost-aware thresholds; save artefacts.
    Parameters: tuned dict; X_train/y_train; cfg; art.
    Returns: (calibrated_models dict, thresholds dict).
    """
    # hold out 25% of TRAIN only, used for prefit calibration + threshold tuning (leak-free)
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=cfg["random_state"])
    cal_models, thresholds, rows = {}, {}, []
    fig, axes = plt.subplots(1, len(tuned), figsize=(5 * len(tuned), 4.5)); axes = np.atleast_1d(axes)
    for ax, (name, pipe) in zip(axes, tuned.items()):
        base = clone(pipe).fit(X_fit, y_fit)
        cal = T.calibrate_prefit(base, X_cal, y_cal, cfg["calib_method"])
        cal_models[name] = cal; art.model(cal, f"calibrated_{name}", subdir=name)
        p_un = base.predict_proba(X_cal)[:, 1]; p_ca = cal.predict_proba(X_cal)[:, 1]
        thresholds[name] = T.choose_threshold_cost(y_cal, p_ca, cfg["fn_fp_cost_ratio"])
        from sklearn.metrics import brier_score_loss
        rows.append({"model": name, "brier_uncalibrated": round(brier_score_loss(y_cal, p_un), 4),
                     "brier_calibrated": round(brier_score_loss(y_cal, p_ca), 4),
                     "threshold": round(thresholds[name], 3)})
        frac, mean_pred = calibration_curve(y_cal, p_ca, n_bins=10, strategy="uniform")
        ax.plot([0, 1], [0, 1], "--", color="grey"); ax.plot(mean_pred, frac, "o-", color="#C44E52")
        ax.set_title(f"{name} reliability"); ax.set_xlabel("mean predicted"); ax.set_ylabel("observed")
    plt.tight_layout(); art.fig(fig, "reliability_curves"); plt.close(fig)
    art.table(pd.DataFrame(rows).set_index("model"), "calibration_thresholds")
    art.logger.info(f"Calibration done | thresholds={thresholds}")
    return cal_models, thresholds, X_cal, y_cal


def run_pipeline(cfg, art, args):
    """Execute all pipeline stages in order (called by main inside error handling).
    Parameters: cfg is the config; art is the Artifacts saver; args holds CLI overrides.
    Returns: None (writes models to models/ and outputs to artifacts/).
    """
    # 1) Ingest + process
    print(">>>>>>>>>>>> Ingesting data... >>>>>>>>>>>>")
    tables = load_tables(cfg["data"]["db_path"], cfg["data"]["url"], art.logger)
    print("========= Data ingested =========\n")

    print(">>>>>>>>>>>> Cleaning data... >>>>>>>>>>>>")
    deliveries = clean_data(tables[cfg["data"]["deliveries_table"]])
    print("========= Data cleaned =========\n")

    print(">>>>>>>>>>>> Engineering features... >>>>>>>>>>>>")
    deliveries = F.engineer_features(deliveries)
    print("========= Features engineered =========\n")

    print(">>>>>>>>>>>> Making X and y... >>>>>>>>>>>>")
    X, y = F.make_xy(deliveries, cfg["target"])
    print("========= X and y made =========\n")

    print(">>>>>>>>>>>> Logging features... >>>>>>>>>>>>")
    art.logger.info(f"Processed {len(deliveries):,} rows | {X.shape[1]} features | positive_rate={y.mean():.4f}")
    art.table(pd.DataFrame(
        [(c, "numeric") for c in F.NUMERIC_FEATURES] + [(c, "binary") for c in F.BINARY_FEATURES] +
        [(c, "nominal") for c in F.NOMINAL_FEATURES] + [(c, "highcard") for c in F.HIGHCARD_FEATURES],
        columns=["feature", "group"]), "feature_inventory", index=False)
    print("========= Features logged =========\n")
    
    # 2) Split + validation scaffold
    print(">>>>>>>>>>>> Splitting data... >>>>>>>>>>>>")
    X_train, X_test, y_train, y_test, cv = _split(X, y, cfg, art)
    print("========= Data split =========\n")
    
    # 3) Multicollinearity (model-type-aware feature lists)
    print(">>>>>>>>>>>> Checking multicollinearity... >>>>>>>>>>>>")
    multicol = _multicollinearity(X_train, cfg, art)
    print("========= Multicollinearity checked =========\n")
    
    # 4) Candidate models (filter to subset if requested)
    print(">>>>>>>>>>>> Building candidate models... >>>>>>>>>>>>")
    candidates = T.build_candidate_models(cfg)
    if args.models:
        keep = {m.strip() for m in args.models.split(",")}
        candidates = {k: v for k, v in candidates.items() if k in keep}
    print("========= Candidate models built =========\n")
    
    # 5) CV baseline -> 6) tune -> 7) calibrate + threshold
    print(">>>>>>>>>>>> Running CV baseline... >>>>>>>>>>>>")
    _, baseline = _cv_baseline(candidates, X_train, y_train, cv, cfg, multicol, art)
    print("========= CV baseline run =========\n")
    
    print(">>>>>>>>>>>> Tuning models... >>>>>>>>>>>>")
    tuned, tuning = _tune(candidates, X_train, y_train, cfg, multicol, art)
    print("========= Models tuned =========\n")
    
    print(">>>>>>>>>>>> Calibrating models... >>>>>>>>>>>>")
    calibrated_models, thresholds, X_cal, y_cal = _calibrate(tuned, X_train, y_train, cfg, art)
    print("========= Models calibrated =========\n")
    
    # 8) Evaluate on test
    print(">>>>>>>>>>>> Evaluating on test... >>>>>>>>>>>>")
    test_metrics, proba, pred = E.evaluate_on_test(calibrated_models, thresholds, X_test, y_test, art)
    print("========= Models evaluated on test =========\n")
    
    # 9) Interpretability
    print(">>>>>>>>>>>> Calculating feature importance... >>>>>>>>>>>>")
    E.feature_importance(tuned, X_test, y_test, cfg, art)
    print("========= Feature importance calculated =========\n")
    
    if cfg.get("run_shap", True):
        print(">>>>>>>>>>>> Running SHAP... >>>>>>>>>>>>")
        E.run_shap(tuned, X_test, cfg, art)
        print("========= SHAP run =========\n")
    print("========= SHAP calculated =========\n")
    
    
    # 10) Selection, error & bias analysis
    print(">>>>>>>>>>>> Selecting best model... >>>>>>>>>>>>")
    best_name, _ = E.compare_and_select(test_metrics, thresholds, calibrated_models, cfg, art)
    print("========= Best model selected =========\n")

    print(">>>>>>>>>>>> Running error analysis... >>>>>>>>>>>>")
    art.logger.info(f"Best model by test PR-AUC = {best_name}")
    E.error_analysis(best_name, deliveries, X_test, y_test, proba, pred, SEG_COLS, art)
    print("========= Error analysis run =========\n")
    
    print(">>>>>>>>>>>> Running bias analysis... >>>>>>>>>>>>")
    E.bias_analysis(best_name, deliveries, X_test, y_test, pred, SEG_COLS, art)
    print("========= Bias analysis run =========\n")

    # 10b) OPTIONAL: per-segment (tier) thresholds to lift high-impact recall

    if cfg.get("per_segment_threshold", False):                    # config-driven toggle (no CLI flag)
        print(">>>>>>>> Running per-segment thresholds... >>>>>>>>")
        seg_col = cfg.get("segment_threshold_column", "delivery_priority")
        

        seg_thr = E.per_segment_thresholds(
            calibrated_models[best_name], X_cal, y_cal, deliveries.loc[X_cal.index, seg_col],
            thresholds[best_name], cfg["fn_fp_cost_ratio"], cfg.get("per_segment_min_positives", 30))
        E.segment_threshold_report(y_test, proba[best_name], deliveries.loc[X_test.index, seg_col],
                                   thresholds[best_name], seg_thr, best_name, art)
        print("====== Per-segment thresholds run ======\n")

    # 11) Business deliverable: risk x impact ranking
    print(">>>>>>>> Running risk x impact scores... >>>>>>>>")
    risk_impact_scores(calibrated_models[best_name], best_name, X_test, deliveries, cfg, art,
                       proba=proba[best_name])
    print("====== Risk x impact scores run ======\n")

    # 12) Final consolidated results table (all models) -> CSV + image
    print(">>>>>>>> Generating final results report... >>>>>>>>")
    E.final_results_report(baseline, tuning, test_metrics, best_name, art)
    print("====== Final results report generated ======\n")


def main():
    """Configure the run, execute the pipeline with error capture, and report total runtime.
    
    Parameters: none (reads CLI/config).
    
    Returns: None. On any unhandled error the full traceback is logged to main.log and re-raised so the process exits non-zero (CI fails correctly).
    """
    print("[main] starting pipeline  (imports OK)...", flush=True)
    args = parse_args()
    overrides = {"search_n_iter": args.search_n_iter, "cv_folds": args.cv_folds,
                 "test_size": args.test_size, "random_state": args.random_state}
    cfg = load_config(args.config, overrides)
    art = Artifacts(cfg)
    start_time = time.time()
    art.logger.info(f"Pipeline start | artifacts -> {cfg['paths']['artifact_dir']} | models -> {cfg['paths']['models_dir']}")
    try:
        run_pipeline(cfg, art, args)
    except Exception:
        art.logger.exception("!!!!!! Pipeline FAILED with an unhandled exception !!!!!!")
        raise
    elapsed = time.time() - start_time
    art.logger.info(f"Pipeline complete in {elapsed:.1f}s ({elapsed / 60:.2f} min).")
    print(f"\n============ TOTAL PIPELINE TIME: {elapsed:.1f} seconds ({elapsed / 60:.2f} minutes) ============")


if __name__ == "__main__":
    main()
