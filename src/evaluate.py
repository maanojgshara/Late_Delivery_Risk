"""Evaluation, interpretability, error analysis, bias slices and model comparison.

Operates on the held-out test set with calibrated models at their cost-aware thresholds:
metrics + confusion matrices + ROC/PR curves, permutation importance, optional SHAP,
misclassification/error slices, per-group fairness metrics, and the final model selection.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (average_precision_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, precision_recall_curve)

from .utils import tqdm, label_bars
from .train import metrics_at, choose_threshold_cost

COLORS = {"logreg": "#4C72B0", "random_forest": "#C44E52", "hist_gbm": "#55A868"}


def evaluate_on_test(calibrated_models, thresholds, X_test, y_test, art):
    """Evaluate calibrated models on the test set; save metrics, confusion matrices, ROC/PR.

    Parameters: calibrated_models is {name: model}; thresholds is {name: thr}; X_test/y_test the
    held-out data; art is the Artifacts saver.
    Returns: (test_metrics_df, test_proba_dict, test_pred_dict).
    """
    rows, proba, pred = [], {}, {}
    for name, cal in tqdm(calibrated_models.items(), total=len(calibrated_models), desc="Evaluation"):
        p = cal.predict_proba(X_test)[:, 1]; thr = thresholds[name]
        proba[name] = p; pred[name] = (p >= thr).astype(int)
        m = metrics_at(y_test, p, thr); m["model"] = name; m["threshold"] = round(thr, 3)
        rows.append(m)
        art.logger.info(f"[TEST] {name}: PR-AUC={m['PR_AUC']:.4f} recall={m['recall']:.3f} "
                        f"precision={m['precision']:.3f} f1={m['f1']:.3f} brier={m['brier']:.4f}")
    test_metrics = pd.DataFrame(rows).set_index("model").round(4)
    art.table(test_metrics, "test_metrics")

    fig, axes = plt.subplots(1, len(calibrated_models), figsize=(5 * len(calibrated_models), 4))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, calibrated_models):
        cm = confusion_matrix(y_test, pred[name])
        ConfusionMatrixDisplay(cm, display_labels=["on-time", "late"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name} @thr={thresholds[name]:.2f}")
    plt.tight_layout(); art.fig(fig, "test_confusion_matrices"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, p in proba.items():
        fpr, tpr, _ = roc_curve(y_test, p); prec, rec, _ = precision_recall_curve(y_test, p)
        ax[0].plot(fpr, tpr, color=COLORS.get(name), label=f"{name} (AUC={roc_auc_score(y_test, p):.3f})")
        ax[1].plot(rec, prec, color=COLORS.get(name), label=f"{name} (AP={average_precision_score(y_test, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="grey"); ax[0].set_title("ROC (test)")
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend(fontsize=8)
    ax[1].axhline(y_test.mean(), ls="--", color="grey", label=f"base {y_test.mean():.3f}")
    ax[1].set_title("Precision-Recall (test)"); ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
    ax[1].legend(fontsize=8)
    plt.tight_layout(); art.fig(fig, "roc_pr_curves"); plt.close(fig)
    return test_metrics, proba, pred


def permutation_importance_bar(estimator, X, y, n_repeats, random_state, desc):
    """Permutation importance (mean PR-AUC drop) with a per-feature 0-100% progress bar.

    Each feature is shuffled n_repeats times; the mean drop in average-precision is recorded.
    Parameters: estimator (fitted), X/y eval sample, n_repeats, random_state, desc bar label.
    Returns: a pandas Series of mean importance per feature (higher = more important).
    """
    rng = np.random.RandomState(random_state); X = X.copy()
    base = average_precision_score(y, estimator.predict_proba(X)[:, 1]); out = {}
    for col in tqdm(list(X.columns), total=X.shape[1], desc=desc, leave=True):
        saved = X[col].copy(); drops = []
        for _ in range(n_repeats):
            X[col] = rng.permutation(saved.values)
            drops.append(base - average_precision_score(y, estimator.predict_proba(X)[:, 1]))
        X[col] = saved; out[col] = float(np.mean(drops))
    return pd.Series(out)


def feature_importance(tuned, X_test, y_test, cfg, art):
    """Compute permutation importance for every model and save bars + a merged table.

    Parameters: tuned is {name: fitted_pipeline}; X_test/y_test the data; cfg the config; art the saver.
    Returns: the merged importance DataFrame (features x models + mean).
    """
    idx = X_test.sample(min(cfg["perm_sample"], len(X_test)), random_state=cfg["random_state"]).index
    Xs, ys = X_test.loc[idx], y_test.loc[idx]
    tables = {}
    fig, axes = plt.subplots(1, len(tuned), figsize=(6 * len(tuned), 5)); axes = np.atleast_1d(axes)
    for ax, (name, pipe) in zip(axes, tuned.items()):
        imp = permutation_importance_bar(pipe, Xs, ys, cfg["perm_repeats"],
                                         cfg["random_state"], f"{name} perm").sort_values(ascending=False)
        tables[name] = imp
        imp.head(12).iloc[::-1].plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_title(f"{name}: top-12 permutation importance"); ax.set_xlabel("delta PR-AUC")
        label_bars(ax, fmt="{:.3f}", horizontal=True, fontsize=7)
    plt.tight_layout(); art.fig(fig, "permutation_importance"); plt.close(fig)
    merged = pd.DataFrame(tables).round(5)
    merged["mean_across_models"] = merged.mean(axis=1)
    merged = merged.sort_values("mean_across_models", ascending=False)
    art.table(merged, "feature_importance_merged")
    art.logger.info(f"Permutation importance done; top feature = {merged.index[0]}")
    return merged


def run_shap(tuned, X_test, cfg, art):
    """Compute SHAP summaries + top-feature tables per model (optional; degrades gracefully).

    Parameters: tuned is {name: fitted_pipeline}; X_test the data; cfg the config; art the saver.
    Returns: dict {name: mean_abs_shap Series}, or empty if shap is unavailable.
    """
    try:
        import shap
    except Exception:
        art.logger.info("shap not installed -> skipping SHAP; see permutation importance.")
        return {}
    idx = X_test.sample(min(cfg["shap_sample"], len(X_test)), random_state=cfg["random_state"]).index
    top = {}
    for name, pipe in tqdm(tuned.items(), total=len(tuned), desc="SHAP"):
        try:
            pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
            Xenc = pre.transform(X_test.loc[idx]); names = pre.get_feature_names_out()
            if name == "logreg":
                sv = shap.LinearExplainer(clf, Xenc).shap_values(Xenc)
            else:
                sv = shap.TreeExplainer(clf).shap_values(Xenc)
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) == 2 else sv[0]
                # newer shap returns (n_samples, n_features, n_classes); keep the positive-class slice
                elif getattr(sv, "ndim", 2) == 3:
                    sv = sv[:, :, 1]
            mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
            top[name] = mean_abs
            fig = plt.figure()
            shap.summary_plot(sv, features=Xenc, feature_names=list(names),
                              plot_type="bar", show=False, max_display=15)
            art.fig(plt.gcf(), f"shap_{name}", subdir=name); plt.close(fig)
            art.table(mean_abs.round(5).rename("mean_abs_shap"), f"shap_top_{name}", subdir=name)
        except Exception as e:
            art.logger.warning(f"SHAP failed for {name}: {e}")
    if top:
        art.table(pd.DataFrame(top).round(5), "shap_merged")
    return top


def _error_slice(df, seg):
    """Summarise FN/FP/error rates within each level of a segment column.
    Parameters: df is the per-row error frame; seg is the segment column.
    Returns: a per-level DataFrame sorted by FN-rate descending.
    """
    g = df.groupby(seg, dropna=False)
    out = pd.DataFrame({
        "n": g.size(),
        "FN_rate": g.apply(lambda d: (d.error_type == "FN").sum() / max(1, (d.y_true == 1).sum())),
        "FP_rate": g.apply(lambda d: (d.error_type == "FP").sum() / max(1, (d.y_true == 0).sum())),
        "error_rate": g.apply(lambda d: (d.error_type != "correct").mean()),
    }).round(4)
    return out.sort_values("FN_rate", ascending=False)


def error_analysis(best_name, deliveries, X_test, y_test, proba, pred, seg_cols, art):
    """Export misclassifications, segment error slices and the top failure modes for the best model.

    Parameters: best_name; deliveries (for ids/segments); X_test/y_test; proba/pred dicts; seg_cols;
    art the saver.
    Returns: the per-row error DataFrame.
    """
    err = deliveries.loc[X_test.index, ["delivery_id"] + seg_cols].copy()
    err["y_true"] = y_test.values; err["proba"] = proba[best_name]; err["y_pred"] = pred[best_name]
    err["error_type"] = np.select(
        [(err.y_true == 1) & (err.y_pred == 0), (err.y_true == 0) & (err.y_pred == 1)],
        ["FN", "FP"], default="correct")
    art.table(err[err.error_type != "correct"].sort_values("proba", ascending=False),
              "misclassified", index=False)
    modes = []
    for seg in seg_cols:
        t = _error_slice(err, seg); art.table(t, f"error_slice_{seg}")
        for lvl, row in t[t["n"] >= 200].head(2).iterrows():
            modes.append({"segment": seg, "level": lvl, "n": int(row["n"]), "FN_rate": row["FN_rate"]})
    art.table(pd.DataFrame(modes).sort_values("FN_rate", ascending=False),
              "top_failure_modes", index=False)
    return err


def slice_metrics(y_true, y_pred, seg):
    """Recall, Precision, FPR and FNR for each level of a segment series.
    Parameters: y_true/y_pred aligned arrays; seg the segment Series.
    Returns: a DataFrame indexed by segment level with n, recall, precision, FPR, FNR.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred), "seg": np.asarray(seg)})
    rows = {}
    for lvl, d in df.groupby("seg", dropna=False):
        tp = int(((d.p == 1) & (d.y == 1)).sum()); fn = int(((d.p == 0) & (d.y == 1)).sum())
        fp = int(((d.p == 1) & (d.y == 0)).sum()); tn = int(((d.p == 0) & (d.y == 0)).sum())
        rows[lvl] = {"n": len(d), "recall": tp / max(1, tp + fn), "precision": tp / max(1, tp + fp),
                     "FPR": fp / max(1, fp + tn), "FNR": fn / max(1, fn + tp)}
    return pd.DataFrame(rows).T.round(4)


def bias_analysis(best_name, deliveries, X_test, y_test, pred, seg_cols, art):
    """Per-group fairness metrics for the best model across segments; save tables + heatmap.

    Parameters: best_name; deliveries; X_test/y_test; pred dict; seg_cols; art the saver.
    Returns: dict {segment: slice-metrics DataFrame}.
    """
    tables = {}
    fig, axes = plt.subplots(1, len(seg_cols), figsize=(4 * len(seg_cols), 4)); axes = np.atleast_1d(axes)
    for ax, seg in zip(axes, seg_cols):
        t = slice_metrics(y_test, pred[best_name], deliveries.loc[X_test.index, seg])
        tables[seg] = t; art.table(t, f"slice_metrics_{seg}")
        worst = t.sort_values("recall").index[0]
        art.logger.info(f"[bias] {seg}: lowest recall group = {worst} ({t.loc[worst, 'recall']:.3f})")
        sns.heatmap(t[["recall", "precision", "FPR", "FNR"]], annot=True, fmt=".2f",
                    cmap="RdYlGn", center=0.5, ax=ax, cbar=False); ax.set_title(seg)
    plt.tight_layout(); art.fig(fig, "slice_metrics_heatmap"); plt.close(fig)
    return tables


def compare_and_select(test_metrics, thresholds, calibrated_models, cfg, art):
    """Compare models, select the best (PR-AUC, then recall, then Brier) and save the final model.

    Parameters: test_metrics DataFrame; thresholds dict; calibrated_models dict; cfg; art the saver.
    Returns: (best_name, selection_dict).
    """
    cols = ["PR_AUC", "ROC_AUC", "recall", "precision", "f1", "balanced_acc", "brier", "threshold"]
    compare = test_metrics[cols].copy(); art.table(compare, "model_comparison")
    fig, ax = plt.subplots(figsize=(9, 5))
    compare[["PR_AUC", "recall", "f1", "balanced_acc"]].plot(kind="bar", ax=ax)
    ax.set_title("Test metrics by model"); ax.set_ylabel("score"); ax.legend(fontsize=8)
    label_bars(ax, fmt="{:.2f}", fontsize=6)
    plt.tight_layout(); art.fig(fig, "model_comparison_bar"); plt.close(fig)
    ranked = compare.sort_values(["PR_AUC", "recall", "brier"], ascending=[False, False, True])
    best_name = ranked.index[0]
    selection = {"chosen_model": best_name,
                 "reason": "highest test PR-AUC; recall and Brier tie-breakers given FN cost + calibration",
                 "threshold": float(thresholds[best_name]),
                 "threshold_policy": f"cost-aware (FN:FP = {cfg['fn_fp_cost_ratio']}:1); "
                                     "primary use is ranking by risk x impact"}
    art.run(selection, "model_selection")
    art.model(calibrated_models[best_name], "final_model")
    return best_name, selection


def final_results_report(baseline, tuning, test_metrics, best_name, art):
    """Consolidate every model's CV and test metrics into one final table; save CSV + image.

    Combines OOF baseline PR-AUC, tuned CV PR-AUC and all test metrics (+ threshold, selected flag)
    so reviewers can compare every model and value at a glance.
    Parameters: baseline/tuning/test_metrics are per-model DataFrames; best_name; art the saver.
    Returns: the final results DataFrame.
    """
    df = test_metrics.copy()
    df.insert(0, "tuned_cv_PR_AUC", tuning["best_cv_PR_AUC"].reindex(df.index))
    df.insert(0, "cv_oof_PR_AUC", baseline["PR_AUC"].reindex(df.index).round(4))
    df["selected"] = [ix == best_name for ix in df.index]
    art.table(df, "final_results")

    show = df.round(4).reset_index()
    fig, ax = plt.subplots(figsize=(min(2 + 1.6 * len(show.columns), 22), 1.4 + 0.5 * len(show)))
    ax.axis("off")
    tbl = ax.table(cellText=show.values, colLabels=list(show.columns), loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.5)
    for j, col in enumerate(show.columns):                       # header styling
        tbl[0, j].set_facecolor("#4C72B0"); tbl[0, j].set_text_props(color="white", weight="bold")
    sel_row = show.index[show["selected"]].tolist()
    for r in sel_row:                                            # highlight selected model row
        for j in range(len(show.columns)):
            tbl[r + 1, j].set_facecolor("#E8F0E8")
    ax.set_title("Final results - all models (CV + test metrics)", pad=14, weight="bold")
    art.fig(fig, "final_results"); plt.close(fig)
    art.logger.info(f"Final results table saved (selected model: {best_name}).")
    return df


def per_segment_thresholds(model, X_cal, y_cal, seg_cal, global_thr, fn_fp_ratio, min_positives=30):
    """Tune a cost-aware decision threshold per segment level on the calibration slice.
    Each tuned threshold is CAPPED at the global threshold, so a segment can only LOWER its cutoff
    (lift recall) and never raise it (the policy can never reduce a tier's recall). Segments with
    fewer than `min_positives` late deliveries fall back to the global threshold (too noisy to tune).
    Parameters: model (calibrated); X_cal/y_cal calibration slice; seg_cal segment labels; global_thr
    cap/fallback; fn_fp_ratio FN:FP cost ratio; min_positives minimum positives to tune a segment.
    Returns: dict {segment_level: threshold} with every value <= global_thr.
    """
    p = model.predict_proba(X_cal)[:, 1]; y = np.asarray(y_cal); seg = np.asarray(seg_cal.astype(str))
    out = {}
    for lvl in pd.unique(seg):
        m = seg == lvl
        if int(y[m].sum()) >= min_positives:
            # cap at the global threshold: a tier can only LOWER its cutoff (lift recall), never raise it
            out[lvl] = min(choose_threshold_cost(y[m], p[m], fn_fp_ratio), float(global_thr))
        else:
            out[lvl] = float(global_thr)
    return out


def apply_segment_thresholds(proba, seg, seg_thr, global_thr):
    """Apply each row's segment-specific threshold to its probability to get binary predictions.
    Parameters: proba is P(late); seg are segment labels; seg_thr maps level->threshold; global_thr
    is the fallback for unseen levels.
    Returns: an integer prediction array (1 = flagged at-risk).
    """
    seg = np.asarray(seg.astype(str)) if hasattr(seg, "astype") else np.asarray(seg)
    thr = np.array([seg_thr.get(s, global_thr) for s in seg])
    return (np.asarray(proba) >= thr).astype(int)


def segment_threshold_report(y_test, proba_best, seg_test, global_thr, seg_thr, best_name, art):
    """Compare per-segment vs global thresholds (recall/precision/FNR per tier); save table+figure+json.
    Shows the recall lift and precision cost of the per-tier policy so the trade-off is auditable.
    Parameters: y_test; proba_best test P(late); seg_test segment labels; global_thr; seg_thr dict;
    best_name; art saver.
    Returns: the comparison DataFrame.
    """
    pred_global = (np.asarray(proba_best) >= global_thr).astype(int)
    pred_seg = apply_segment_thresholds(proba_best, seg_test, seg_thr, global_thr)
    g = slice_metrics(y_test, pred_global, seg_test)[["n", "recall", "precision", "FNR"]].add_suffix("_global")
    s = slice_metrics(y_test, pred_seg, seg_test)[["recall", "precision", "FNR"]].add_suffix("_segment")
    comp = pd.concat([g, s], axis=1)
    comp["threshold"] = [round(seg_thr.get(str(i), global_thr), 3) for i in comp.index]
    comp["recall_lift"] = (comp["recall_segment"] - comp["recall_global"]).round(4)
    art.table(comp, "segment_threshold_comparison")
    art.run({"model": best_name, "global_threshold": float(global_thr),
             "segment_thresholds": {str(k): float(v) for k, v in seg_thr.items()}}, "segment_thresholds")

    idx = np.arange(len(comp)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(idx - w / 2, comp["recall_global"], w, label="recall (global thr)", color="#4C72B0")
    ax.bar(idx + w / 2, comp["recall_segment"], w, label="recall (per-segment thr)", color="#55A868")
    ax.set_xticks(idx); ax.set_xticklabels(comp.index, rotation=10); ax.set_ylabel("recall")
    ax.set_title(f"Per-segment threshold recall lift - {best_name}"); ax.legend(fontsize=8)
    label_bars(ax, fmt="{:.2f}", fontsize=7)
    plt.tight_layout(); art.fig(fig, "segment_threshold_comparison"); plt.close(fig)
    art.logger.info(f"Per-segment thresholds ({best_name}): {seg_thr}")
    return comp
