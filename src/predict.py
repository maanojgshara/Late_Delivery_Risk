"""Batch inference / serving entry point.

Loads the selected calibrated model (and its decision threshold + any per-segment thresholds)
and scores new pickup-time deliveries by risk x impact, writing a ranked dispatch queue. This is
the deployment counterpart of the training pipeline (src/main.py) - no retraining.
Run:  python -m src.predict --input data/delivery.db
"""
import os
import json
import argparse
import joblib
import pandas as pd

from .config import load_config
from .utils import get_logger
from .data_load import load_tables
from .preprocess import clean_data
from . import feature_engineer as F


def parse_args():
    """Parse CLI arguments for batch inference.
    Parameters: none (reads sys.argv).
    Returns: an argparse.Namespace (config, input, model, output, table).
    """
    p = argparse.ArgumentParser(description="Score new deliveries by risk x impact (inference).")
    p.add_argument("--config", default=None, help="path to config YAML")
    p.add_argument("--input", default=None, help="new data: a .csv file or SQLite .db (default: config db_path)")
    p.add_argument("--model", default=None, help="model .joblib (default: <models_dir>/final_model.joblib)")
    p.add_argument("--output", default="artifacts/tables/predictions.csv", help="output CSV path")
    p.add_argument("--table", default=None, help="table name if input is a SQLite db (default: config)")
    return p.parse_args()


def load_input(path, table, url, logger):
    """Load inference records from a CSV file or a SQLite database into a DataFrame.
    Parameters: path is the .csv/.db file; table is the SQLite table; url downloads the db if missing;
    logger for messages.
    Returns: a raw deliveries DataFrame.
    """
    if str(path).lower().endswith(".csv"):
        logger.info(f"Reading CSV input: {path}")
        return pd.read_csv(path)
    logger.info(f"Reading SQLite input: {path} (table={table})")
    return load_tables(path, url, logger)[table]


def load_thresholds(cfg, logger):
    """Read the selected model's global threshold and any saved per-segment thresholds.
    Parameters: cfg is the config; logger for messages.
    Returns: (global_threshold float, segment_thresholds dict or None).
    """
    runs = os.path.join(cfg["paths"]["artifact_dir"], "runs")
    gthr, seg = 0.5, None
    sel = os.path.join(runs, "model_selection.json")
    if os.path.exists(sel):
        gthr = float(json.load(open(sel)).get("threshold", 0.5))
    segp = os.path.join(runs, "segment_thresholds.json")
    if os.path.exists(segp):
        seg = json.load(open(segp)).get("segment_thresholds")
    logger.info(f"Decision threshold={gthr} | per-segment={'yes' if seg else 'no'}")
    return gthr, seg


def main():
    """Run batch inference: load model, engineer features, score risk x impact, save ranked queue.
    Parameters: none (reads CLI/config).
    Returns: None (writes predictions.csv + predictions_top20.csv under artifacts/).
    """
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("predict", os.path.join(cfg["paths"]["artifact_dir"], "logs"))
    model_path = args.model or os.path.join(cfg["paths"]["models_dir"], "final_model.joblib")
    input_path = args.input or cfg["data"]["db_path"]
    table = args.table or cfg["data"]["deliveries_table"]

    logger.info(f"Loading model: {model_path}")
    model = joblib.load(model_path)

    raw = load_input(input_path, table, cfg["data"]["url"], logger)
    df = F.engineer_features(clean_data(raw))
    X = F.make_x(df)

    proba = model.predict_proba(X)[:, 1]
    impact = F.compute_impact_weight(df, tuple(cfg["impact_weights"]))
    gthr, seg_thr = load_thresholds(cfg, logger)

    out = pd.DataFrame({
        "delivery_id": df.get("delivery_id"),
        "branch": df.get("branch"),
        "delivery_priority": df.get("delivery_priority"),
        "parcel_category": df.get("parcel_category"),
        "parcel_value_sgd": df.get("parcel_value_sgd"),
        "risk": proba.round(4),
        "impact": impact.round(4).values,
    })
    out["risk_x_impact"] = (out["risk"] * out["impact"]).round(4)
    if seg_thr and "delivery_priority" in df.columns:
        thr_row = df["delivery_priority"].astype(str).map(lambda s: seg_thr.get(s, gthr)).fillna(gthr).values
        out["at_risk"] = (proba >= thr_row).astype(int)
        logger.info("Applied per-segment thresholds for the at-risk flag.")
    else:
        out["at_risk"] = (proba >= gthr).astype(int)
    out = out.sort_values("risk_x_impact", ascending=False)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    top_path = args.output.replace(".csv", "_top20.csv")
    out.head(20).to_csv(top_path, index=False)
    logger.info(f"Scored {len(out):,} deliveries | flagged at-risk: {int(out['at_risk'].sum())} "
                f"-> {args.output}")
    print(f"Predictions: {args.output}\nTop-20 dispatch queue: {top_path}")


if __name__ == "__main__":
    main()
