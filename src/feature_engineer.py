"""Feature engineering, feature groups and the deterministic impact weight.

Reproduces the EDA's leakage-safe, pickup-time feature set (label `is_late` derived only
from delivery time), defines the model-input feature groups, and builds the business impact
weight used in the risk x impact score.
"""
import numpy as np
import pandas as pd

# --- Feature groups (pickup-time only; leakage fields excluded). ---
NUMERIC_FEATURES = [
    "distance_km", "parcel_weight_kg", "parcel_value_sgd", "num_stops_on_route",
    "driver_experience_months", "delivery_priority_rank", "parcel_handling_sensitivity",
    "pickup_hour", "pickup_month", "pickup_day",
    "stops_x_distance", "exp_per_stop", "value_per_kg", "route_load_per_exp",
]
BINARY_FEATURES = [
    "is_high_risk_handling", "is_standard_priority", "pickup_is_peakhour",
    # Missing-indicator flags retained per EDA Section 9.
    "has_num_stops_on_route", "has_vehicle_type", "has_payment_method",
    "has_parcel_value_sgd", "has_recipient_postal_code",
    "has_parcel_weight_kg", "has_driver_experience_months",
]
# Near-null categoricals (vehicle_type, payment_method, pickup_is_weekend) dropped per EDA Sec 12/14.
NOMINAL_FEATURES = ["branch", "parcel_category", "delivery_priority", "promised_service_tier"]
HIGHCARD_FEATURES = ["recipient_postal_code"]          # target/mean encoding (EDA Sec 14)
SKEWED_FOR_LOG = ["parcel_value_sgd", "parcel_weight_kg", "distance_km"]   # log1p (linear only)
BIN_SOURCES = ["distance_km", "parcel_value_sgd", "driver_experience_months"]  # bins (linear only)
WINSOR_COLS = ["parcel_value_sgd", "parcel_weight_kg", "distance_km",
               "num_stops_on_route", "driver_experience_months"]            # winsorise (linear only)
# Reserved for the risk x impact weight, NOT the risk model (no lateness signal; EDA Sec 7/14).
IMPACT_FEATURES = ["client_id_frequency", "delivery_priority_rank", "parcel_value_sgd"]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + NOMINAL_FEATURES + HIGHCARD_FEATURES


def engineer_features(df):
    """Build the binary target plus the pickup-time features finalised in the EDA.

    Creates calendar parts, ordinal ranks, promised-service tier, interaction/ratio/flag
    features, the has_* missing-indicators and client_id_frequency (kept for the impact
    weight). All features are row-local (no cross-row leakage) except client_id_frequency.
    Parameters: df is the cleaned deliveries DataFrame.
    Returns: a copy with engineered columns and the `is_late` label added.
    """
    out = df.copy()
    af = lambda s: pd.to_numeric(s, errors="coerce").astype("float64")

    if "delivery_datetime" in out.columns:                   # label only exists for historical/training data
        lateness = (out["delivery_datetime"] - out["promised_delivery_datetime"]).dt.total_seconds() / 60
        out["is_late"] = (lateness > 0).astype("int8")       # LABEL (uses delivery time) - not a feature

    ts = out["pickup_datetime"]
    out["pickup_hour"] = ts.dt.hour.astype("Int16")
    out["pickup_month"] = ts.dt.month.astype("Int16")
    out["pickup_day"] = ts.dt.dayofweek.astype("Int16")
    out["pickup_is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype("Int8")
    h = ts.dt.hour
    out["pickup_is_peakhour"] = (((h >= 7) & (h <= 9)) | ((h >= 17) & (h <= 20))).astype("Int8")

    out["delivery_priority_rank"] = out["delivery_priority"].map(
        {"standard": 1, "express": 2, "premium": 3, "vip": 4}).astype("Int16")
    out["parcel_handling_sensitivity"] = out["parcel_category"].map(
        {"standard": 0, "oversized": 1, "fragile": 2, "refrigerated": 3}).astype("Int16")

    win = (out["promised_delivery_datetime"] - out["pickup_datetime"]).dt.total_seconds() / 60
    same_day = out["promised_delivery_datetime"].dt.normalize() == out["pickup_datetime"].dt.normalize()
    tier = pd.Series(pd.NA, index=out.index, dtype="object")
    tier[win <= 120] = "2_hour"
    tier[(win > 120) & same_day] = "same_day"
    tier[(win > 120) & ~same_day] = "next_day"
    out["promised_service_tier"] = tier.astype("string")

    freq = out["client_id"].value_counts() / len(out)        # impact axis; refit on train in production
    out["client_id_frequency"] = out["client_id"].map(freq)

    out["stops_x_distance"] = af(out["num_stops_on_route"]) * af(out["distance_km"])
    # +1 / +0.1 in the ratio denominators guard against division by zero
    out["exp_per_stop"] = af(out["driver_experience_months"]) / (af(out["num_stops_on_route"]) + 1)
    out["value_per_kg"] = af(out["parcel_value_sgd"]) / (af(out["parcel_weight_kg"]) + 0.1)
    out["route_load_per_exp"] = af(out["num_stops_on_route"]) / (af(out["driver_experience_months"]) + 1)
    out["is_high_risk_handling"] = out["parcel_category"].isin(["refrigerated", "oversized"]).astype("int8")
    out["is_standard_priority"] = (out["delivery_priority"] == "standard").astype("int8")

    for col in ["num_stops_on_route", "vehicle_type", "payment_method", "parcel_value_sgd",
                "recipient_postal_code", "parcel_weight_kg", "driver_experience_months"]:
        out[f"has_{col}"] = out[col].notna().astype("int8")
    return out


def make_x(df):
    """Build the typed candidate feature matrix X (no label) for training or inference.

    Numerics -> float (NaNs kept for in-pipeline imputation); binary flags -> int8;
    categoricals -> object with real np.nan (sklearn-safe; avoids pandas pd.NA).
    Parameters: df is the engineered DataFrame.
    Returns: X with the ALL_FEATURES columns, correctly typed.
    """
    X = df[ALL_FEATURES].copy()
    for c in NUMERIC_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    for c in BINARY_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype("int8")
    for c in NOMINAL_FEATURES + HIGHCARD_FEATURES:
        col = X[c].astype(object)
        X[c] = col.where(col.notna(), np.nan)
    return X


def make_xy(df, target):
    """Assemble the typed feature matrix X and the integer target y for modelling.
    Parameters: df is the engineered DataFrame; target is the label column name.
    Returns: (X, y) where X = make_x(df) and y is the integer label Series.
    """
    return make_x(df), df[target].astype(int)


def minmax(s):
    """Min-max scale a numeric series to [0, 1] (NaNs filled with the median first).
    Parameters: s is the input numeric Series.
    Returns: a Series scaled to [0, 1] (constant input maps to 0).
    """
    s = pd.to_numeric(s, errors="coerce"); s = s.fillna(s.median())
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0.0


def compute_impact_weight(df, weights=(0.4, 0.4, 0.2)):
    """Build a deterministic [0,1] impact weight from client importance, priority and value.

    Higher client frequency, higher priority tier and higher (log) parcel value imply a more
    costly delivery if it is late.
    Parameters: df holds IMPACT_FEATURES; weights are the (client, priority, value) blend.
    Returns: a Series of impact weights in [0, 1] aligned with df.
    """
    client = minmax(df["client_id_frequency"])
    priority = minmax(df["delivery_priority_rank"])
    value = minmax(np.log1p(pd.to_numeric(df["parcel_value_sgd"], errors="coerce").clip(lower=0)))
    wc, wp, wv = weights
    return (wc * client + wp * priority + wv * value).clip(0, 1)
