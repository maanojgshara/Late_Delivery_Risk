"""Cleaning, multicollinearity diagnosis and model-type-aware preprocessors.

Mirrors the EDA: standardise/clean the raw table, diagnose multicollinearity (Spearman + VIF)
on training data, and build two ColumnTransformers - a lean tree preprocessor and a linear
preprocessor that additionally winsorises, log-transforms, scales and bins (linear-only).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder, RobustScaler, FunctionTransformer,
                                    KBinsDiscretizer)
from sklearn.base import BaseEstimator, TransformerMixin

from . import feature_engineer as F

try:
    from sklearn.preprocessing import TargetEncoder
    HAVE_TE = True
except Exception:                                   # sklearn < 1.3
    HAVE_TE = False


def clean_data(deliveries):
    """Standardise the raw deliveries table (EDA Sections 1-2) before feature engineering.

    Lowercases/fixes category typos, casts datetimes/numerics/ids, nulls invalid -1 sentinels
    and drops exact duplicate rows.
    Parameters: deliveries is the raw DataFrame from the SQLite 'deliveries' table.
    Returns: a cleaned, de-duplicated copy with corrected dtypes (index reset).
    """
    df = deliveries.copy()
    has = lambda c: c in df.columns                          # guard: inference data may omit outcome fields
    for c in ["branch", "parcel_category", "delivery_priority", "vehicle_type"]:
        if has(c):
            df[c] = df[c].astype("string").str.strip().str.lower()
    if has("branch"):
        df["branch"] = df["branch"].replace({"cnetral": "central", "noth": "north"})
    if has("parcel_category"):
        df["parcel_category"] = df["parcel_category"].replace(
            {"refrig.": "refrigerated", "over-sized": "oversized"})
    for c in ["booking_datetime", "pickup_datetime", "promised_delivery_datetime", "delivery_datetime"]:
        if has(c):
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.round("s")
    for c in ["driver_experience_months", "num_stops_on_route"]:
        if has(c):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["parcel_weight_kg", "parcel_value_sgd"]:
        if has(c):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["delivery_id", "client_id", "driver_id", "payment_method", "recipient_postal_code"]:
        if has(c):
            df[c] = df[c].astype("string")
    for c in ["parcel_weight_kg", "parcel_value_sgd", "num_stops_on_route", "driver_experience_months"]:
        if has(c):
            df[c] = df[c].replace(-1.0, pd.NA)
    return df.drop_duplicates().reset_index(drop=True)


def vif_table(Xnum):
    """Variance Inflation Factor per numeric feature via R^2 (no statsmodels needed).

    Each column is regressed on the others (median-imputed, standardised); VIF = 1/(1-R^2).
    Parameters: Xnum is a DataFrame of numeric feature columns.
    Returns: a Series of VIF values indexed by feature, sorted descending.
    """
    Z = Xnum.apply(lambda s: s.fillna(s.median())).values.astype(float)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    vifs = {}
    for i, col in enumerate(Xnum.columns):
        yi = Z[:, i]; Xo = np.delete(Z, i, axis=1)
        beta, *_ = np.linalg.lstsq(Xo, yi, rcond=None)
        r2 = 1 - ((yi - Xo @ beta) ** 2).sum() / ((yi - yi.mean()) ** 2).sum()
        # VIF = 1/(1-R^2): how strongly this feature is explained by the other numerics
        vifs[col] = 1.0 / max(1e-9, 1 - r2)
    return pd.Series(vifs).sort_values(ascending=False)


def compute_multicollinearity(X_train, cfg):
    """Diagnose collinearity on training numerics and derive the linear-model drop list.

    Flags |Spearman| >= corr_threshold pairs and VIF > vif_threshold features; the linear
    model drops these, the tree model keeps everything (it is collinearity-invariant).
    Parameters: X_train is the training feature frame; cfg holds the thresholds.
    Returns: dict with spearman, vif, pairs, linear_drop, linear_numeric, tree_numeric.
    """
    num = F.NUMERIC_FEATURES
    num_train = X_train[num]
    spearman = num_train.corr(method="spearman")
    vif = vif_table(num_train).round(2)
    pairs = []
    for i in range(len(num)):
        for j in range(i + 1, len(num)):
            r = spearman.iloc[i, j]
            if abs(r) >= cfg["corr_threshold"]:
                pairs.append({"feat_a": num[i], "feat_b": num[j], "spearman": round(r, 3)})
    linear_drop = [c for c in vif.index if vif[c] > cfg["vif_threshold"]]
    for p in pairs:
        # of a correlated pair, drop the higher-VIF member (keeps the more independent feature)
        worse = p["feat_a"] if vif.get(p["feat_a"], 0) >= vif.get(p["feat_b"], 0) else p["feat_b"]
        if worse not in linear_drop:
            linear_drop.append(worse)
    linear_numeric = [c for c in num if c not in linear_drop]
    return {"spearman": spearman, "vif": vif, "pairs": pd.DataFrame(pairs),
            "linear_drop": linear_drop, "linear_numeric": linear_numeric,
            "tree_numeric": list(num)}


def _safe_log1p(a):
    """Numerically safe log1p that clips negatives to zero before transforming.
    Parameters: a is an array-like of numeric values.
    Returns: an array of log1p(clip(a, 0, None)) values.
    """
    return np.log1p(np.clip(a, 0, None))


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Winsorise each column at train-fitted low/high percentiles (EDA Section 10).

    Bounds are learned on the training fold only and reused on validation/test, capping
    legitimate extreme values so they do not dominate the linear fit.
    Parameters: low/high are the lower/upper percentile fractions (default 0.5% / 99.5%).
    Returns (transform): the input array clipped element-wise to the fitted bounds.
    """
    def __init__(self, low=0.005, high=0.995):
        """Store the winsorisation percentile bounds for later fitting.
        Parameters: low/high are the lower/upper percentile fractions to clip at.
        Returns: None (constructor)."""
        self.low = low; self.high = high

    def fit(self, X, y=None):
        """Learn per-column clip bounds from training data.
        Parameters: X is the numeric array/DataFrame; y is ignored.
        Returns: self, with low_/high_ percentile bounds stored."""
        Xv = np.asarray(X, dtype=float)
        self.low_ = np.nanpercentile(Xv, self.low * 100, axis=0)
        self.high_ = np.nanpercentile(Xv, self.high * 100, axis=0)
        return self

    def transform(self, X):
        """Clip the input to the fitted [low_, high_] bounds.
        Parameters: X is the numeric array/DataFrame.
        Returns: a float array clipped column-wise."""
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        """Pass feature names through unchanged (one-to-one transform).
        Parameters: input_features is the incoming column-name array.
        Returns: the same names as a numpy array."""
        return np.asarray(input_features)


def _make_nominal_encoder(cfg):
    """One-hot encoder for low-cardinality nominal features (rare levels collapsed; unseen-safe).
    Parameters: cfg supplies ohe_min_frequency.
    Returns: a configured OneHotEncoder (dense)."""
    return OneHotEncoder(handle_unknown="infrequent_if_exist",
                         min_frequency=cfg["ohe_min_frequency"], sparse_output=False)


def _make_highcard_encoder(cfg):
    """Encoder for the high-cardinality postal code: cross-fitted TargetEncoder, else OHE fallback.
    Parameters: cfg supplies random_state / ohe_min_frequency.
    Returns: a TargetEncoder or OneHotEncoder instance."""
    if HAVE_TE:
        return TargetEncoder(target_type="binary", random_state=cfg["random_state"])
    return OneHotEncoder(handle_unknown="infrequent_if_exist",
                         min_frequency=cfg["ohe_min_frequency"], sparse_output=False)


def _nominal_pipe(cfg):
    """Mode-impute then one-hot encode nominal features (EDA Sec 9 imputation).
    Parameters: cfg for the encoder.
    Returns: a Pipeline (impute -> one-hot)."""
    return Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                     ("ohe", _make_nominal_encoder(cfg))])


def _highcard_pipe(cfg):
    """Mode-impute then target/mean-encode the high-cardinality feature.
    Parameters: cfg for the encoder.
    Returns: a Pipeline (impute -> target/one-hot encode)."""
    return Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                     ("enc", _make_highcard_encoder(cfg))])


def build_tree_preprocessor(cfg, tree_numeric):
    """Tree-model ColumnTransformer: median-impute numerics, pass flags, encode categoricals.

    No scaling/winsorising/binning and no feature removal - tree ensembles are scale- and
    collinearity-invariant.
    Parameters: cfg is the config; tree_numeric is the numeric feature list.
    Returns: an unfitted ColumnTransformer for tree models.
    """
    return ColumnTransformer([
        ("num",  SimpleImputer(strategy="median"), tree_numeric),
        ("bin",  "passthrough",                    F.BINARY_FEATURES),
        ("nom",  _nominal_pipe(cfg),               F.NOMINAL_FEATURES),
        ("high", _highcard_pipe(cfg),              F.HIGHCARD_FEATURES),
    ])


def build_linear_preprocessor(cfg, linear_numeric):
    """Linear-model ColumnTransformer (model-type-aware preprocessing).

    Numerics: median-impute -> winsorise(0.5/99.5) -> log1p (skewed only) -> RobustScaler; plus
    quantile bins for threshold effects; categoricals encoded as for the tree model.
    Parameters: cfg is the config; linear_numeric is the post-VIF numeric feature list.
    Returns: an unfitted ColumnTransformer for the linear model.
    """
    log_num = [c for c in linear_numeric if c in F.SKEWED_FOR_LOG]
    plain_num = [c for c in linear_numeric if c not in F.SKEWED_FOR_LOG]
    bin_cols = [c for c in F.BIN_SOURCES if c in linear_numeric]
    # linear-only chain: impute -> winsorise (cap tails) -> log1p (skewed) -> RobustScaler
    log_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("winsor", PercentileClipper(0.005, 0.995)),
                         ("log1p", FunctionTransformer(_safe_log1p, feature_names_out="one-to-one")),
                         ("scale", RobustScaler())])
    plain_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("winsor", PercentileClipper(0.005, 0.995)),
                           ("scale", RobustScaler())])
    bins_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("kbins", KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="quantile"))])
    transformers = []
    if log_num:   transformers.append(("lognum", log_pipe, log_num))
    if plain_num: transformers.append(("num", plain_pipe, plain_num))
    if bin_cols:  transformers.append(("bins", bins_pipe, bin_cols))
    transformers += [("binflags", "passthrough", F.BINARY_FEATURES),
                     ("nom", _nominal_pipe(cfg), F.NOMINAL_FEATURES),
                     ("high", _highcard_pipe(cfg), F.HIGHCARD_FEATURES)]
    return ColumnTransformer(transformers)
