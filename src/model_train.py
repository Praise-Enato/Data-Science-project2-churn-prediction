# src/model_train.py
# Train LogReg, RandomForest, and XGBoost with light tuning + imbalance care.
# Select a recall-first operating threshold on validation, then freeze for test.
# Saves:
#   models/logreg_pipeline.pkl              (best model = usually LOGREG here)
#   models/logreg_metrics.json              (chosen/best model metrics)
#   models/metrics_all.json                 (per-model val/test)
#   models/roc_curves_test.json             (for the ROC overlay in the app)

import inspect
import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional: XGBoost (skip gracefully if not present)
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception as e:
    _HAS_XGB = False
    _XGB_ERR = str(e)

warnings.filterwarnings("ignore")

MODELS_DIR = "models"
DATA_DIR = "data/processed"
os.makedirs(MODELS_DIR, exist_ok=True)

VAL_RECALL_MIN = 0.60  # recall-first policy on validation
THRESH_GRID = np.round(np.linspace(0.30, 0.80, 51), 3)  # 0.30 → 0.80 (step ~0.01)
RANDOM_STATE = 42


# ------------------------- Data Loading -------------------------
def _load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [os.path.join(DATA_DIR, f) for f in ["train.csv", "val.csv", "test.csv"]]
    if not all(os.path.exists(p) for p in paths):
        raise FileNotFoundError(
            f"Processed splits not found in {DATA_DIR}. Run: python -m src.data_prep"
        )
    train = pd.read_csv(paths[0])
    val   = pd.read_csv(paths[1])
    test  = pd.read_csv(paths[2])
    return train, val, test


# ------------------------- Features -------------------------
def _feature_lists(df: pd.DataFrame) -> Tuple[list, list]:
    # Numeric engineered features you created in data_prep.py
    num_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "CLV",
        "services_count",
        "monthly_to_total_ratio",
        "internet_no_tech_support",
    ]
    # Categorical features
    cat_cols = [
        "gender",
        "SeniorCitizen",           # treat as categorical (0/1)
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "tenure_bucket",
    ]
    # Keep only columns that actually exist
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]
    return num_cols, cat_cols


def _preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    # scikit-learn >=1.4 renamed `sparse` to `sparse_output`; handle both.
    ohe_kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames:
        ohe_kwargs["sparse_output"] = True
    else:
        ohe_kwargs["sparse"] = True
    cat_encoder = OneHotEncoder(**ohe_kwargs)

    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("ohe", cat_encoder)])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre


# ------------------------- Models -------------------------
def make_logreg() -> LogisticRegression:
    # class_weight balances the minority (churn) class
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        n_jobs=None,
        random_state=RANDOM_STATE,
    )


def make_rf() -> RandomForestClassifier:
    # Light tuning that generalizes well on Telco; balanced_subsample helps imbalance
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_xgb(scale_pos_weight: float):
    # Common, robust defaults + early stopping (handled in fit block)
    return XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
    )


# ------------------------- Metrics & Thresholds -------------------------
@dataclass
class EvalResult:
    metrics_val: Dict
    metrics_test: Dict
    thr: float
    roc_test: Dict[str, list]


def _bin_metrics(y_true, y_prob, thr) -> Dict:
    y_pred = (y_prob >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc": float(auc),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "used_threshold": float(thr),
    }


def _choose_threshold_recall_first(y_true, y_prob, recall_min=VAL_RECALL_MIN):
    best_thr = 0.50
    best_f1 = -1.0
    met = None
    for thr in THRESH_GRID:
        m = _bin_metrics(y_true, y_prob, thr)
        if m["recall"] >= recall_min and m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = thr
            met = m
    if met is None:
        # If no threshold reaches target recall, fall back to best f1 overall
        for thr in THRESH_GRID:
            m = _bin_metrics(y_true, y_prob, thr)
            if m["f1"] > best_f1:
                best_f1 = m["f1"]; best_thr = thr; met = m
    return best_thr, met


def _roc_points(y_true, y_prob) -> Tuple[list, list]:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return list(map(float, fpr)), list(map(float, tpr))


# ------------------------- Train/Eval -------------------------
def train_and_eval():
    train, val, test = _load_splits()
    y_tr = train["ChurnFlag"].astype(int).values
    y_va = val["ChurnFlag"].astype(int).values
    y_te = test["ChurnFlag"].astype(int).values

    X_tr = train.drop(columns=["ChurnFlag"])
    X_va = val.drop(columns=["ChurnFlag"])
    X_te = test.drop(columns=["ChurnFlag"])

    num_cols, cat_cols = _feature_lists(train)
    pre = _preprocessor(num_cols, cat_cols)

    # Pos/neg counts for XGB scale_pos_weight
    pos = int(y_tr.sum())
    neg = int((y_tr == 0).sum())
    spw = neg / max(1, pos)

    results = {}
    roc_dict = {}

    # ---------- LOGREG ----------
    logreg = make_logreg()
    pipe_logreg = Pipeline(steps=[("pre", pre), ("model", logreg)])
    pipe_logreg.fit(X_tr, y_tr)
    joblib.dump(pipe_logreg, os.path.join(MODELS_DIR, "logreg_baseline_pipeline.pkl"))

    prob_va = pipe_logreg.predict_proba(X_va)[:, 1]
    thr, met_val = _choose_threshold_recall_first(y_va, prob_va, VAL_RECALL_MIN)
    prob_te = pipe_logreg.predict_proba(X_te)[:, 1]
    met_test = _bin_metrics(y_te, prob_te, thr)
    fpr, tpr = _roc_points(y_te, prob_te)
    roc_dict["logreg"] = {"fpr": fpr, "tpr": tpr}

    results["logreg"] = {"val": met_val, "test": met_test, "thr": thr}

    # ---------- RANDOM FOREST ----------
    rf = make_rf()
    pipe_rf = Pipeline(steps=[("pre", pre), ("model", rf)])
    pipe_rf.fit(X_tr, y_tr)

    prob_va = pipe_rf.predict_proba(X_va)[:, 1]
    thr_rf, met_val_rf = _choose_threshold_recall_first(y_va, prob_va, VAL_RECALL_MIN)
    prob_te = pipe_rf.predict_proba(X_te)[:, 1]
    met_test_rf = _bin_metrics(y_te, prob_te, thr_rf)
    fpr, tpr = _roc_points(y_te, prob_te)
    roc_dict["rf"] = {"fpr": fpr, "tpr": tpr}
    results["rf"] = {"val": met_val_rf, "test": met_test_rf, "thr": thr_rf}
    joblib.dump(pipe_rf, os.path.join(MODELS_DIR, "rf_pipeline.pkl"))

    # ---------- XGBOOST (optional) ----------
    if _HAS_XGB:
        xgb = make_xgb(spw)
        # Fit XGB with early stopping on validation
        # We pass the preprocessed arrays explicitly to leverage early stopping.
        X_tr_pre = pipe_logreg.named_steps["pre"].fit_transform(X_tr)  # reuse pre to get feature matrix
        X_va_pre = pipe_logreg.named_steps["pre"].transform(X_va)

        # XGBoost>=2.1 prefers callbacks for early stopping; fall back for older versions.
        fit_kwargs = {
            "X": X_tr_pre,
            "y": y_tr,
            "eval_set": [(X_va_pre, y_va)],
            "verbose": False,
        }
        fit_sig = inspect.signature(xgb.fit)
        try:
            if "callbacks" in fit_sig.parameters:
                try:
                    from xgboost.callback import EarlyStopping
                    callbacks = [EarlyStopping(rounds=50, save_best=True)]
                    xgb.fit(**fit_kwargs, callbacks=callbacks)
                except Exception:
                    xgb.fit(**fit_kwargs)
            elif "early_stopping_rounds" in fit_sig.parameters:
                xgb.fit(**fit_kwargs, early_stopping_rounds=50)
            else:
                xgb.fit(**fit_kwargs)
        except TypeError:
            xgb.fit(**fit_kwargs)
        # Build a pipeline wrapper so app can use a consistent interface
        pipe_xgb = Pipeline(steps=[("pre", pre), ("model", xgb)])

        prob_va = pipe_xgb.predict_proba(X_va)[:, 1]
        thr_xgb, met_val_xgb = _choose_threshold_recall_first(y_va, prob_va, VAL_RECALL_MIN)
        prob_te = pipe_xgb.predict_proba(X_te)[:, 1]
        met_test_xgb = _bin_metrics(y_te, prob_te, thr_xgb)
        fpr, tpr = _roc_points(y_te, prob_te)
        roc_dict["xgb"] = {"fpr": fpr, "tpr": tpr}
        results["xgb"] = {"val": met_val_xgb, "test": met_test_xgb, "thr": thr_xgb}
        joblib.dump(pipe_xgb, os.path.join(MODELS_DIR, "xgb_pipeline.pkl"))
    else:
        print(f"⚠️ XGBoost not available, skipping. Error: {_XGB_ERR}")

    # ---------------- Choose best model by validation F1 under recall constraint ----------------
    # (You can change selection logic if the brief asks for a different tie-breaker.)
    def key_fn(k):
        return results[k]["val"]["f1"]

    best_name = max(results.keys(), key=key_fn)
    best_thr = results[best_name]["thr"]

    # Save the *LogReg* pipeline as default (the app expects this path);
    # if another model won, we still save the winning pipeline under this path for simplicity.
    best_pipe = {
        "logreg": pipe_logreg,
        "rf": pipe_rf,
        "xgb": pipe_xgb if _HAS_XGB else pipe_logreg,
    }[best_name]

    # Persist the winning pipeline + its metrics
    joblib.dump(best_pipe, os.path.join(MODELS_DIR, "logreg_pipeline.pkl"))

    with open(os.path.join(MODELS_DIR, "logreg_metrics.json"), "w") as f:
        json.dump(
            {
                "val": results[best_name]["val"],
                "test": results[best_name]["test"],
                "used_threshold": best_thr,
                "model_name": best_name,
            },
            f,
            indent=2,
        )

    # Persist per-model metrics for the app comparison table
    with open(os.path.join(MODELS_DIR, "metrics_all.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Persist ROC curves for the app overlay
    with open(os.path.join(MODELS_DIR, "roc_curves_test.json"), "w") as f:
        json.dump(roc_dict, f, indent=2)

    # Convenience prints
    print(f"\n=== BEST MODEL: {best_name.upper()} (test at thr={best_thr:.2f}) ===")
    print(results[best_name]["test"])


if __name__ == "__main__":
    train_and_eval()
