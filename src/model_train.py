# src/model_train.py
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Optional: if you ever see many pandas warnings, you can filter them:
import warnings
warnings.filterwarnings("ignore", message="The default of observed=False", category=FutureWarning)

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def load_splits(processed_dir: str = PROCESSED_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    test = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    return train, val, test

def split_X_y(df: pd.DataFrame, target: str = "ChurnFlag") -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    return X, y

def detect_columns(X: pd.DataFrame) -> Tuple[list, list]:
    """Identify numeric vs categorical columns for preprocessing."""
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    # Booleans we’ll treat as numeric (0/1); object/category => categorical
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols

def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_cols),
            ("cat", categorical_tf, categorical_cols),
        ]
    )
    return preprocessor

def make_models(class_weight_balanced=True, scale_pos_weight: float = 1.0) -> Dict[str, object]:
    models = {}

    # 1) Logistic Regression
    models["logreg"] = LogisticRegression(
        max_iter=2000,
        class_weight="balanced" if class_weight_balanced else None,
        n_jobs=None,
    )

    # 2) Random Forest
    models["rf"] = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        class_weight="balanced" if class_weight_balanced else None,
        random_state=42,
    )

    # 3) XGBoost (optional import because xgboost might not be installed in some envs)
    try:
        from xgboost import XGBClassifier
        models["xgb"] = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,  # key for imbalance
        )
    except Exception as e:
        print("⚠️ XGBoost not available, skipping XGB. Error:", e)

    return models

def threshold_search(y_true: np.ndarray, probs: np.ndarray, recall_floor=0.60) -> Dict[str, float]:
    """
    Scan thresholds from 0.10 to 0.90 and pick the one that achieves recall >= recall_floor
    and maximizes F1 among those. Returns dict with best_threshold, precision, recall, f1.
    """
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for thr in np.linspace(0.1, 0.9, 81):
        preds = (probs >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
        if r >= recall_floor and f1 > best["f1"]:
            best = {"threshold": float(thr), "precision": float(p), "recall": float(r), "f1": float(f1)}
    return best

def evaluate(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc": float(auc),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def get_model_probs(pipe: Pipeline, X_df: pd.DataFrame) -> np.ndarray:
    """Return probabilities (or scaled scores) for positive class from a fitted pipeline."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X_df)[:, 1]
    if hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(X_df)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return pipe.predict(X_df).astype(float)

def fit_and_select_model(
    train: pd.DataFrame, val: pd.DataFrame, recall_floor=0.60
) -> Tuple[str, Pipeline, Dict, Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
    # Split
    X_train, y_train = split_X_y(train)
    X_val, y_val = split_X_y(val)

    # Columns & preprocessing
    num_cols, cat_cols = detect_columns(X_train)
    pre = build_preprocessor(num_cols, cat_cols)

    # Compute imbalance ratio for XGB
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = (neg / pos) if pos > 0 else 1.0

    models = make_models(class_weight_balanced=True, scale_pos_weight=spw)

    best_name = None
    best_pipe = None
    best_metrics = {"f1": -1}
    fitted_pipes: Dict[str, Pipeline] = {}
    val_metrics_map: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)

        val_probs = get_model_probs(pipe, X_val)

        # Choose threshold on validation with recall constraint
        ts = threshold_search(y_val.values, val_probs, recall_floor=recall_floor)
        # Evaluate at chosen threshold
        metrics = evaluate(y_val.values, val_probs, threshold=ts["threshold"])
        metrics.update({"chosen_threshold": ts["threshold"]})
        fitted_pipes[name] = pipe
        val_metrics_map[name] = metrics

        print(f"\n=== {name.upper()} (val) ===")
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

        # Track best by F1 (you can change to recall-first if desired)
        if metrics["f1"] > best_metrics["f1"]:
            best_name, best_pipe, best_metrics = name, pipe, metrics

    return best_name, best_pipe, best_metrics, fitted_pipes, val_metrics_map

def test_edge_case(pipe: Pipeline):
    """
    Sanity check from the brief:
    SeniorCitizen=1, Contract=month-to-month, InternetService=Fiber optic,
    no add-on services, PaymentMethod=Electronic check, MonthlyCharges >= 100.
    Other fields to reasonable defaults.
    """
    sample = {
        "tenure": 2,
        "MonthlyCharges": 105.0,
        "TotalCharges": 210.0,
        "CLV": 105.0 * 6,  # month-to-month -> 6 months expected
        "services_count": 1,  # just internet
        "monthly_to_total_ratio": 210.0 / max(1, 2 * 105.0),
        "internet_no_tech_support": 1,
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "tenure_bucket": "0-6m",
    }
    X = pd.DataFrame([sample])
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        prob = pipe.predict_proba(X)[:, 1][0]
    else:
        pred = pipe.predict(X)[0]
        prob = float(pred)
    pred = int(prob >= 0.5)
    print("\nEdge-case sanity check:")
    print(f"Predicted churn prob ~ {prob:.3f}  -> class @0.5 = {pred} (1 means churn)")

def main():
    train, val, test = load_splits()
    (
        best_name,
        best_pipe,
        best_val_metrics,
        model_pipes,
        val_metrics_map,
    ) = fit_and_select_model(train, val, recall_floor=0.60)

    # Save the best pipeline
    model_path = os.path.join(MODELS_DIR, f"{best_name}_pipeline.pkl")
    joblib.dump(best_pipe, model_path)
    print(f"\n✅ Saved best model pipeline to: {model_path}")

    # Evaluate on TEST at the chosen threshold
    X_test, y_test = split_X_y(test)
    test_metrics_map: Dict[str, Dict[str, float]] = {}
    test_probs_map: Dict[str, np.ndarray] = {}

    for name, pipe in model_pipes.items():
        test_probs = get_model_probs(pipe, X_test)
        thr = val_metrics_map[name]["chosen_threshold"]
        metrics = evaluate(y_test.values, test_probs, threshold=thr)
        metrics["used_threshold"] = thr
        test_metrics_map[name] = metrics
        test_probs_map[name] = test_probs

    best_test_metrics = test_metrics_map[best_name]

    # Save metrics
    metrics_path = os.path.join(MODELS_DIR, f"{best_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"val": best_val_metrics, "test": best_test_metrics}, f, indent=2)
    print(f"📄 Saved metrics to: {metrics_path}")

    # Print test metrics
    best_thr = best_val_metrics["chosen_threshold"]
    print(f"\n=== BEST MODEL: {best_name.upper()} (test at thr={best_thr:.2f}) ===")
    pretty = {k: round(v, 4) if isinstance(v, float) else v for k, v in best_test_metrics.items()}
    print(pretty)

    # Persist aggregated metrics for downstream comparison
    all_metrics = {}
    for name in ["logreg", "rf", "xgb"]:
        if name in val_metrics_map and name in test_metrics_map:
            all_metrics[name] = {
                "val": val_metrics_map[name],
                "test": test_metrics_map[name],
            }

    if all_metrics:
        metrics_all_path = os.path.join(MODELS_DIR, "metrics_all.json")
        with open(metrics_all_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"📄 Saved all-model metrics to: {metrics_all_path}")

    # Store ROC curve coordinates for overlay plot
    def _roc_points(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, object]:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    roc_bundle: Dict[str, Dict[str, object]] = {}
    for name in ["logreg", "rf", "xgb"]:
        pipe = model_pipes.get(name)
        if pipe is None or name not in test_probs_map:
            continue
        roc_bundle[name] = _roc_points(y_test.values, test_probs_map[name])

    if roc_bundle:
        roc_path = os.path.join(MODELS_DIR, "roc_curves_test.json")
        with open(roc_path, "w") as f:
            json.dump(roc_bundle, f, indent=2)
        print(f"📄 Saved ROC curves to: {roc_path}")

    # Edge-case check
    test_edge_case(best_pipe)

if __name__ == "__main__":
    main()
