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
import joblib
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
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols

def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
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
    models["logreg"] = LogisticRegression(
        max_iter=500,
        class_weight="balanced" if class_weight_balanced else None,
        n_jobs=None,
    )
    models["rf"] = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        class_weight="balanced" if class_weight_balanced else None,
        random_state=42,
    )
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
            scale_pos_weight=scale_pos_weight,
        )
    except Exception as e:
        print("⚠️ XGBoost not available, skipping XGB. Error:\n", e)
    return models

def threshold_search(y_true: np.ndarray, probs: np.ndarray, recall_floor=0.60) -> Dict[str, float]:
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

def fit_and_select_model(train: pd.DataFrame, val: pd.DataFrame, recall_floor=0.60):
    # Split
    X_train, y_train = split_X_y(train)
    X_val, y_val = split_X_y(val)

    # Columns & preprocessing
    num_cols, cat_cols = detect_columns(X_train)
    pre = build_preprocessor(num_cols, cat_cols)

    # Imbalance ratio for XGB
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = (neg / pos) if pos > 0 else 1.0

    models = make_models(class_weight_balanced=True, scale_pos_weight=spw)

    best_name, best_pipe, best_metrics = None, None, {"f1": -1}
    # For saving per-model validation metrics and pipelines
    per_model = {}
    pipes = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        pipes[name] = pipe

        # Validation probabilities
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            val_probs = pipe.predict_proba(X_val)[:, 1]
        elif hasattr(pipe.named_steps["model"], "decision_function"):
            scores = pipe.decision_function(X_val)
            val_probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            val_probs = pipe.predict(X_val).astype(float)

        # Choose threshold on validation with recall constraint
        ts = threshold_search(y_val.values, val_probs, recall_floor=recall_floor)
        metrics = evaluate(y_val.values, val_probs, threshold=ts["threshold"])
        metrics.update({"chosen_threshold": ts["threshold"]})

        print(f"\n=== {name.upper()} (val) ===")
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

        per_model[name] = {"val": metrics}

        if metrics["f1"] > best_metrics["f1"]:
            best_name, best_pipe, best_metrics = name, pipe, metrics

    return best_name, best_pipe, best_metrics, per_model, pipes

def main():
    train, val, test = load_splits()
    best_name, best_pipe, best_val_metrics, per_model, pipes = fit_and_select_model(train, val, recall_floor=0.60)

    # Save the best pipeline
    model_path = os.path.join(MODELS_DIR, f"{best_name}_pipeline.pkl")
    joblib.dump(best_pipe, model_path)
    print(f"\n✅ Saved best model pipeline to: {model_path}")

    # Evaluate on TEST using the chosen threshold from best model (validation)
    X_test, y_test = split_X_y(test)
    if hasattr(best_pipe.named_steps["model"], "predict_proba"):
        test_probs_best = best_pipe.predict_proba(X_test)[:, 1]
    elif hasattr(best_pipe.named_steps["model"], "decision_function"):
        scores = best_pipe.named_steps["model"].decision_function(X_test)
        test_probs_best = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        test_probs_best = best_pipe.predict(X_test).astype(float)

    thr = best_val_metrics["chosen_threshold"]
    test_metrics_best = evaluate(y_test.values, test_probs_best, threshold=thr)
    test_metrics_best["used_threshold"] = thr

    # Save metrics for the best model
    metrics_path = os.path.join(MODELS_DIR, f"{best_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"val": best_val_metrics, "test": test_metrics_best}, f, indent=2)
    print(f"📄 Saved metrics to: {metrics_path}")

    print(f"\n=== BEST MODEL: {best_name.upper()} (test at thr={thr:.2f}) ===")
    pretty = {k: round(v, 4) if isinstance(v, float) else v for k, v in test_metrics_best.items()}
    print(pretty)

    # Edge-case sanity check (kept from earlier)
    def test_edge_case(pipe: Pipeline):
        sample = {
            "tenure": 2,
            "MonthlyCharges": 105.0,
            "TotalCharges": 210.0,
            "CLV": 105.0 * 6,
            "services_count": 1,
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

    test_edge_case(best_pipe)

    # --------- EXTRA: Save per-model TEST metrics + ROC points for app comparison/ROC plots ---------
    all_metrics = {}
    roc_bundle = {}

    for name, pipe in pipes.items():
        # Skip if pipeline wasn't created (e.g., xgb missing)
        if pipe is None:
            continue

        # Test probabilities for each model
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            probs = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe.named_steps["model"], "decision_function"):
            scores = pipe.named_steps["model"].decision_function(X_test)
            probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            probs = pipe.predict(X_test).astype(float)

        # Use the best model's threshold for apples-to-apples comparison
        m_test = evaluate(y_test.values, probs, threshold=thr)
        m_test["used_threshold"] = thr

        # For validation metrics we already stored in per_model[name]["val"]
        all_metrics[name] = {
            "val": per_model[name]["val"],
            "test": m_test
        }

        fpr, tpr, _ = roc_curve(y_test.values, probs)
        roc_bundle[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    with open(os.path.join(MODELS_DIR, "metrics_all.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(os.path.join(MODELS_DIR, "roc_curves_test.json"), "w") as f:
        json.dump(roc_bundle, f, indent=2)
    print("📄 Saved models/metrics_all.json and models/roc_curves_test.json")

if __name__ == "__main__":
    main()
