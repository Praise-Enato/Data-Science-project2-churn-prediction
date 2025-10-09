# src/interpretability.py
import os, json, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path("figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models")
DATA_DIR = Path("data/processed")

def _load_splits():
    train = pd.read_csv(DATA_DIR/"train.csv")
    val   = pd.read_csv(DATA_DIR/"val.csv")
    test  = pd.read_csv(DATA_DIR/"test.csv")
    return train, val, test

def _safe_load(path):
    return joblib.load(path) if path.exists() else None

def logreg_importance():
    pipe = _safe_load(MODELS_DIR/"logreg_pipeline.pkl")
    if pipe is None:
        print("LogReg pipeline missing; skip."); return
    pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]

    # compute feature std on training preprocessed data (for standardized importances)
    train, _, _ = _load_splits()
    Xtr = train.drop(columns=["ChurnFlag"])
    Xtr_pre = pre.transform(Xtr)
    Xtr_pre = Xtr_pre.toarray() if hasattr(Xtr_pre, "toarray") else np.asarray(Xtr_pre)
    feat_names = pre.get_feature_names_out()

    if hasattr(model, "coef_"):
        std = Xtr_pre.std(axis=0)
        coefs = model.coef_.ravel()
        imp = np.abs(coefs * std)
        title = "LogReg global importance (|coef × std|)"
    elif hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        title = f"{model.__class__.__name__} feature importances"
    else:
        print(f"{model.__class__.__name__} lacks coefficients/feature_importances; skipping global plot.")
        return

    top_idx = np.argsort(imp)[::-1][:20]
    top_feats = feat_names[top_idx]
    top_vals  = imp[top_idx]

    plt.figure()
    plt.barh(range(len(top_vals))[::-1], top_vals[::-1])
    plt.yticks(range(len(top_vals))[::-1], top_feats[::-1])
    plt.title(title)
    plt.tight_layout()
    out = FIG_DIR/"logreg_global_importance.png"
    plt.savefig(out, dpi=200); plt.close()
    print(f"Saved {out}")

def shap_trees(which="rf"):
    try:
        import shap
    except Exception as e:
        print("SHAP not available; skip trees.", e); return
    model_path = MODELS_DIR/(f"{which}_pipeline.pkl")
    pipe = _safe_load(model_path)
    if pipe is None:
        print(f"{which.upper()} pipeline missing; skip."); return

    pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
    _, _, test = _load_splits()
    X = test.drop(columns=["ChurnFlag"])
    # sample to keep it snappy
    Xs = X.sample(min(300, len(X)), random_state=42)
    Xs_pre = pre.transform(Xs)
    if hasattr(Xs_pre, "toarray"):
        Xs_pre = Xs_pre.toarray()
    else:
        Xs_pre = np.asarray(Xs_pre)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs_pre)
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_use = shap_values[-1]
    else:
        shap_values_use = shap_values
    # Global bar plot
    feat_names = pre.get_feature_names_out()
    plt.figure()
    shap.summary_plot(shap_values_use, features=Xs_pre, feature_names=feat_names, plot_type="bar", show=False)
    out = FIG_DIR/f"{which}_shap_global_bar.png"
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
    print(f"Saved {out}")

if __name__ == "__main__":
    logreg_importance()
    shap_trees("rf")
    shap_trees("xgb")
