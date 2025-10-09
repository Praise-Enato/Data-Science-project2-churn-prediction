# src/interpret.py
from __future__ import annotations
import os
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

MODELS_DIR = "models"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def get_feature_names(preprocessor) -> List[str]:
    """
    Recover the final feature names after ColumnTransformer + OneHotEncoder.
    Works in sklearn>=1.0 where get_feature_names_out is supported.
    """
    try:
        names = preprocessor.get_feature_names_out()
        return names.tolist()
    except Exception:
        # Fallback: build names manually for older versions (unlikely with your env)
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if hasattr(trans, "named_steps") and "ohe" in trans.named_steps:
                # categorical pipeline
                ohe = trans.named_steps["ohe"]
                imputer = trans.named_steps.get("imputer", None)
                base_cols = cols
                # Handle case where imputer may change dtype but not names
                cat_names = ohe.get_feature_names_out(base_cols)
                names.extend(cat_names.tolist())
            else:
                # numeric pipeline: names are just cols
                names.extend(cols if isinstance(cols, list) else list(cols))
        return names

def main():
    # Load best pipeline (we saved logreg as best)
    model_path = os.path.join(MODELS_DIR, "logreg_pipeline.pkl")
    pipe = joblib.load(model_path)

    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]

    # We handle linear models via coefficients; otherwise fall back to feature_importances.
    use_coeffs = hasattr(model, "coef_")
    use_importance = hasattr(model, "feature_importances_")

    if not use_coeffs and not use_importance:
        raise ValueError(
            f"{model.__class__.__name__} lacks coefficients or feature_importances_. "
            "Cannot generate interpretability artifacts."
        )

    # Get expanded feature names after preprocessing
    feat_names = get_feature_names(pre)

    # Coefficients (shape: [1, n_features])
    if use_coeffs:
        values = model.coef_.ravel()
    else:
        values = model.feature_importances_.ravel()
    if len(values) != len(feat_names):
        raise ValueError(f"Mismatch: {len(values)} weights vs {len(feat_names)} feature names")

    # Build a tidy table
    df_imp = pd.DataFrame({
        "feature": feat_names,
        "value": values,
    })
    if use_coeffs:
        df_imp["odds_ratio"] = np.exp(df_imp["value"])
    else:
        df_imp["odds_ratio"] = np.nan  # not meaningful for tree models
    df_imp["abs_value"] = df_imp["value"].abs()

    # Sort for top drivers
    sort_col = "value"
    top_positive = df_imp.sort_values(sort_col, ascending=False).head(20)
    top_negative = df_imp.sort_values(sort_col, ascending=True).head(20)

    # Save CSVs
    out_csv_all = os.path.join(MODELS_DIR, "logreg_feature_importance_full.csv")
    out_csv_pos = os.path.join(MODELS_DIR, "logreg_top_positive.csv")
    out_csv_neg = os.path.join(MODELS_DIR, "logreg_top_negative.csv")
    df_imp.to_csv(out_csv_all, index=False)
    top_positive.to_csv(out_csv_pos, index=False)
    top_negative.to_csv(out_csv_neg, index=False)

    print(f"Saved: {out_csv_all}")
    print(f"Saved: {out_csv_pos}")
    print(f"Saved: {out_csv_neg}")

    # --- PLOTS (horizontal bar charts) ---
    def plot_barh(df, title, savepath):
        plt.figure(figsize=(8, 6))
        df_plot = df.copy()
        df_plot = df_plot.sort_values(sort_col, ascending=True)
        labels = df_plot["feature"].str.replace("cat__ohe__", "", regex=False)
        plt.barh(labels, df_plot[sort_col])
        plt.xlabel("Coefficient" if use_coeffs else "Feature Importance")
        plt.title(title)
        if use_coeffs:
            for y, (_, row) in enumerate(df_plot.iterrows()):
                plt.text(row[sort_col] + 0.01, y, f"{row['odds_ratio']:.2f}", va="center")
        plt.tight_layout()
        plt.savefig(savepath, dpi=200)
        plt.close()

    pos_title = "Top Positive Drivers of Churn (higher → more likely to churn)" if use_coeffs else "Top Features Increasing Predicted Risk"
    neg_title = "Top Negative Drivers of Churn (higher → less likely to churn)" if use_coeffs else "Top Features Decreasing Predicted Risk"
    plot_barh(top_positive, pos_title, os.path.join(FIG_DIR, "logreg_top_positive_odds.png"))
    plot_barh(top_negative, neg_title, os.path.join(FIG_DIR, "logreg_top_negative_odds.png"))

    print("Saved figures:")
    print(" - figures/logreg_top_positive_odds.png")
    print(" - figures/logreg_top_negative_odds.png")

if __name__ == "__main__":
    main()
