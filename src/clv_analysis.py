# src/clv_analysis.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED_DIR = "data/processed"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------
# Helper: load the processed splits
# -----------------------------
def load_splits(processed_dir: str = PROCESSED_DIR):
    """
    We analyze on the TRAIN split to avoid peeking at validation/test.
    Why? It prevents subtle information leakage in your narrative & prevents 'tuning' to test.
    """
    train = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    test = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    return train, val, test

# -----------------------------
# Helper: cut CLV into quartiles
# -----------------------------
def add_clv_quartile(df: pd.DataFrame, clv_col: str = "CLV") -> pd.DataFrame:
    """
    pd.qcut makes equal-sized bins by rank (25% each).
    We label them in ascending order of CLV.
    """
    out = df.copy()
    # qcut will fail if there are duplicate edges; 'duplicates=drop' handles ties gracefully.
    out["CLV_quartile"] = pd.qcut(out[clv_col], q=4, labels=["Low", "Medium", "High", "Premium"], duplicates="drop")
    return out

# -----------------------------
# Helper: churn rate by quartile
# -----------------------------
def churn_rate_by_quartile(df: pd.DataFrame, y_col: str = "ChurnFlag"):
    """
    Returns a small summary table with size and churn rate for each quartile.
    churn_rate = mean of ChurnFlag (since it's 0/1).
    """
    summary = (
        df.groupby("CLV_quartile")[y_col]
          .agg(size="count", churn_rate="mean")
          .reset_index()
          .sort_values("CLV_quartile")  # keeps the logical Low→Premium order
    )
    return summary

# -----------------------------
# Helper: bar plot
# -----------------------------
def plot_churn_rate(summary: pd.DataFrame, savepath: str):
    """
    Matplotlib only, single plot, no custom colors (keeps it clean & portable).
    """
    plt.figure(figsize=(6, 4))
    # We multiply by 100 to show percentages
    plt.bar(summary["CLV_quartile"].astype(str), summary["churn_rate"] * 100)
    plt.title("Churn Rate by CLV Quartile")
    plt.xlabel("CLV Quartile")
    plt.ylabel("Churn Rate (%)")
    # Annotate bars with values
    for x, y in zip(summary["CLV_quartile"].astype(str), summary["churn_rate"] * 100):
        plt.text(x, y + 0.5, f"{y:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()

# -----------------------------
# Helper: print narrative insights
# -----------------------------
def print_insights(summary: pd.DataFrame):
    """
    Generates a few business-friendly bullets.
    We compute relative differences to highlight strongest contrasts.
    """
    # Convert to dicts for quick lookup
    sdict = {row["CLV_quartile"]: row for _, row in summary.iterrows()}
    # Guard: if labels got collapsed (rare with duplicates=drop), handle gracefully
    labels = [str(x) for x in summary["CLV_quartile"].tolist()]

    print("\n===== Business Insights (auto-generated) =====")
    # 1) Overall ordering insight (Low vs Premium if both exist)
    if "Low" in sdict and "Premium" in sdict:
        low = sdict["Low"]["churn_rate"] * 100
        prem = sdict["Premium"]["churn_rate"] * 100
        diff = prem - low
        trend = "higher" if diff > 0 else "lower"
        print(f"- Premium vs Low: Premium churn is {abs(diff):.1f} pp {trend} than Low "
              f"({prem:.1f}% vs {low:.1f}%).")
    # 2) Highest-risk quartile
    worst = summary.iloc[summary["churn_rate"].argmax()]
    print(f"- Highest churn segment: {worst['CLV_quartile']} at {worst['churn_rate']*100:.1f}% churn.")
    # 3) Prioritization hint (top two)
    top2 = summary.sort_values("churn_rate", ascending=False).head(2)
    segs = ", ".join([f"{r['CLV_quartile']} ({r['churn_rate']*100:.1f}%)" for _, r in top2.iterrows()])
    print(f"- Retention priority suggestion: focus on {segs}.")

    # 4) Size context
    total = summary["size"].sum()
    for _, r in summary.iterrows():
        pct = r["size"] / total * 100 if total else 0
        print(f"  • {r['CLV_quartile']}: {r['size']} customers ({pct:.1f}% of train), churn {r['churn_rate']*100:.1f}%")
    print("=============================================\n")

# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading processed splits…")
    train, val, test = load_splits()

    # Basic sanity checks for required columns
    needed = {"CLV", "ChurnFlag"}
    missing = needed - set(train.columns)
    if missing:
        raise ValueError(f"Missing columns in train split: {missing}. "
                         f"Did you run src/data_prep.py successfully?")

    print("Adding CLV quartiles to TRAIN…")
    train_q = add_clv_quartile(train, clv_col="CLV")

    print("Computing churn rate by quartile…")
    summary = churn_rate_by_quartile(train_q, y_col="ChurnFlag")
    print(summary)

    # Save a version with the quartile label (could be useful for the app)
    out_csv = os.path.join(PROCESSED_DIR, "train_with_clv_quartile.csv")
    train_q.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Plot & save
    fig_path = os.path.join(FIG_DIR, "churn_rate_by_clv_quartile.png")
    plot_churn_rate(summary, fig_path)
    print(f"Saved figure: {fig_path}")

    # Narrative insights
    print_insights(summary)

if __name__ == "__main__":
    main()
