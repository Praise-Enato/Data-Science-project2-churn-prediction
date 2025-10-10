# src/data_prep.py
from __future__ import annotations
import os
import io
import sys
import textwrap
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
PROCESSED_DIR = "data/processed"

# ---- Helpers: teaching notes in comments ----
def load_raw_data(url: str = DATA_URL) -> pd.DataFrame:
    """
    Load the IBM Telco Customer Churn dataset directly from the public URL.
    Why direct URL? Keeps repo light and reproducible.
    """
    df = pd.read_csv(url)
    return df

def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    The dataset sometimes has blank strings in TotalCharges (often when tenure==0).
    Approach (explain & justify):
      1) Coerce to numeric; blanks -> NaN
      2) If tenure == 0 and TotalCharges is NaN: set to 0 (no months billed yet)
      3) Else if tenure > 0 and TotalCharges is NaN: approximate as MonthlyCharges * tenure
         (reasonable imputation tied to business meaning)
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    mask_tenure0 = (df["tenure"] == 0) & (df["TotalCharges"].isna())
    df.loc[mask_tenure0, "TotalCharges"] = 0.0

    mask_tenure_pos = (df["tenure"] > 0) & (df["TotalCharges"].isna())
    df.loc[mask_tenure_pos, "TotalCharges"] = df.loc[mask_tenure_pos, "MonthlyCharges"] * df.loc[mask_tenure_pos, "tenure"]

    # safety: still any NaNs? fill with median as a last resort
    if df["TotalCharges"].isna().any():
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple, explainable features requested by the brief:
      - tenure_bucket
      - services_count
      - monthly_to_total_ratio
      - internet_no_tech_support (flag)
      - Additional interaction/business signals to improve recall/precision
      - Secondary business features used by the Streamlit app
      - Map target y: Churn {No,Yes} -> {0,1}
    """
    df = df.copy()

    if "CLV" not in df.columns or "ExpectedTenure" not in df.columns:
        raise KeyError(
            "engineer_features expects CLV and ExpectedTenure columns. "
            "Run compute_expected_tenure_and_clv() before engineering features."
        )

    # tenure_bucket
    bins = [-0.1, 6, 12, 24, np.inf]
    labels = ["0-6m", "6-12m", "12-24m", "24m+"]
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    # services_count: count how many services are "on"
    # We'll treat 'Yes' as 1, and for InternetService (DSL/Fiber optic != 'No') as 1.
    service_cols_yesno = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    def yes_to_1(x):  # MultipleLines can be "No phone service" — treat that as 0
        return 1 if str(x).strip().lower() == "yes" else 0

    count_yes = df[service_cols_yesno].applymap(yes_to_1).sum(axis=1)
    internet_on = (df["InternetService"].str.lower() != "no").astype(int)
    df["services_count"] = count_yes + internet_on

    # monthly_to_total_ratio
    denom = (df["tenure"] * df["MonthlyCharges"]).replace(0, 1)  # avoid divide-by-zero
    df["monthly_to_total_ratio"] = df["TotalCharges"] / denom

    # flag: internet but no tech support
    df["internet_no_tech_support"] = (
        (df["InternetService"].str.lower() != "no") &
        (df["TechSupport"].str.lower() == "no")
    ).astype(int)

    # --- Additional engineered features to help the models ---
    # Ordinal encoding of tenure bucket for linear models
    tenure_ord = {"0-6m": 0, "6-12m": 1, "12-24m": 2, "24m+": 3}
    df["tenure_bucket_ord"] = df["tenure_bucket"].map(tenure_ord).fillna(0).astype(int)

    # Auto-pay indicator
    autopay_methods = {"bank transfer (automatic)", "credit card (automatic)"}
    df["is_auto_pay"] = df["PaymentMethod"].str.strip().str.lower().isin(autopay_methods).astype(int)

    # Long contract indicator (one or two year contracts)
    df["is_long_contract"] = df["Contract"].str.contains("year", case=False, na=False).astype(int)

    # Streaming / support service counts
    streaming_cols = ["StreamingTV", "StreamingMovies"]
    support_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["streaming_services"] = df[streaming_cols].applymap(lambda x: 1 if str(x).strip().lower() == "yes" else 0).sum(axis=1)
    df["support_services"] = df[support_cols].applymap(lambda x: 1 if str(x).strip().lower() == "yes" else 0).sum(axis=1)

    # Senior citizens on fiber optic internet (known high-risk segment)
    df["senior_fiber_optic"] = (
        (df["SeniorCitizen"] == 1) &
        (df["InternetService"].str.strip().str.lower() == "fiber optic")
    ).astype(int)

    # Charges per tenure month (helps capture early high spenders)
    df["charges_per_month_of_tenure"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # Indicator for high monthly charges relative to CLV (above median ratio)
    clv_ratio = df["MonthlyCharges"] / df["CLV"].replace(0, 1)
    median_ratio = clv_ratio.median()
    df["high_charge_to_clv_ratio"] = (clv_ratio >= median_ratio).astype(int)

    # Secondary engineered metrics used by the app for retention logic
    df["tenure_years"] = df["tenure"] / 12.0
    df["per_service_charge"] = df["MonthlyCharges"] / df["services_count"].replace(0, 1)
    df["is_fiber_optic"] = (df["InternetService"].str.strip().str.lower() == "fiber optic").astype(int)
    df["has_streaming_any"] = (df["streaming_services"] > 0).astype(int)
    df["has_support_any"] = (df["support_services"] > 0).astype(int)
    df["has_multiple_lines_yes"] = (df["MultipleLines"].str.strip().str.lower() == "yes").astype(int)
    pay_method_lower = df["PaymentMethod"].fillna("").str.strip().str.lower()
    df["is_electronic_check"] = (pay_method_lower == "electronic check").astype(int)
    paperless_lower = df["PaperlessBilling"].fillna("").str.strip().str.lower()
    df["paperless_autopay"] = ((paperless_lower == "yes") & (df["is_auto_pay"] == 1)).astype(int)
    partner_lower = df["Partner"].fillna("").str.strip().str.lower()
    dependents_lower = df["Dependents"].fillna("").str.strip().str.lower()
    df["family_bundle"] = ((partner_lower == "yes") | (dependents_lower == "yes")).astype(int)
    df["early_tenure_flag"] = (df["tenure"] <= 6).astype(int)
    monthly_median = df["MonthlyCharges"].median()
    df["high_monthly_charge"] = (df["MonthlyCharges"] >= monthly_median).astype(int)

    # Expected tenure derived features
    df["tenure_remaining"] = (df["ExpectedTenure"] - df["tenure"]).clip(lower=0)
    df["tenure_fraction_complete"] = (df["tenure"] / df["ExpectedTenure"].replace(0, 1)).clip(upper=5.0)
    df["remaining_value"] = (df["CLV"] - df["TotalCharges"]).clip(lower=0)
    df["remaining_value_ratio"] = df["remaining_value"] / df["CLV"].replace(0, 1)

    # Target mapping
    df["ChurnFlag"] = (df["Churn"].str.strip().str.lower() == "yes").astype(int)

    return df

def compute_expected_tenure_and_clv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline ExpectedTenure assumption (documented):
      Month-to-month -> 6 months
      One year      -> 12 months
      Two year      -> 24 months
    Then CLV = MonthlyCharges * ExpectedTenure
    """
    df = df.copy()
    mapping = {
        "month-to-month": 6,
        "one year": 12,
        "two year": 24,
    }
    df["ExpectedTenure"] = df["Contract"].str.strip().str.lower().map(mapping)
    # fallback if any unexpected labels
    df["ExpectedTenure"] = df["ExpectedTenure"].fillna(6)
    df["CLV"] = df["MonthlyCharges"] * df["ExpectedTenure"]
    return df

def select_model_columns(df: pd.DataFrame):
    """
    Separate features and target for later modeling.
    We keep raw categoricals for now; we'll one-hot encode in the model pipeline later.
    """
    target = "ChurnFlag"
    # keep core original columns + engineered ones:
    feature_cols = [
        # numerics
        "tenure", "MonthlyCharges", "TotalCharges", "CLV", "services_count",
        "monthly_to_total_ratio", "internet_no_tech_support",
        "tenure_bucket_ord", "is_auto_pay", "is_long_contract",
        "streaming_services", "support_services", "senior_fiber_optic",
        "charges_per_month_of_tenure", "high_charge_to_clv_ratio",
        "tenure_years", "per_service_charge", "is_fiber_optic",
        "has_streaming_any", "has_support_any", "has_multiple_lines_yes",
        "is_electronic_check", "paperless_autopay", "family_bundle",
        "early_tenure_flag", "high_monthly_charge", "tenure_remaining",
        "tenure_fraction_complete", "remaining_value",
        "remaining_value_ratio",
        # categoricals (kept as strings for now; encoder later)
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
        "tenure_bucket",
    ]
    # prune any missing columns defensively
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()
    y = df[target].copy()
    return X, y

def stratified_splits_and_save(X: pd.DataFrame, y: pd.Series, out_dir: str = PROCESSED_DIR, seed: int = 42):
    """
    60/20/20 stratified split. Save CSVs to data/processed/.
    """
    os.makedirs(out_dir, exist_ok=True)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    def save_pair(Xd, yd, name):
        df_out = Xd.copy()
        df_out["ChurnFlag"] = yd.values
        df_out.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

    save_pair(X_train, y_train, "train")
    save_pair(X_val, y_val, "val")
    save_pair(X_test, y_test, "test")

    print(f"Saved: {os.path.join(out_dir,'train.csv')}")
    print(f"       {os.path.join(out_dir,'val.csv')}")
    print(f"       {os.path.join(out_dir,'test.csv')}")

def main():
    print("Loading raw data…")
    df = load_raw_data()
    print(f"Rows: {len(df):,}")

    print("Cleaning TotalCharges…")
    df = clean_total_charges(df)

    print("Computing ExpectedTenure + CLV…")
    df = compute_expected_tenure_and_clv(df)

    print("Engineering features…")
    df = engineer_features(df)

    print("Preparing splits…")
    X, y = select_model_columns(df)
    stratified_splits_and_save(X, y)

    print(textwrap.dedent("""
    ✅ Data prep complete.
       - 60/20/20 splits saved to data/processed/
       - Features include: tenure_bucket, services_count, monthly_to_total_ratio, internet_no_tech_support, CLV
       Next: CLV quartiles + churn rate by quartile + charts.
    """))

if __name__ == "__main__":
    main()
