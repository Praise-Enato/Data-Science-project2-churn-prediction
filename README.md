# Customer Churn Prediction & CLV

Predict which telecom customers are likely to churn, estimate their Customer Lifetime Value (CLV), and prioritize retention in a deployed Streamlit app.

**Live app:** <ADD_YOUR_PUBLIC_URL>  
**Repo Clone URL:** <ADD_YOUR_REPO_URL>

---

## 1) Problem & Approach

**Goal.** Predict churn and estimate value (CLV) so we can focus retention where it matters most. The brief requires a single-page Streamlit app with tabs for Predict, Model Performance, and CLV Overview, plus deployment to Streamlit Community Cloud. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

**Models.** Logistic Regression (primary), Random Forest, and XGBoost; light tuning and test metrics: Precision, Recall, F1, ROC-AUC. :contentReference[oaicite:2]{index=2}

**Operating policy.** Validation chooses the operating threshold to **meet Recall ≥ 0.60** and maximize F1 among those thresholds; that threshold is then frozen and reported on test (reflected in the app’s Model Performance tab).

---

## 2) Data & Features

**Dataset.** IBM Telco Customer Churn (CSV). :contentReference[oaicite:3]{index=3}

**Processing.**
- Stratified 60/20/20 train/val/test; saved to `data/processed/`. :contentReference[oaicite:4]{index=4}
- Handle `TotalCharges` and simple imputations; categorical OHE; numeric median imputation.
- Engineered features per brief:
  - `tenure_bucket` ∈ {0–6m, 6–12m, 12–24m, 24m+} :contentReference[oaicite:5]{index=5}
  - `services_count` (count of enabled services) :contentReference[oaicite:6]{index=6}
  - `monthly_to_total_ratio` (safe divide) :contentReference[oaicite:7]{index=7}
  - `internet_no_tech_support` flag :contentReference[oaicite:8]{index=8}
  - **CLV** = `MonthlyCharges × ExpectedTenure(months)` with ExpectedTenure assumptions documented (6/12/24 for Month-to-month/One-year/Two-year). :contentReference[oaicite:9]{index=9}

---

## 3) Results

The **Model Performance** tab shows:
- Validation (chosen threshold) and Test (frozen threshold) metrics,
- A **model comparison (Test)** table for LOGREG / RF / XGB, and
- A **ROC (Test)** overlay chart. :contentReference[oaicite:10]{index=10}

**Interpretability.**
- **Local (per-prediction):** “Why this prediction?” tables based on Logistic Regression coefficients (processed feature space), with Log-odds Δ and Odds×. The brief asks to use **coefficient analysis** (standardized) for LogReg rather than SHAP KernelExplainer. :contentReference[oaicite:11]{index=11}
- **Global:** LogReg importance via \|coef × std\|, and optional SHAP global bars for trees. :contentReference[oaicite:12]{index=12} :contentReference[oaicite:13]{index=13}

**CLV Overview.** CLV quartiles (Low/Medium/High/Premium) and **churn rate by quartile** plus concise takeaways. :contentReference[oaicite:14]{index=14}

---

## 4) App Tour

- **Predict.** Friendly form → churn probability + risk badge; CLV estimate; local explanation (“pushes risk up/down”). :contentReference[oaicite:15]{index=15}
- **Model Performance.** Metrics table (all three models), ROC curves, confusion matrix; optional isotonic calibration toggle for probability quality. :contentReference[oaicite:16]{index=16}
- **CLV Overview.** CLV distribution & churn by quartile + short insight. :contentReference[oaicite:17]{index=17}

**Performance & caching.** We cache data and models with Streamlit’s `@st.cache_data` and `@st.cache_resource` so the app loads quickly (<2s per prediction on Cloud). :contentReference[oaicite:18]{index=18} :contentReference[oaicite:19]{index=19}

---

## 5) How to Run Locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build artifacts once
python -m src.data_prep
python -m src.clv_analysis
python -m src.model_train
python -m src.interpretability   # optional global plots

# Launch app
streamlit run app.py
