# AI Usage

## Summary

This project was completed by **Praise Enato** with targeted assistance from an AI coding tutor (ChatGPT 5).  
The AI acted as a **pair-programmer/coach**, helping design the workflow, draft code, explain concepts line-by-line, and troubleshoot deployment/UX issues.

**Live app:** <https://praise-enato-churn-project.streamlit.app/>
**Repository:** <https://github.com/Praise-Enato/Data-Science-project2-churn-prediction>

## What AI Helped With

### Project planning & structure

- Broke work into phases: setup → data prep → CLV analysis → modeling → interpretability → Streamlit app → deploy.

### Data preparation (`src/data_prep.py`)

- Line-by-line guidance to:
  - clean/raw conversions (e.g., `TotalCharges` numeric),
  - create target `ChurnFlag`,
  - engineer features: `services_count`, `monthly_to_total_ratio`, `internet_no_tech_support`, `tenure_bucket`,
  - compute simple **CLV = MonthlyCharges × ExpectedTenure** (6/12/24 by contract),
  - produce **60/20/20** train/val/test splits and save to `data/processed/`.

### CLV analysis (`src/clv_analysis.py`)

- Built quartiles with `qcut`, computed **churn rate by CLV quartile**, and saved a figure.
- Added **inline fallback** so the Streamlit app renders the CLV chart even if the PNG isn’t present in the cloud environment.

### Modeling & evaluation (`src/model_train.py`)

- Implemented **Logistic Regression**, **Random Forest**, **XGBoost** with:
  - imbalance care (LogReg/RF `class_weight`, XGB `scale_pos_weight`),
  - **early stopping** for XGB (validation AUC),
  - **recall-first threshold selection** on validation (≥0.60 recall, maximize F1), then **freeze threshold** for test.
- Exported artifacts for the app:
  - winning **pipeline** → `models/logreg_pipeline.pkl`,
  - `logreg_metrics.json` (winner’s val/test metrics + threshold),
  - `metrics_all.json` (per-model metrics),
  - `roc_curves_test.json` (ROC overlay points).

### Interpretability ('src/interpretability.py')

- **Per-prediction** explanation for LogReg using coefficient contributions (log-odds & odds multipliers).
- **Global importance**:
  - SHAP summary for tree models (when available),
  - fallback to `feature_importances_` for trees or absolute coefficients for LogReg.
- Humanized feature names (e.g., “Contract: Two year” instead of raw one-hot names).

### Streamlit app (`app.py`)

- Built modern UI (hero header, gradient background, KPI tiles, cards).
- **Predict** tab:
  - clean input form with presets,
  - probability + decision at operating threshold,
  - CLV estimate & segment,
  - **recommended retention strategies** driven by risk × CLV,
  - **Why this prediction?** tables (top features up/down).
- **Model Performance** tab:
  - Validation (chosen) vs Test (frozen thr.) table,
  - **model comparison** table,
  - **ROC overlay**,
  - **isotonic calibration** toggle (fit on validation, compare on test).
- **CLV Overview** tab:
  - displays precomputed chart or renders it on the fly from `train.csv`.

## What I the Author(Praise Enato) Did

- Ran and validated all code locally and in Streamlit Cloud.
- Chose final thresholds, inspected metrics, and evaluated trade-offs.
- Reviewed, edited, and integrated all generated code & UI.
- Deployed the app.
- Noticed errors in the code, and deployed and app and fixed them.

I reviewed all AI-generated outputs, corrected mistakes, and confirmed the final behaviour aligns with the grading rubric (business framing, model metrics, CLV insights, and deployment requirements).

## Issues Encountered & Fixes

- **“Why this prediction?” table showed raw one-hot names & confusing features**  
  *Fix:* Added a robust **pretty-naming** layer that maps pipeline feature names (e.g., `cat__PaperlessBilling_No`) to readable labels (e.g., **“Paperless Billing: No”**). Also ensured numeric contributions use **log-odds** and **odds multipliers** for interpretation.

- **Model results looked off (RF/XGB underperforming, inconsistent across runs)**  
  *Fixes:*  
  - Standardized preprocessing via a single **Pipeline** used end-to-end.  
  - Tuned baselines (RF depth/leafs, XGB learning rate/trees) + **early stopping** on validation.  
  - Added **imbalance handling** (LogReg/RF `class_weight`, XGB `scale_pos_weight`).  
  - Increased LogReg `max_iter=1000` to avoid convergence warnings.  
  - Ensured consistent **random seeds** and a clear **recall-first** threshold policy.  
  - Upgraded **Python version / requirements** so all libraries (incl. XGBoost/SHAP) install cleanly in Cloud.

- **CLV bar chart failed after deployment (“plot not found”)**  
  *Fix:* Implemented an **on-the-fly render** in the CLV tab: if the PNG isn’t available, the app recomputes quartiles & churn from `train.csv` and draws the chart inline.

- **Global importance for Random Forest/XGBoost not showing**  
  *Fix:* Added **SHAP** support for tree models (with safe sampling) and a **fallback** to `feature_importances_` if SHAP isn’t installed or supported by the environment.

- **Isotonic calibration toggle broke (API + caching errors)**  
  *Symptoms:*  
  - `CalibratedClassifierCV.__init__() got an unexpected keyword argument 'base_estimator'` (version mismatch).  
  - Streamlit cache error hashing unhashable arguments (Pipeline).  
  *Fixes:*  
  - Version-safe constructor: try `estimator=` then fall back to `base_estimator=`.  
  - Use `cv="prefit"` (calibrate the already-trained base model on validation).  
  - **Underscore-prefixed** cached function args (e.g., `_pipe`, `_val_df`) to prevent Streamlit from hashing objects it can’t serialize.  
  - Kept the **frozen threshold** constant; calibration only adjusts probability quality.

- **Feature/value mismatches from the Predict form (e.g., “No phone service”, senior citizen 0/1 vs Yes/No)**  
  *Fix:* Normalized categorical values exactly as during training and recomputed engineered features (`services_count`, `internet_no_tech_support`, `tenure_bucket`) **inside** the app before prediction.

- **SHAP performance risk in Cloud**  
  *Fix:* Added a small **sample** for the SHAP summary and a clean fallback path, so the app remains responsive.

## How AI Was Used (Scope & Boundaries)

- **Generation & refactoring:** Drafted Python/Streamlit code and iterated based on errors/results.
- **Teaching:** Gave beginner-friendly, **line-by-line explanations** for each module.
- **No background execution:** The AI did not run code or access private systems; all execution was performed by the author.
- **No private data:** Work used the public telco dataset and repository files only.
- **Libraries:** Standard Python ecosystem (`pandas`, `scikit-learn`, `xgboost`, `streamlit`, optional `shap`)
