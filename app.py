# app.py
# Customer Churn Prediction & CLV — Streamlit App
# Polished UI + per-feature explanation (LogReg) + optional isotonic calibration

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

# ---------------------------------------------------------------
# Page config + light CSS polish
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn & CLV",
    page_icon="📉",
    layout="wide",
)

st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: .2px; }
[data-testid="stMetricValue"] { font-weight: 700; }

/* Card container */
.card {
  background: var(--secondary-background-color);
  padding: 1rem 1.25rem;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.06);
}
.section { margin-top: .5rem; }

/* Risk badges */
.badge { display:inline-block; padding:.25rem .5rem; border-radius: 999px;
  font-size:.8rem; border:1px solid rgba(255,255,255,.15); }
.badge-low  { background:#10381a; color:#9BE29B; border-color:#205c32; }
.badge-med  { background:#3a2b12; color:#f6d087; border-color:#6c4e19; }
.badge-high { background:#3d1718; color:#FF9E9E; border-color:#6a2426; }

.table-compact td, .table-compact th { padding: .35rem .5rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"
FIG_DIR = "figures"

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "logreg_pipeline.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "logreg_metrics.json")
CLV_QUARTILE_PLOT = os.path.join(FIG_DIR, "churn_rate_by_clv_quartile.png")

# Operating threshold picked on validation during training
OPERATING_THRESHOLD = 0.59

# Simple CLV assumption (documented): Month-to-month=6, One year=12, Two year=24
EXPECTED_TENURE_MAP = {"Month-to-month": 6, "One year": 12, "Two year": 24}

# Pretty labels for original column names
COLUMN_TITLE_MAP = {
    "MonthlyCharges": "Monthly Charges",
    "TotalCharges": "Total Charges",
    "SeniorCitizen": "Senior Citizen",
    "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Method",
    "InternetService": "Internet Service",
    "DeviceProtection": "Device Protection",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "MultipleLines": "Multiple Lines",
    "PhoneService": "Phone Service",
    "tenure_bucket": "Tenure Bucket",
    "Contract": "Contract",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "gender": "Gender",
    "CLV": "CLV",
    "services_count": "Services Count",
    "monthly_to_total_ratio": "Monthly/Total Ratio",
    "internet_no_tech_support": "Internet w/o Tech Support",
    "tenure": "Tenure",
}

# ---------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------
@st.cache_resource
def load_best_pipeline():
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model pipeline not found at {BEST_MODEL_PATH}. "
            "Run training first: python -m src.model_train"
        )
    return joblib.load(BEST_MODEL_PATH)

@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            f"Metrics JSON not found at {METRICS_PATH}. "
            "Run training first: python -m src.model_train"
        )
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

@st.cache_data
def load_processed_splits():
    train_p = os.path.join(PROCESSED_DIR, "train.csv")
    val_p = os.path.join(PROCESSED_DIR, "val.csv")
    test_p = os.path.join(PROCESSED_DIR, "test.csv")
    if not (os.path.exists(train_p) and os.path.exists(val_p) and os.path.exists(test_p)):
        raise FileNotFoundError(
            f"Processed splits not found in {PROCESSED_DIR}. "
            "Generate them with: python -m src.data_prep"
        )
    return (
        pd.read_csv(train_p),
        pd.read_csv(val_p),
        pd.read_csv(test_p),
    )

# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------
def compute_clv(monthly_charges: float, contract: str) -> float:
    exp_tenure = EXPECTED_TENURE_MAP.get(contract, 6)
    return float(monthly_charges) * exp_tenure

def risk_bucket(prob: float):
    if prob >= 0.70: return "High"
    if prob >= 0.40: return "Medium"
    return "Low"

def predict_single(pipe, payload: dict) -> dict:
    X = pd.DataFrame([payload])
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        prob = float(pipe.predict_proba(X)[:, 1][0])
    else:
        prob = float(pipe.predict(X)[0])
    pred = int(prob >= OPERATING_THRESHOLD)
    return {"prob": prob, "pred": pred, "threshold": OPERATING_THRESHOLD}

# --- Feature name recovery & pretty-printing for the explainer ---
def _get_feature_names(preprocessor) -> list:
    # sklearn >= 1.0 has get_feature_names_out; fall back if needed
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if hasattr(trans, "named_steps") and "ohe" in trans.named_steps:
                ohe = trans.named_steps["ohe"]
                cat_names = ohe.get_feature_names_out(cols)
                names.extend(cat_names.tolist())
            else:
                names.extend(cols if isinstance(cols, list) else list(cols))
        return names

def _nice_col(col_raw: str) -> str:
    key = col_raw.strip()
    if key in COLUMN_TITLE_MAP:
        return COLUMN_TITLE_MAP[key]
    pretty = re.sub(r"[_]+", " ", key)
    pretty = re.sub(r"(?<!^)(?=[A-Z])", " ", pretty).strip()
    return pretty[:1].upper() + pretty[1:]

def pretty_feature_name(raw: str) -> str:
    """
    Turn pipeline feature names into readable labels.
    Handles:
      - "num__MonthlyCharges" -> "Monthly Charges"
      - "cat__Contract_Two year" -> "Contract: Two year"
      - "cat__ohe__PaperlessBilling_No" -> "Paperless Billing: No"
      - case-insensitive variants like "cat_paperlessbilling_no"
    """
    s = str(raw)
    s_lower = s.lower()

    if s_lower.startswith("num__"):
        base = s[len("num__"):]
        return _nice_col(base)

    m = re.match(r"(?i)^(?:cat__)(?:ohe__)?([^_]+)_(.*)$", s)
    if m:
        col, cat = m.groups()
        return f"{_nice_col(col)}: {cat}"

    m2 = re.match(r"(?i)^(?:cat__)(.*)$", s)
    if m2 and "_" in m2.group(1):
        rest = m2.group(1)
        parts = rest.split("_", 1)
        if len(parts) == 2:
            col, cat = parts
            return f"{_nice_col(col)}: {cat}"

    s = re.sub(r"(?i)^(num__|cat__|ohe__)", "", s)
    return _nice_col(s)

def explain_logreg_prediction(pipe, payload: dict, top_k: int = 5):
    """
    Returns:
      - intercept
      - contributions: DataFrame with [feature, value, coef, logit_contrib, odds_ratio_contrib]
      - top_pos / top_neg
    logit(p) = intercept + sum_i (coef_i * x_i)
    """
    X = pd.DataFrame([payload])

    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    if not hasattr(model, "coef_"):
        return None

    X_pre = pre.transform(X)
    x_vec = X_pre.toarray().ravel() if hasattr(X_pre, "toarray") else np.asarray(X_pre).ravel()

    feat_names = _get_feature_names(pre)
    coefs = model.coef_.ravel()
    intercept = float(model.intercept_[0])

    n = min(len(coefs), len(x_vec), len(feat_names))
    coefs, x_vec, feat_names = coefs[:n], x_vec[:n], feat_names[:n]

    logit_contrib = coefs * x_vec
    df = pd.DataFrame({
        "feature": feat_names,
        "value": x_vec,
        "coef": coefs,
        "logit_contrib": logit_contrib,
    })
    df["odds_ratio_contrib"] = np.exp(df["logit_contrib"])

    top_pos = df.sort_values("logit_contrib", ascending=False).head(top_k)
    top_neg = df.sort_values("logit_contrib", ascending=True).head(top_k)

    return {
        "intercept": intercept,
        "contributions": df,
        "top_pos": top_pos,
        "top_neg": top_neg,
    }

def _format_or(or_val: float) -> str:
    pct = (or_val - 1.0) * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{or_val:.2f} ({sign}{pct:.0f}%)"

# --- Cached calibrated model built on validation (underscore args = no hashing) ---
@st.cache_resource
def build_calibrated_model_from_validation(_pipe, _val_df: pd.DataFrame, method: str = "isotonic"):
    """
    Fits an isotonic calibration layer using the validation split (cv='prefit').
    We underscore args so Streamlit won't try to hash them.
    """
    pipe, val_df = _pipe, _val_df

    pre = pipe.named_steps["pre"]
    base = pipe.named_steps["model"]
    if not hasattr(base, "predict_proba"):
        raise ValueError("Calibration requires a probabilistic classifier.")

    y_val = val_df["ChurnFlag"].astype(int).values
    X_val = val_df.drop(columns=["ChurnFlag"])
    X_val_pre = pre.transform(X_val)

    calib = CalibratedClassifierCV(base_estimator=base, cv="prefit", method=method)
    calib.fit(X_val_pre, y_val)
    return {"pre": pre, "calib": calib}

def calibrated_predict_proba(calib_bundle, X_df: pd.DataFrame) -> np.ndarray:
    pre = calib_bundle["pre"]
    calib = calib_bundle["calib"]
    X_pre = pre.transform(X_df)
    return calib.predict_proba(X_pre)[:, 1]

def pretty_metrics_row(name, m):
    return {
        "Set": name,
        "Precision": round(m.get("precision", float("nan")), 3),
        "Recall": round(m.get("recall", float("nan")), 3),
        "F1": round(m.get("f1", float("nan")), 3),
        "AUC": round(m.get("auc", float("nan")), 3),
        "Thr": round(m.get("used_threshold", m.get("chosen_threshold", 0.5)), 2),
        "TP": m.get("tp", None), "FP": m.get("fp", None),
        "TN": m.get("tn", None), "FN": m.get("fn", None),
    }

# ---------------------------------------------------------------
# Sidebar — About
# ---------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "Predict customer churn using a trained Logistic Regression pipeline "
        "with a recall-first operating threshold selected on validation."
    )
    st.write(
        "CLV = **Monthly Charges × Expected Tenure** "
        "(Month-to-month=6, One year=12, Two year=24 months)."
    )
    st.caption("Data: IBM Telco Customer Churn. Built with Streamlit.")

# ---------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------
st.title("Customer Churn Prediction & CLV")
st.caption(
    "Predict churn risk, estimate value (CLV), and prioritize retention. "
    "The decision threshold is fixed from validation (recall-first)."
)

tabs = st.tabs(["Predict", "Model Performance", "CLV Overview"])

# ===============================================================
# Tab 1 — Predict (with Why-this-prediction explainer)
# ===============================================================
with tabs[0]:
    st.subheader("Enter Customer Details")

    # Presets for quick demo
    colp1, colp2 = st.columns([1, 1])
    with colp1:
        if st.button("Use Edge-Case High-Risk Preset"):
            st.session_state.update({
                "tenure": 2, "monthly": 105.0, "contract": "Month-to-month",
                "internet_service": "Fiber optic", "payment_method": "Electronic check",
                "senior": True, "paperless": True, "gender": "Female",
                "phone": False, "multiline": "No phone service",
                "onsec": False, "onbak": False, "devprot": False, "techsup": False,
                "stv": False, "smov": False
            })
    with colp2:
        if st.button("Use Low-Risk Preset"):
            st.session_state.update({
                "tenure": 24, "monthly": 60.0, "contract": "Two year",
                "internet_service": "DSL", "payment_method": "Bank transfer (automatic)",
                "senior": False, "paperless": False, "gender": "Male",
                "phone": True, "multiline": "Yes",
                "onsec": True, "onbak": True, "devprot": True, "techsup": True,
                "stv": True, "smov": True
            })

    # Form prevents reruns on each widget change
    with st.form("predict_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.slider("Tenure (months)", 0, 72, st.session_state.get("tenure", 2))
            monthly = st.slider("Monthly Charges ($)", 0, 200, int(st.session_state.get("monthly", 105.0)))
        with c2:
            contract_opts = ["Month-to-month", "One year", "Two year"]
            contract = st.selectbox("Contract", contract_opts,
                                    index=contract_opts.index(st.session_state.get("contract","Month-to-month")))
            internet_opts = ["No", "DSL", "Fiber optic"]
            internet_service = st.selectbox("Internet Service", internet_opts,
                                    index=internet_opts.index(st.session_state.get("internet_service","No")))
            pm_opts = ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
            payment_method = st.selectbox("Payment Method", pm_opts,
                                    index=pm_opts.index(st.session_state.get("payment_method","Electronic check")))
        with c3:
            senior = st.toggle("Senior Citizen", value=st.session_state.get("senior", False))
            paperless = st.toggle("Paperless Billing", value=st.session_state.get("paperless", False))
            gender = st.selectbox("Gender", ["Female","Male"],
                                  index=["Female","Male"].index(st.session_state.get("gender","Female")))

        st.divider()
        st.markdown("##### Services")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            phone = st.toggle("Phone Service", value=st.session_state.get("phone", False))
            multiline = st.selectbox("Multiple Lines", ["No phone service","No","Yes"],
                                     index=["No phone service","No","Yes"].index(st.session_state.get("multiline","No phone service")))
        with s2:
            onsec = st.toggle("Online Security", value=st.session_state.get("onsec", False))
            onbak = st.toggle("Online Backup", value=st.session_state.get("onbak", False))
        with s3:
            devprot = st.toggle("Device Protection", value=st.session_state.get("devprot", False))
            techsup = st.toggle("Tech Support", value=st.session_state.get("techsup", False))
        with s4:
            stv = st.toggle("Streaming TV", value=st.session_state.get("stv", False))
            smov = st.toggle("Streaming Movies", value=st.session_state.get("smov", False))

        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    if submitted:
        # Engineer features exactly like training
        total_charges = monthly * tenure if tenure > 0 else 0.0
        services_count = (
            int(phone) + int(internet_service != "No") + int(onsec) + int(onbak) +
            int(devprot) + int(techsup) + int(stv) + int(smov)
        )
        denom = max(1.0, tenure * monthly)
        monthly_to_total_ratio = total_charges / denom
        internet_no_tech_support = 1 if (internet_service != "No" and not techsup) else 0
        tenure_bucket = "0-6m" if tenure <= 6 else "6-12m" if tenure <= 12 else "12-24m" if tenure <= 24 else "24m+"
        clv_est = compute_clv(monthly, contract)

        payload = {
            "tenure": int(tenure),
            "MonthlyCharges": float(monthly),
            "TotalCharges": float(total_charges),
            "CLV": float(clv_est),
            "services_count": int(services_count),
            "monthly_to_total_ratio": float(monthly_to_total_ratio),
            "internet_no_tech_support": int(internet_no_tech_support),
            "gender": gender,
            "SeniorCitizen": int(senior),
            "Partner": "No",
            "Dependents": "No",
            "PhoneService": "Yes" if phone else "No",
            "MultipleLines": multiline,
            "InternetService": internet_service,
            "OnlineSecurity": "Yes" if onsec else "No",
            "OnlineBackup": "Yes" if onbak else "No",
            "DeviceProtection": "Yes" if devprot else "No",
            "TechSupport": "Yes" if techsup else "No",
            "StreamingTV": "Yes" if stv else "No",
            "StreamingMovies": "Yes" if smov else "No",
            "Contract": contract,
            "PaperlessBilling": "Yes" if paperless else "No",
            "PaymentMethod": payment_method,
            "tenure_bucket": tenure_bucket,
        }

        try:
            pipe = load_best_pipeline()
            result = predict_single(pipe, payload)
            prob, pred = result["prob"], result["pred"]
            risk = risk_bucket(prob)

            # Result card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            cA, cB = st.columns([1, 1])
            with cA:
                st.metric("Churn probability", f"{prob*100:.1f}%")
                st.progress(min(max(prob, 0.0), 1.0))
            with cB:
                badge_cls = "badge-high" if risk == "High" else "badge-med" if risk == "Medium" else "badge-low"
                st.markdown(f"**Decision (@thr={OPERATING_THRESHOLD:.2f}):** {'Churn (1)' if pred==1 else 'No Churn (0)'}")
                st.markdown(f'<span class="badge {badge_cls}">Risk: {risk}</span>', unsafe_allow_html=True)
                st.write(f"**Estimated CLV:** ${clv_est:,.2f}  \n(= Monthly Charges × Expected Tenure)")
            st.markdown("</div>", unsafe_allow_html=True)

            # Why this prediction? (per-feature contributions)
            expl = explain_logreg_prediction(pipe, payload, top_k=5)
            if expl is not None:
                st.markdown("##### Why this prediction?")
                st.caption("Top drivers are computed from logistic regression coefficients on the one-hot/processed features.")
                c1, c2 = st.columns(2)

                top_pos = expl["top_pos"].copy()
                top_pos["Feature"] = top_pos["feature"].apply(pretty_feature_name)
                top_pos["Odds×"] = top_pos["odds_ratio_contrib"].apply(_format_or)
                top_pos = top_pos.rename(columns={"logit_contrib": "Log-odds Δ"})[
                    ["Feature", "Log-odds Δ", "Odds×"]
                ]

                top_neg = expl["top_neg"].copy()
                top_neg["Feature"] = top_neg["feature"].apply(pretty_feature_name)
                top_neg["Odds×"] = top_neg["odds_ratio_contrib"].apply(_format_or)
                top_neg = top_neg.rename(columns={"logit_contrib": "Log-odds Δ"})[
                    ["Feature", "Log-odds Δ", "Odds×"]
                ]

                with c1:
                    st.write("**Pushes risk UP**")
                    st.dataframe(top_pos.style.format({"Log-odds Δ": "{:.3f}"}), use_container_width=True)
                with c2:
                    st.write("**Pushes risk DOWN**")
                    st.dataframe(top_neg.style.format({"Log-odds Δ": "{:.3f}"}), use_container_width=True)

                st.caption("Interpretation: each row shows the contribution of a processed feature to the log-odds. "
                           "`Odds×` is the multiplicative impact on odds for this input (e.g., 1.30 = +30%).")

        except Exception as e:
            st.error(str(e))

# ===============================================================
# Tab 2 — Model Performance (with optional calibration)
# ===============================================================
with tabs[1]:
    st.subheader("Model Performance")

    try:
        metrics = load_metrics()
        val_row  = pretty_metrics_row("Validation (chosen)", metrics["val"])
        test_row = pretty_metrics_row("Test (frozen thr.)", metrics["test"])
        dfm = pd.DataFrame([val_row, test_row])

        st.markdown('<div class="card section">', unsafe_allow_html=True)
        st.dataframe(
            dfm.rename(columns={
                "Precision": "Precision",
                "Recall": "Recall",
                "F1": "F1-score",
                "AUC": "ROC-AUC",
                "Thr": "Threshold",
            }),
            use_container_width=True, hide_index=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Validation chose the threshold to satisfy recall ≥ 0.60 and maximize F1 among those thresholds. "
                   "We froze that threshold and reported metrics on test.")

        # ----- Optional: compare calibrated probabilities -----
        st.markdown("##### Optional: Compare calibrated probabilities")
        use_calib = st.toggle(
            "Use isotonic calibration (fit on validation, compare on test)", value=False,
            help=("Fits an isotonic calibration layer on the validation split (cv='prefit') and "
                  "applies it to the test set for comparison. This adjusts probabilities, not the chosen threshold.")
        )

        if use_calib:
            train_df, val_df, test_df = load_processed_splits()
            pipe = load_best_pipeline()
            bundle = build_calibrated_model_from_validation(pipe, val_df, method="isotonic")

            y_test = test_df["ChurnFlag"].astype(int).values
            X_test = test_df.drop(columns=["ChurnFlag"])
            probs_cal = calibrated_predict_proba(bundle, X_test)
            thr = metrics["test"]["used_threshold"]

            preds_cal = (probs_cal >= thr).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(y_test, preds_cal, average="binary", zero_division=0)
            auc = roc_auc_score(y_test, probs_cal)
            tn, fp, fn, tp = confusion_matrix(y_test, preds_cal).ravel()

            calibrated_row = {
                "Set": "Test (calibrated, same thr.)",
                "Precision": round(float(p), 3),
                "Recall": round(float(r), 3),
                "F1": round(float(f1), 3),
                "AUC": round(float(auc), 3),
                "Thr": round(float(thr), 2),
                "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            }

            comp = pd.DataFrame([test_row, calibrated_row]).rename(
                columns={"F1": "F1-score", "AUC": "ROC-AUC", "Thr": "Threshold"}
            )
            st.markdown('<div class="card section">', unsafe_allow_html=True)
            st.dataframe(comp, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.caption("Calibration aims to make probabilities better match observed frequencies. "
                       "AUC may stay similar; precision/recall at a fixed threshold can shift.")

        st.write("**Confusion Matrix (Test)**")
        st.write(f"TP: {metrics['test']['tp']} | FP: {metrics['test']['fp']} | "
                 f"TN: {metrics['test']['tn']} | FN: {metrics['test']['fn']}")
    except Exception as e:
        st.error(str(e))

# ===============================================================
# Tab 3 — CLV Overview
# ===============================================================
with tabs[2]:
    st.subheader("CLV Segments & Churn")

    st.markdown('<div class="card section">', unsafe_allow_html=True)
    if os.path.exists(CLV_QUARTILE_PLOT):
        st.image(CLV_QUARTILE_PLOT, caption="Churn rate by CLV quartile (train)")
    else:
        st.warning("CLV plot not found. Run `python -m src.clv_analysis` to generate it.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Optional stats preview (friendlier labels)
    try:
        train, _, _ = load_processed_splits()
        if "CLV" in train.columns:
            stats = train["CLV"].describe()[["count","mean","std","min","25%","50%","75%","max"]]
            st.markdown('<div class="card section">', unsafe_allow_html=True)
            st.write("**CLV Summary (train)**")
            st.dataframe(stats.to_frame("CLV").rename_axis("Statistic"), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.info("Processed splits unavailable for preview. " + str(e))

    st.write("**Takeaways**")
    st.markdown("""
- Highest churn risk: **High**, then **Medium** CLV segments.
- **Premium** segment has much lower churn → maintain with loyalty offers.
- Prioritize retention offers for **High → Medium**; test incentives on **Low** selectively.
""")
