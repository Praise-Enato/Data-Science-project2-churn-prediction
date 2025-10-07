# app.py
# Telco Customer Churn Prediction & CLV — Modern UI + Retention + Global Importance
# - Hero header, pill tabs, glass cards
# - Per-prediction explanation (LogReg)
# - Global importance: SHAP summary (trees) or fallback (tree importances / LogReg coefs)
# - Version-safe isotonic calibration toggle
# - CLV inline fallback (works on Streamlit Cloud)
# - Retention KPI tiles + strategies

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

# Optional SHAP (trees)
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# --------------------------- Page & Global Styles ---------------------------
st.set_page_config(page_title="Telco Customer Churn & CLV", page_icon="📊", layout="wide")

st.markdown("""
<style>
:root {
  --bg1: #0b1220;  --bg2: #0f1830;  --accent: #7C4DFF;  --accent2: #00E5A5;  --accent3: #FF6B6B;
  --text: #E6EAF2; --muted: #9AA6BF; --border: rgba(255,255,255,.06);
  --shadow: 0 6px 28px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04);
}
.stApp {
  background: radial-gradient(1200px 600px at 5% -10%, #1a2b5b22, transparent),
              radial-gradient(1200px 600px at 105% 10%, #7C4DFF22, transparent),
              linear-gradient(180deg, var(--bg1) 0%, #0a111e 100%);
  color: var(--text);
}
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Hero */
.hero {
  border-radius: 22px; padding: 28px 28px 24px;
  background: radial-gradient(1200px 500px at 0% -20%, #00E5A533, transparent 60%),
              radial-gradient(900px 400px at 110% 0%, #7C4DFF33, transparent 60%),
              linear-gradient(180deg, var(--bg2), #0f1730);
  border: 1px solid var(--border); box-shadow: var(--shadow);
}
.hero h1 {
  margin: 0 0 .25rem 0; font-size: clamp(28px, 3vw, 40px); line-height: 1.15;
  background: linear-gradient(90deg, #B3A7FF 0%, #7C4DFF 45%, #00E5A5 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent; letter-spacing: .4px;
}
.hero p { margin: .15rem 0 0 0; color: var(--muted); font-size: 15px; }

/* Pill tabs */
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] {
  color: #C9D2EA; background: #0e172d; border: 1px solid var(--border);
  border-radius: 999px; padding: 10px 18px; box-shadow: inset 0 1px 0 rgba(255,255,255,.06);
}
.stTabs [aria-selected="true"] { color: #0b1220 !important; background: linear-gradient(90deg, #7C4DFF, #00E5A5); border-color: transparent !important; }

/* Cards & badges */
.card { background: linear-gradient(180deg, #0f1830, #0d162b); border: 1px solid var(--border);
  border-radius: 18px; padding: 16px 16px; box-shadow: var(--shadow); }
.section { margin-top: .75rem; }
.badge { display:inline-block; padding:.28rem .6rem; border-radius:999px; font-size:.82rem; border:1px solid var(--border); background:#0c1a14; color:#7ff5c9; }
.badge.med { background:#2e1d0f; color:#ffd88a; } .badge.high { background:#2a0f12; color:#ff9aa5; }

/* Tables + buttons + metrics */
.dataframe, .stDataFrame { font-size: .93rem; } thead th { background:#0d162b !important; }
.stButton>button { border-radius: 999px; border: 1px solid var(--border); background: #0e1730; color: var(--text); padding: .5rem .9rem; }
.stButton>button:hover { border-color:#1f2a44; background:#101c3b; }
[data-testid="stMetricValue"] { font-weight: 800; }

/* KPI tiles + strategies */
.kpi { border-radius: 18px; padding: 18px 16px; color: #fff; box-shadow: 0 10px 28px rgba(0,0,0,.30), inset 0 1px 0 rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.08); text-align: center; }
.kpi h4 { margin: 0 0 .35rem 0; font-size: .95rem; opacity: .9; font-weight: 600; }
.kpi .v { font-size: 2.0rem; font-weight: 800; margin: .1rem 0 .2rem; }
.kpi .sub { font-size: .85rem; opacity:.95; }
.kpi.red { background: linear-gradient(180deg,#ff6b6b,#b83a3a); }
.kpi.amber { background: linear-gradient(180deg,#ffb75d,#c98531); }
.kpi.green { background: linear-gradient(180deg,#34d399,#1f8f6e); }
.priority { border-radius: 12px; padding: 10px 14px; margin-top: 8px; background: #3a0f12; color:#ffb3bb; border:1px solid #6c2227; font-weight: 600; }
.reco li { margin: .35rem 0; }
</style>
""", unsafe_allow_html=True)

# --------------------------- Constants ---------------------------
MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"
FIG_DIR = "figures"

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "logreg_pipeline.pkl")
METRICS_PATH   = os.path.join(MODELS_DIR, "logreg_metrics.json")
CLV_QUARTILE_PLOT = os.path.join(FIG_DIR, "churn_rate_by_clv_quartile.png")

OPERATING_THRESHOLD = 0.59
EXPECTED_TENURE_MAP = {"Month-to-month": 6, "One year": 12, "Two year": 24}

COLUMN_TITLE_MAP = {
    "MonthlyCharges": "Monthly Charges", "TotalCharges": "Total Charges",
    "SeniorCitizen": "Senior Citizen", "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Method", "InternetService": "Internet Service",
    "DeviceProtection": "Device Protection", "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup", "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming Movies",
    "MultipleLines": "Multiple Lines", "PhoneService": "Phone Service",
    "tenure_bucket": "Tenure Bucket", "Contract": "Contract",
    "Partner": "Partner", "Dependents": "Dependents", "gender": "Gender",
    "CLV": "CLV", "services_count": "Services Count",
    "monthly_to_total_ratio": "Monthly/Total Ratio",
    "internet_no_tech_support": "Internet w/o Tech Support", "tenure": "Tenure",
}

# --------------------------- Cached loaders ---------------------------
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
    val_p   = os.path.join(PROCESSED_DIR, "val.csv")
    test_p  = os.path.join(PROCESSED_DIR, "test.csv")
    if not (os.path.exists(train_p) and os.path.exists(val_p) and os.path.exists(test_p)):
        raise FileNotFoundError(
            f"Processed splits not found in {PROCESSED_DIR}. "
            "Generate them with: python -m src.data_prep"
        )
    return (pd.read_csv(train_p), pd.read_csv(val_p), pd.read_csv(test_p))

# --------------------------- Utilities ---------------------------
def compute_clv(monthly_charges: float, contract: str) -> float:
    return float(monthly_charges) * EXPECTED_TENURE_MAP.get(contract, 6)

def risk_bucket(prob: float):
    return "High" if prob >= 0.70 else "Medium" if prob >= 0.40 else "Low"

def predict_single(pipe, payload: dict) -> dict:
    X = pd.DataFrame([payload])
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        prob = float(pipe.predict_proba(X)[:, 1][0])
    else:
        prob = float(pipe.predict(X)[0])
    return {"prob": prob, "pred": int(prob >= OPERATING_THRESHOLD), "threshold": OPERATING_THRESHOLD}

def _get_feature_names(preprocessor) -> list:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if hasattr(trans, "named_steps") and "ohe" in trans.named_steps:
                ohe = trans.named_steps["ohe"]
                names.extend(ohe.get_feature_names_out(cols).tolist())
            else:
                names.extend(cols if isinstance(cols, list) else list(cols))
        return names

def _nice_col(col_raw: str) -> str:
    key = col_raw.strip()
    if key in COLUMN_TITLE_MAP: return COLUMN_TITLE_MAP[key]
    pretty = re.sub(r"[_]+", " ", key)
    pretty = re.sub(r"(?<!^)(?=[A-Z])", " ", pretty).strip()
    return pretty[:1].upper() + pretty[1:]

def pretty_feature_name(raw: str) -> str:
    s = str(raw); s_lower = s.lower()
    if s_lower.startswith("num__"): return _nice_col(s[len("num__"):])
    m = re.match(r"(?i)^(?:cat__)(?:ohe__)?([^_]+)_(.*)$", s)
    if m:  col, cat = m.groups(); return f"{_nice_col(col)}: {cat}"
    m2 = re.match(r"(?i)^(?:cat__)(.*)$", s)
    if m2 and "_" in m2.group(1):
        col, cat = m2.group(1).split("_", 1); return f"{_nice_col(col)}: {cat}"
    s = re.sub(r"(?i)^(num__|cat__|ohe__)", "", s); return _nice_col(s)

def explain_logreg_prediction(pipe, payload: dict, top_k: int = 5):
    X = pd.DataFrame([payload])
    pre = pipe.named_steps["pre"]; model = pipe.named_steps["model"]
    if not hasattr(model, "coef_"): return None
    X_pre = pre.transform(X)
    x_vec = X_pre.toarray().ravel() if hasattr(X_pre, "toarray") else np.asarray(X_pre).ravel()
    feat_names = _get_feature_names(pre)
    coefs = model.coef_.ravel()
    n = min(len(coefs), len(x_vec), len(feat_names))
    df = pd.DataFrame({"feature": feat_names[:n], "value": x_vec[:n], "coef": coefs[:n]})
    df["logit_contrib"] = df["coef"] * df["value"]
    df["odds_ratio_contrib"] = np.exp(df["logit_contrib"])
    top_pos = df.sort_values("logit_contrib", ascending=False).head(top_k)
    top_neg = df.sort_values("logit_contrib", ascending=True).head(top_k)
    return {"top_pos": top_pos, "top_neg": top_neg}

def _format_or(or_val: float) -> str:
    pct = (or_val - 1.0) * 100.0
    return f"{or_val:.2f} ({'+' if pct>=0 else ''}{pct:.0f}%)"

@st.cache_resource
def build_calibrated_model_from_validation(_pipe, _val_df: pd.DataFrame, method: str = "isotonic"):
    """Version-safe calibration; underscore args avoid Streamlit hashing issues."""
    pipe, val_df = _pipe, _val_df
    pre  = pipe.named_steps["pre"]
    base = pipe.named_steps["model"]
    if not hasattr(base, "predict_proba"):
        raise ValueError("Calibration requires a probabilistic classifier.")
    y_val = val_df["ChurnFlag"].astype(int).values
    X_val = val_df.drop(columns=["ChurnFlag"]); X_val_pre = pre.transform(X_val)
    try:
        calib = CalibratedClassifierCV(estimator=base, cv="prefit", method=method)
    except TypeError:
        calib = CalibratedClassifierCV(base_estimator=base, cv="prefit", method=method)
    calib.fit(X_val_pre, y_val)
    return {"pre": pre, "calib": calib}

def calibrated_predict_proba(bundle, X_df: pd.DataFrame) -> np.ndarray:
    X_pre = bundle["pre"].transform(X_df)
    return bundle["calib"].predict_proba(X_pre)[:, 1]

def pretty_metrics_row(name, m):
    return {
        "Set": name,
        "Precision": round(m.get("precision", float("nan")), 3),
        "Recall":    round(m.get("recall", float("nan")), 3),
        "F1":        round(m.get("f1", float("nan")), 3),
        "AUC":       round(m.get("auc", float("nan")), 3),
        "Thr":       round(m.get("used_threshold", m.get("chosen_threshold", 0.5)), 2),
        "TP": m.get("tp"), "FP": m.get("fp"), "TN": m.get("tn"), "FN": m.get("fn"),
    }

def render_clv_plot_inline():
    try:
        train, _, _ = load_processed_splits()
        if "CLV" not in train.columns or "ChurnFlag" not in train.columns:
            st.warning("CLV or ChurnFlag not in processed train data."); return
        quart = pd.qcut(train["CLV"], 4, labels=["Low","Medium","High","Premium"])
        tmp = train.assign(CLV_quartile=quart)
        agg = (tmp.groupby("CLV_quartile")["ChurnFlag"]
                 .agg(size="count", churn_rate="mean").reset_index())
        fig, ax = plt.subplots(figsize=(6,3.6))
        ax.bar(agg["CLV_quartile"], agg["churn_rate"], color="#7C4DFF")
        ax.set_ylim(0, max(.01, agg["churn_rate"].max())*1.15)
        ax.set_ylabel("Churn rate"); ax.set_title("Churn rate by CLV quartile (train)")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not build CLV plot: {e}")

# ---------- Global Importance helpers ----------
def global_importance_logreg(pipe, top_n=25) -> pd.DataFrame:
    pre  = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["Feature","Importance"])
    names = _get_feature_names(pre)
    coefs = np.abs(model.coef_.ravel())
    n = min(len(names), len(coefs))
    df = pd.DataFrame({"FeatureRaw": names[:n], "Importance": coefs[:n]})
    df["Feature"] = df["FeatureRaw"].apply(pretty_feature_name)
    return df.sort_values("Importance", ascending=False).head(top_n)[["Feature","Importance"]]

def global_importance_tree_feature_importances(pipe, top_n=25) -> pd.DataFrame:
    pre   = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Feature","Importance"])
    names = _get_feature_names(pre)
    imps  = model.feature_importances_
    n = min(len(names), len(imps))
    df = pd.DataFrame({"FeatureRaw": names[:n], "Importance": imps[:n]})
    df["Feature"] = df["FeatureRaw"].apply(pretty_feature_name)
    return df.sort_values("Importance", ascending=False).head(top_n)[["Feature","Importance"]]

def tree_shap_summary_fig(pipe, sample_df: pd.DataFrame, max_display=25):
    if not _HAS_SHAP:
        return None
    pre   = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    X_pre = pre.transform(sample_df)
    feat_names = _get_feature_names(pre)
    X_d = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_d)
    except Exception:
        return None
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_vals_use = shap_vals[1]
    else:
        shap_vals_use = shap_vals
    fig = plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_vals_use, X_d, feature_names=feat_names, plot_type="dot",
        show=False, max_display=max_display
    )
    plt.tight_layout()
    return fig

# ---------- Retention logic ----------
@st.cache_data
def clv_quartiles_from_train():
    train, _, _ = load_processed_splits()
    if "CLV" not in train.columns:
        raise ValueError("CLV column not found in train split.")
    qs = np.quantile(train["CLV"].values, [0.25, 0.5, 0.75])
    return float(qs[0]), float(qs[1]), float(qs[2])

def clv_segment(clv_value: float) -> str:
    q1, q2, q3 = clv_quartiles_from_train()
    if clv_value < q1:  return "Low"
    if clv_value < q2:  return "Medium"
    if clv_value < q3:  return "High"
    return "Premium"

def prediction_confidence(prob: float) -> tuple[str, str]:
    d = abs(prob - 0.5)
    if d >= 0.30: return "High",  "100% Certain"
    if d >= 0.20: return "Medium","~80% Certain"
    return "Low",  "~60% Certain"

def retention_recommendations(risk: str, value_seg: str, payload: dict) -> tuple[list[str], str]:
    bullets, priority = [], ""
    if risk == "High" and value_seg in {"High", "Premium"}:
        bullets += [
            "Executive Intervention: personal call from account manager.",
            "Premium Offer: 20–30% discount or value-add bundle (security/backup).",
            "Contract Upgrade: propose annual term with incentive.",
            "Support Enhancement: free tech support for 3–6 months.",
        ]
        priority = "HIGH PRIORITY: Contact within 24 hours"
    elif risk == "High":
        bullets += [
            "Winback Offer: time-limited discount (10–15%).",
            "Friction Removal: enable paperless & auto-pay with small credit.",
            "Service Fix: address frequent issues (tech support ticket).",
        ]
        priority = "PRIORITY: Contact within 48–72 hours"
    elif risk == "Medium" and value_seg in {"High", "Premium"}:
        bullets += [
            "Loyalty Perk: small monthly credit or feature add-on.",
            "Nudge: encourage paperless billing & auto-pay.",
            "Education: tips to use included services (security/backup).",
        ]
        priority = "Monitor & nudge this month"
    else:
        bullets += [
            "Maintain Satisfaction: periodic check-ins (quarterly).",
            "Upsell Light: non-intrusive recommendations.",
            "No heavy discounts needed.",
        ]
        priority = "No urgent action"

    if payload.get("PaperlessBilling") == "No":
        bullets.append("Activate Paperless Billing: small one-time bill credit.")
    if str(payload.get("PaymentMethod","")).startswith("Electronic"):
        bullets.append("Suggest Auto-Pay: reduce payment friction.")
    return bullets, priority

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("About")
    st.write("Predict customer churn using a trained Logistic Regression pipeline with a recall-first threshold.")
    st.write("CLV = **Monthly Charges × Expected Tenure** (6/12/24 months).")
    st.caption("Data: IBM Telco Customer Churn. Built with Streamlit.")
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------- Hero ---------------------------
st.markdown("""
<div class="hero">
  <h1>IBM Telco Customer Churn Prediction & CLV</h1>
  <p>An intelligent analytics app to predict churn, understand drivers, and optimize customer lifetime value.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------- Tabs ---------------------------
tabs = st.tabs(["Churn Prediction", "Model Performance", "CLV Overview"])

# ===================== Tab 1: Predict =====================
with tabs[0]:
    st.markdown('<div class="card section">', unsafe_allow_html=True)
    st.subheader("Customer Churn Prediction Engine")
    st.caption("Enter customer details to get instant predictions with AI-powered explanations.")

    colp1, colp2 = st.columns(2)
    with colp1:
        st.button("⚡ High-risk preset", key="hi",
                  on_click=lambda: st.session_state.update({
                      "tenure":2,"monthly":105,"contract":"Month-to-month","internet_service":"Fiber optic",
                      "payment_method":"Electronic check","senior":True,"paperless":True,"gender":"Female",
                      "phone":False,"multiline":"No phone service","onsec":False,"onbak":False,"devprot":False,
                      "techsup":False,"stv":False,"smov":False
                  }))
    with colp2:
        st.button("🛡️ Low-risk preset", key="lo",
                  on_click=lambda: st.session_state.update({
                      "tenure":24,"monthly":60,"contract":"Two year","internet_service":"DSL",
                      "payment_method":"Bank transfer (automatic)","senior":False,"paperless":False,"gender":"Male",
                      "phone":True,"multiline":"Yes","onsec":True,"onbak":True,"devprot":True,
                      "techsup":True,"stv":True,"smov":True
                  }))

    with st.form("predict_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure  = st.slider("Tenure (months)", 0, 72, st.session_state.get("tenure", 2))
            monthly = st.slider("Monthly Charges ($)", 0, 200, int(st.session_state.get("monthly", 105)))
        with c2:
            contract_opts = ["Month-to-month","One year","Two year"]
            contract = st.selectbox("Contract", contract_opts,
                        index=contract_opts.index(st.session_state.get("contract","Month-to-month")))
            internet_opts = ["No","DSL","Fiber optic"]
            internet_service = st.selectbox("Internet Service", internet_opts,
                        index=internet_opts.index(st.session_state.get("internet_service","No")))
            pm_opts = ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
            payment_method = st.selectbox("Payment Method", pm_opts,
                        index=pm_opts.index(st.session_state.get("payment_method","Electronic check")))
        with c3:
            senior    = st.toggle("Senior Citizen", value=st.session_state.get("senior", False))
            paperless = st.toggle("Paperless Billing", value=st.session_state.get("paperless", False))
            gender    = st.selectbox("Gender", ["Female","Male"],
                        index=["Female","Male"].index(st.session_state.get("gender","Female")))

        st.markdown("###### Services")
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

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        total_charges = monthly * tenure if tenure > 0 else 0.0
        services_count = (
            int(phone) + int(internet_service!="No") + int(onsec) + int(onbak) +
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
            "gender": gender, "SeniorCitizen": int(senior),
            "Partner":"No","Dependents":"No",
            "PhoneService": "Yes" if phone else "No",
            "MultipleLines": multiline, "InternetService": internet_service,
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
            res  = predict_single(pipe, payload)
            prob, pred = res["prob"], res["pred"]
            risk = risk_bucket(prob)
            badge = "badge high" if risk=="High" else "badge med" if risk=="Medium" else "badge"

            st.markdown('<div class="card section">', unsafe_allow_html=True)
            m1, m2 = st.columns([1,1])
            with m1:
                st.metric("Churn probability", f"{prob*100:.1f}%")
                st.progress(min(max(prob,0.0),1.0))
            with m2:
                st.markdown(f"**Decision (@thr={OPERATING_THRESHOLD:.2f}):** {'Churn (1)' if pred else 'No Churn (0)'}")
                st.markdown(f'<span class="{badge}">Risk: {risk}</span>', unsafe_allow_html=True)
                st.write(f"**Estimated CLV:** ${clv_est:,.2f}  \n(= Monthly Charges × Expected Tenure)")
            st.markdown('</div>', unsafe_allow_html=True)

            # KPI + Retention strategies
            value_seg = clv_segment(clv_est)
            conf_label, conf_text = prediction_confidence(prob)
            reco_bullets, priority_line = retention_recommendations(risk, value_seg, payload)

            k1, k2, k3 = st.columns([1,1,1])
            with k1:
                html = {
                    "High":   f"""<div class="kpi red"><h4>Churn Probability</h4><div class="v">{prob*100:.1f}%</div><div class="sub">HIGH RISK</div></div>""",
                    "Medium": f"""<div class="kpi amber"><h4>Churn Probability</h4><div class="v">{prob*100:.1f}%</div><div class="sub">MEDIUM RISK</div></div>""",
                    "Low":    f"""<div class="kpi green"><h4>Churn Probability</h4><div class="v">{prob*100:.1f}%</div><div class="sub">LOW RISK</div></div>""",
                }[risk]
                st.markdown(html, unsafe_allow_html=True)
            with k2:
                clv_class = {"Low":"red","Medium":"amber","High":"amber","Premium":"green"}[value_seg]
                seg_caption = {"Low":"Below Average","Medium":"Average","High":"Above Average","Premium":"Top Tier"}[value_seg]
                st.markdown(f"""
                <div class="kpi {clv_class}">
                  <h4>Customer Lifetime Value</h4>
                  <div class="v">${clv_est:,.0f}</div>
                  <div class="sub">{seg_caption}</div>
                </div>
                """, unsafe_allow_html=True)
            with k3:
                conf_class = {"High":"green","Medium":"amber","Low":"red"}[conf_label]
                st.markdown(f"""
                <div class="kpi {conf_class}">
                  <h4>Prediction Confidence</h4>
                  <div class="v">{conf_label}</div>
                  <div class="sub">{conf_text}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="card section">', unsafe_allow_html=True)
            st.markdown("### Recommended Retention Strategies")
            st.caption(f"Customer Segment: **{risk} Risk – {value_seg} Value**")
            st.markdown("<ul class='reco'>" + "".join([f"<li>{reco}</li>" for reco in reco_bullets]) + "</ul>", unsafe_allow_html=True)
            if priority_line:
                st.markdown(f"<div class='priority'>{priority_line}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Why this prediction?
            expl = explain_logreg_prediction(pipe, payload, top_k=5)
            if expl is not None:
                st.markdown('<div class="card section">', unsafe_allow_html=True)
                st.markdown("#### Why this prediction?")
                st.caption("Top drivers computed from logistic regression coefficients on the processed features.")
                c1, c2 = st.columns(2)
                top_pos = expl["top_pos"].copy()
                top_pos["Feature"] = top_pos["feature"].apply(pretty_feature_name)
                top_pos["Odds×"]   = top_pos["odds_ratio_contrib"].apply(_format_or)
                top_pos = top_pos.rename(columns={"logit_contrib":"Log-odds Δ"})[["Feature","Log-odds Δ","Odds×"]]
                top_neg = expl["top_neg"].copy()
                top_neg["Feature"] = top_neg["feature"].apply(pretty_feature_name)
                top_neg["Odds×"]   = top_neg["odds_ratio_contrib"].apply(_format_or)
                top_neg = top_neg.rename(columns={"logit_contrib":"Log-odds Δ"})[["Feature","Log-odds Δ","Odds×"]]
                with c1:
                    st.write("**Pushes risk UP**")
                    st.dataframe(top_pos.style.format({"Log-odds Δ":"{:.3f}"}), use_container_width=True)
                with c2:
                    st.write("**Pushes risk DOWN**")
                    st.dataframe(top_neg.style.format({"Log-odds Δ":"{:.3f}"}), use_container_width=True)
                st.caption("Interpretation: each row shows the contribution to the log-odds. "
                           "`Odds×` is the multiplicative impact on odds for this input (e.g., 1.30 = +30%).")
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(str(e))

    st.markdown('</div>', unsafe_allow_html=True)

# ===================== Tab 2: Model Performance =====================
with tabs[1]:
    st.markdown('<div class="card section">', unsafe_allow_html=True)
    st.subheader("Model Performance")
    try:
        metrics = load_metrics()
        dfm = pd.DataFrame([
            pretty_metrics_row("Validation (chosen)", metrics["val"]),
            pretty_metrics_row("Test (frozen thr.)", metrics["test"])
        ]).rename(columns={"F1":"F1-score","AUC":"ROC-AUC","Thr":"Threshold"})
        st.dataframe(dfm, use_container_width=True, hide_index=True)
        st.caption("Threshold selected on validation to satisfy recall ≥ 0.60 and maximize F1; frozen for test.")
    except Exception as e:
        st.error(str(e))
    st.markdown('</div>', unsafe_allow_html=True)

    # Comparison table
    mm_path = os.path.join(MODELS_DIR, "metrics_all.json")
    if os.path.exists(mm_path):
        with open(mm_path) as f: mm = json.load(f)
        order = ["logreg","rf","xgb"]; label = {"logreg":"LOGREG","rf":"RANDOM FOREST","xgb":"XGBOOST"}
        rows=[]
        for m in order:
            if m in mm and "test" in mm[m]:
                t = mm[m]["test"]
                rows.append({
                    "Model": label[m],
                    "Precision": round(float(t.get("precision",np.nan)),3),
                    "Recall": round(float(t.get("recall",np.nan)),3),
                    "F1-score": round(float(t.get("f1",np.nan)),3),
                    "ROC-AUC": round(float(t.get("auc",np.nan)),3),
                    "Threshold": round(float(t.get("used_threshold",0.5)),2),
                })
        if rows:
            st.markdown('<div class="card section">', unsafe_allow_html=True)
            st.write("**Model comparison (Test)**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ROC overlay
    rc_path = os.path.join(MODELS_DIR, "roc_curves_test.json")
    if os.path.exists(rc_path):
        with open(rc_path) as f: rc = json.load(f)
        if rc:
            fig = plt.figure()
            for name, pts in rc.items():
                plt.plot(pts["fpr"], pts["tpr"], label=name.upper())
            plt.plot([0,1],[0,1], ls="--", lw=1)
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("ROC (Test)"); plt.legend()
            st.markdown('<div class="card section">', unsafe_allow_html=True)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    # Calibration toggle
    st.markdown('<div class="card section">', unsafe_allow_html=True)
    st.markdown("#### Optional: Compare calibrated probabilities")
    use_calib = st.toggle(
        "Use isotonic calibration (fit on validation, compare on test)",
        value=False,
        help="Fits isotonic calibration on validation (cv='prefit'), applies to test. Adjusts probabilities only."
    )
    if use_calib:
        try:
            train_df, val_df, test_df = load_processed_splits()
            pipe = load_best_pipeline()
            bundle = build_calibrated_model_from_validation(pipe, val_df, method="isotonic")
            y_test = test_df["ChurnFlag"].astype(int).values
            X_test = test_df.drop(columns=["ChurnFlag"])
            probs_cal = calibrated_predict_proba(bundle, X_test)
            thr = metrics["test"]["used_threshold"]
            preds = (probs_cal >= thr).astype(int)
            p,r,f1,_ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
            auc = roc_auc_score(y_test, probs_cal)
            tn,fp,fn,tp = confusion_matrix(y_test, preds).ravel()
            comp = pd.DataFrame([{
                "Set":"Test (calibrated, same thr.)", "Precision":round(float(p),3),
                "Recall":round(float(r),3), "F1-score":round(float(f1),3),
                "ROC-AUC":round(float(auc),3), "Threshold":round(float(thr),2),
                "TP":int(tp), "FP":int(fp), "TN":int(tn), "FN":int(fn)
            }])
            st.dataframe(comp, use_container_width=True, hide_index=True)
            st.caption("Calibration aims to match predicted probabilities with observed frequencies.")
        except Exception as e:
            st.error(str(e))
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Global Importance ----------------
    st.markdown('<div class="card section">', unsafe_allow_html=True)
    st.subheader("Global Importance")

    pipe = None
    try:
        pipe = load_best_pipeline()
    except Exception as e:
        st.info("Model pipeline not available for importance plots. " + str(e))

    if pipe is not None:
        model_name = pipe.named_steps["model"].__class__.__name__.upper()
        is_tree_model = any(k in model_name for k in ["FOREST", "XGB", "BOOST", "TREE"])

        try:
            _, _, test_df = load_processed_splits()
            X_test = test_df.drop(columns=["ChurnFlag"])
            sample_n = min(1000, len(X_test))
            X_sample = X_test.sample(sample_n, random_state=42) if len(X_test) > sample_n else X_test
        except Exception as e:
            X_sample = None
            st.caption("Note: could not load test set sample for SHAP — " + str(e))

        if is_tree_model and _HAS_SHAP and X_sample is not None:
            fig = tree_shap_summary_fig(pipe, X_sample, max_display=25)
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
                st.caption("SHAP summary: global explanation of feature impact (positive class).")
            else:
                st.info("SHAP could not be computed; showing fallback feature importances instead.")
                df_imp = global_importance_tree_feature_importances(pipe, top_n=25)
                st.dataframe(df_imp, use_container_width=True, hide_index=True)
        else:
            if "LOGISTIC" in model_name or "LOGISTICREGRESSION" in model_name or "LOGREG" in model_name:
                df_imp = global_importance_logreg(pipe, top_n=25)
                st.dataframe(df_imp, use_container_width=True, hide_index=True)
                st.caption("Logistic Regression: absolute coefficient magnitude as global importance.")
            else:
                df_imp = global_importance_tree_feature_importances(pipe, top_n=25)
                if not df_imp.empty:
                    st.dataframe(df_imp, use_container_width=True, hide_index=True)
                    st.caption("Fallback: model.feature_importances_.")
                else:
                    st.info("No global importance method available for this model.")
    else:
        st.info("Load a model to view global importance.")
    st.markdown('</div>', unsafe_allow_html=True)

# ===================== Tab 3: CLV Overview =====================
with tabs[2]:
    st.markdown('<div class="card section">', unsafe_allow_html=True)
    st.subheader("CLV Segments & Churn")
    if os.path.exists(CLV_QUARTILE_PLOT):
        st.image(CLV_QUARTILE_PLOT, caption="Churn rate by CLV quartile (train)")
    else:
        st.info("CLV plot file not found. Building it on the fly…")
        render_clv_plot_inline()
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        train, _, _ = load_processed_splits()
        if "CLV" in train.columns:
            stats = train["CLV"].describe()[["count","mean","std","min","25%","50%","75%","max"]]
            st.markdown('<div class="card section">', unsafe_allow_html=True)
            st.write("**CLV Summary (train)**")
            st.dataframe(stats.to_frame("CLV").rename_axis("Statistic"), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.info("Processed splits unavailable for preview. " + str(e))

    st.markdown('<div class="card section">', unsafe_allow_html=True)
    st.write("**Takeaways**")
    st.markdown("""
- Highest churn risk typically sits in **High** then **Medium** CLV segments.
- **Premium** CLV segment shows much lower churn → maintain with loyalty and service quality.
- Prioritize retention offers for **High → Medium**; trial incentives for **Low** selectively.
""")
    st.markdown('</div>', unsafe_allow_html=True)
