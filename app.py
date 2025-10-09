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
try:
    import altair as alt
    _HAS_ALTAIR = True
except Exception:
    alt = None
    _HAS_ALTAIR = False
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
LOGREG_BASELINE_PATH = os.path.join(MODELS_DIR, "logreg_baseline_pipeline.pkl")
RF_PIPELINE_PATH = os.path.join(MODELS_DIR, "rf_pipeline.pkl")
XGB_PIPELINE_PATH = os.path.join(MODELS_DIR, "xgb_pipeline.pkl")

DEFAULT_OPERATING_THRESHOLD = 0.60  # fallback if metrics are unavailable
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


@st.cache_resource
def load_pipeline_at(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline not found at {path}.")
    return joblib.load(path)


def get_operating_threshold() -> float:
    """
    Keep the app's decision threshold aligned with the persisted metrics.
    Falls back to DEFAULT_OPERATING_THRESHOLD if metrics are missing.
    """
    try:
        metrics = load_metrics()
        thr = metrics.get("test", {}).get("used_threshold")
        if thr is None:
            thr = metrics.get("val", {}).get("chosen_threshold")
        if thr is not None:
            return float(thr)
    except Exception:
        pass
    return DEFAULT_OPERATING_THRESHOLD


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

@st.cache_data
def get_feature_thresholds() -> tuple[float, float]:
    train_df, _, _ = load_processed_splits()
    ratio = (train_df["MonthlyCharges"] / train_df["CLV"].replace(0, 1)).median()
    median_monthly = train_df["MonthlyCharges"].median()
    return float(ratio), float(median_monthly)

# --------------------------- Utilities ---------------------------
def compute_clv(monthly_charges: float, contract: str) -> float:
    return float(monthly_charges) * EXPECTED_TENURE_MAP.get(contract, 6)

def risk_bucket(prob: float):
    return "High" if prob >= 0.70 else "Medium" if prob >= 0.40 else "Low"

def predict_single(pipe, payload: dict) -> dict:
    X = pd.DataFrame([payload])
    threshold = get_operating_threshold()
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        prob = float(pipe.predict_proba(X)[:, 1][0])
    else:
        prob = float(pipe.predict(X)[0])
    return {"prob": prob, "pred": int(prob >= threshold), "threshold": threshold}

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
    try:
        train_df, _, _ = load_processed_splits()
        X_tr = train_df.drop(columns=["ChurnFlag"])
        X_tr_pre = pre.transform(X_tr)
        X_tr_pre = X_tr_pre.toarray() if hasattr(X_tr_pre, "toarray") else np.asarray(X_tr_pre)
        std = X_tr_pre.std(axis=0)
    except Exception:
        std = None
    names = _get_feature_names(pre)
    coefs = model.coef_.ravel()
    n = min(len(names), len(coefs))
    if std is not None and len(std) >= n:
        importance_vals = np.abs(coefs[:n] * std[:n])
    else:
        importance_vals = np.abs(coefs[:n])
    df = pd.DataFrame({
        "FeatureRaw": names[:n],
        "Importance": importance_vals,
        "Signed": coefs[:n]
    })
    df["Feature"] = df["FeatureRaw"].apply(pretty_feature_name)
    df["Direction"] = np.where(df["Signed"] >= 0, "Increases churn", "Decreases churn")
    return df.sort_values("Importance", ascending=False).head(top_n)[["Feature","Importance","Direction","Signed"]]

def global_importance_tree_feature_importances(pipe, top_n=25) -> pd.DataFrame:
    pre   = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Feature","Importance"])
    names = _get_feature_names(pre)
    imps  = model.feature_importances_
    n = min(len(names), len(imps))
    df = pd.DataFrame({
        "FeatureRaw": names[:n],
        "Importance": imps[:n],
    })
    df["Feature"] = df["FeatureRaw"].apply(pretty_feature_name)
    df["Direction"] = "Contributes to churn"
    return df.sort_values("Importance", ascending=False).head(top_n)[["Feature","Importance","Direction"]]

def tree_shap_importance(pipe, sample_df: pd.DataFrame, top_n=25) -> pd.DataFrame:
    if not _HAS_SHAP:
        return pd.DataFrame()
    pre   = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    X_pre = pre.transform(sample_df)
    feat_names = _get_feature_names(pre)
    X_d = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_d)
    except Exception:
        return pd.DataFrame()
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_vals_use = shap_vals[1]
    else:
        shap_vals_use = shap_vals
    importance = np.mean(np.abs(shap_vals_use), axis=0)
    if hasattr(importance, "A1"):  # sparse matrix -> flatten
        importance = importance.A1
    else:
        importance = np.asarray(importance).ravel()
    n = min(len(feat_names), len(importance))
    feature_list = list(feat_names[:n])
    df = pd.DataFrame({
        "FeatureRaw": feature_list,
        "Importance": importance[:n],
        "Direction": "Contributes to churn"
    })
    df["Feature"] = df["FeatureRaw"].apply(pretty_feature_name)
    return df.sort_values("Importance", ascending=False).head(top_n)[["Feature","Importance","Direction"]]

def render_importance_chart(df: pd.DataFrame, title: str):
    if df.empty:
        st.info("No importance data available.")
        return
    display_cols = [c for c in df.columns if c != "FeatureRaw"]
    if not _HAS_ALTAIR:
        st.write(title)
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
        return
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Importance:Q", title="Importance"),
        y=alt.Y("Feature:N", sort="-x", title="Feature"),
        color=alt.Color("Direction:N", legend=alt.Legend(title="Effect"), scale=alt.Scale(scheme="tealblues")) if "Direction" in df.columns else alt.value("#7C4DFF"),
        tooltip=display_cols
    ).properties(title=title, width="container", height=320)
    st.altair_chart(chart, use_container_width=True)

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
                      "techsup":False,"stv":False,"smov":False,"partner":False,"dependents":False
                  }))
    with colp2:
        st.button("🛡️ Low-risk preset", key="lo",
                  on_click=lambda: st.session_state.update({
                      "tenure":24,"monthly":60,"contract":"Two year","internet_service":"DSL",
                      "payment_method":"Bank transfer (automatic)","senior":False,"paperless":False,"gender":"Male",
                      "phone":True,"multiline":"Yes","onsec":True,"onbak":True,"devprot":True,
                      "techsup":True,"stv":True,"smov":True,"partner":True,"dependents":False
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
            partner   = st.toggle("Partner", value=st.session_state.get("partner", False))
            dependents = st.toggle("Dependents", value=st.session_state.get("dependents", False))

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
        tenure_ord_map = {"0-6m": 0, "6-12m": 1, "12-24m": 2, "24m+": 3}
        tenure_bucket_ord = tenure_ord_map.get(tenure_bucket, 0)
        autopay_methods = {"Bank transfer (automatic)", "Credit card (automatic)"}
        is_auto_pay = int(payment_method in autopay_methods)
        is_long_contract = int("year" in contract.lower())
        streaming_services = int(stv) + int(smov)
        support_services = int(onsec) + int(onbak) + int(devprot) + int(techsup)
        senior_fiber_optic = int(senior and internet_service.lower() == "fiber optic")
        tenure_years = tenure / 12.0
        per_service_charge = monthly / max(services_count, 1)
        is_fiber_optic = int(internet_service.lower() == "fiber optic")
        has_streaming_any = int(streaming_services > 0)
        has_support_any = int(support_services > 0)
        has_multiple_lines_yes = int(multiline.strip().lower() == "yes")
        is_electronic_check = int(payment_method.strip().lower() == "electronic check")
        paperless_autopay = int(paperless and is_auto_pay == 1)
        family_bundle = int(partner or dependents)
        early_tenure_flag = int(tenure <= 6)
        remaining_value = max(clv_est - total_charges, 0.0)
        expected_tenure = EXPECTED_TENURE_MAP.get(contract, 6)
        tenure_remaining = max(expected_tenure - tenure, 0)
        tenure_fraction_complete = min(tenure / max(expected_tenure, 1), 5.0)
        remaining_value_ratio = remaining_value / max(clv_est, 1.0)
        charges_per_month_tenure = total_charges / max(tenure, 1)

        # Load training medians for thresholds
        ratio_median, monthly_median = get_feature_thresholds()
        high_charge_to_clv_ratio = int((monthly / max(clv_est, 1.0)) >= ratio_median)
        high_monthly_charge = int(monthly >= monthly_median)

        payload = {
            "tenure": int(tenure),
            "MonthlyCharges": float(monthly),
            "TotalCharges": float(total_charges),
            "CLV": float(clv_est),
            "services_count": int(services_count),
            "monthly_to_total_ratio": float(monthly_to_total_ratio),
            "internet_no_tech_support": int(internet_no_tech_support),
            "tenure_bucket_ord": int(tenure_bucket_ord),
            "is_auto_pay": int(is_auto_pay),
            "is_long_contract": int(is_long_contract),
            "streaming_services": int(streaming_services),
            "support_services": int(support_services),
            "senior_fiber_optic": int(senior_fiber_optic),
            "charges_per_month_of_tenure": float(charges_per_month_tenure),
            "high_charge_to_clv_ratio": int(high_charge_to_clv_ratio),
            "tenure_years": float(tenure_years),
            "per_service_charge": float(per_service_charge),
            "is_fiber_optic": int(is_fiber_optic),
            "has_streaming_any": int(has_streaming_any),
            "has_support_any": int(has_support_any),
            "has_multiple_lines_yes": int(has_multiple_lines_yes),
            "is_electronic_check": int(is_electronic_check),
            "paperless_autopay": int(paperless_autopay),
            "family_bundle": int(family_bundle),
            "early_tenure_flag": int(early_tenure_flag),
            "high_monthly_charge": int(high_monthly_charge),
            "tenure_remaining": int(tenure_remaining),
            "tenure_fraction_complete": float(tenure_fraction_complete),
            "remaining_value": float(remaining_value),
            "remaining_value_ratio": float(remaining_value_ratio),
            "gender": gender,
            "SeniorCitizen": int(senior),
            "Partner": "Yes" if partner else "No",
            "Dependents": "Yes" if dependents else "No",
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
            explain_pipe = None
            try:
                explain_pipe = load_pipeline_at(LOGREG_BASELINE_PATH)
            except Exception:
                explain_pipe = pipe if hasattr(pipe.named_steps["model"], "coef_") else None
            prob, pred = res["prob"], res["pred"]
            risk = risk_bucket(prob)
            badge = "badge high" if risk=="High" else "badge med" if risk=="Medium" else "badge"

            st.markdown('<div class="card section">', unsafe_allow_html=True)
            m1, m2 = st.columns([1,1])
            with m1:
                st.metric("Churn probability", f"{prob*100:.1f}%")
                st.progress(min(max(prob,0.0),1.0))
            with m2:
                st.markdown(f"**Decision (@thr={res['threshold']:.2f}):** {'Churn (1)' if pred else 'No Churn (0)'}")
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
            expl = explain_logreg_prediction(explain_pipe, payload, top_k=5) if explain_pipe is not None else None
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
            else:
                st.info("Logistic-regression interpretability unavailable. Regenerate models to refresh coefficients.")

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

    mm_path = os.path.join(MODELS_DIR, "metrics_all.json")
    if os.path.exists(mm_path):
        with open(mm_path) as f:
            all_metrics = json.load(f, object_pairs_hook=dict)
        if all_metrics:
            st.markdown('<div class="card section">', unsafe_allow_html=True)
            st.write("**Confusion matrices (Test)**")
            cols = st.columns(len(all_metrics))
            label_map = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
            for col, (model_key, stats) in zip(cols, all_metrics.items()):
                with col:
                    test_stats = stats.get("test", {})
                    cm = np.array([
                        [int(test_stats.get("tn", 0)), int(test_stats.get("fp", 0))],
                        [int(test_stats.get("fn", 0)), int(test_stats.get("tp", 0))],
                    ], dtype=int)
                    fig, ax = plt.subplots(figsize=(3.6, 3.6))
                    im = ax.imshow(cm, cmap="Blues")
                    for (i, j), value in np.ndenumerate(cm):
                        text_color = "white" if value > cm.max() / 2 else "#0b1220"
                        ax.text(j, i, f"{value}", va="center", ha="center", color=text_color, fontsize=11, fontweight="bold")
                    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                    ax.set_xticklabels(["No churn", "Churn"])
                    ax.set_yticklabels(["No churn", "Churn"])
                    ax.set_xlabel("Predicted label")
                    ax.set_ylabel("True label")
                    ax.set_title(label_map.get(model_key, model_key.upper()), fontsize=12, pad=8)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            st.caption("Each matrix uses the model's saved validation threshold evaluated on the held-out test set.")
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

    pipeline_options = [
        ("Logistic Regression", LOGREG_BASELINE_PATH, "linear"),
        ("Random Forest", RF_PIPELINE_PATH, "tree"),
        ("XGBoost", XGB_PIPELINE_PATH, "tree"),
    ]
    available_pipes = [(label, path, kind) for label, path, kind in pipeline_options if os.path.exists(path)]
    if not available_pipes:
        st.info("No saved model pipelines found. Run training to generate them.")
    else:
        metrics_model = None
        try:
            metrics_model = load_metrics().get("model_name")
        except Exception:
            metrics_model = None
        default_label = None
        if metrics_model:
            label_map = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
            default_label = label_map.get(metrics_model)
        labels = [lbl for lbl, _, _ in available_pipes]
        default_index = labels.index(default_label) if default_label in labels else 0
        selected_label = st.selectbox("Select model for global importance", labels, index=default_index)
        selected_path, selected_kind = next((p, k) for lbl, p, k in available_pipes if lbl == selected_label)

        try:
            pipe = load_pipeline_at(selected_path)
        except Exception as e:
            st.error(f"Could not load pipeline: {e}")
            pipe = None

        if pipe is not None:
            sample_df = None
            try:
                _, _, test_df = load_processed_splits()
                X_test = test_df.drop(columns=["ChurnFlag"])
                sample_n = min(1000, len(X_test))
                sample_df = X_test.sample(sample_n, random_state=42) if len(X_test) > sample_n else X_test
            except Exception as e:
                st.caption("Note: could not load test set sample for SHAP — " + str(e))

            if selected_kind == "tree":
                df_imp = pd.DataFrame()
                if _HAS_SHAP and sample_df is not None:
                    df_imp = tree_shap_importance(pipe, sample_df, top_n=25)
                    if not df_imp.empty:
                        render_importance_chart(df_imp, f"{selected_label} SHAP feature importance")
                        st.caption("Mean |SHAP| values on a sampled test subset.")
                if df_imp.empty:
                    df_imp = global_importance_tree_feature_importances(pipe, top_n=25)
                    if df_imp.empty:
                        st.info("Feature importances unavailable for this model.")
                    else:
                        render_importance_chart(df_imp, f"{selected_label} feature importance")
                        st.caption("Tree-based importance derived from feature_importances_.")
            else:
                df_imp = global_importance_logreg(pipe, top_n=25)
                if df_imp.empty:
                    st.info("Coefficient-based importance unavailable for this model.")
                else:
                    render_importance_chart(df_imp, "Logistic Regression feature importance")
                    st.caption("Bars use |coefficient × std|; color indicates churn direction.")
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
            st.markdown('<div class="card section">', unsafe_allow_html=True)
            st.write("**CLV Distribution (train)**")
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            ax.hist(train["CLV"], bins=30, color="#7C4DFF", alpha=0.85, edgecolor="#1f2233")
            ax.set_xlabel("Customer Lifetime Value ($)")
            ax.set_ylabel("Customers")
            ax.set_title("CLV Distribution (Train Split)")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption("Histogram highlights how customer value skews; CLV = Monthly Charges × Expected Tenure.")
            st.markdown('</div>', unsafe_allow_html=True)

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
