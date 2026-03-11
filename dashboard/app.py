"""
dashboard/app.py — Streamlit CRM Churn Dashboard.

Features:
  - Single customer churn predictor form
  - Risk gauge + recommended CRM action
  - Bulk CSV upload & scoring
  - Feature importance chart
  - Business metrics summary

Run with: streamlit run dashboard/app.py
"""

import sys
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from config import BEST_MODEL_PATH, CHURN_RISK_THRESHOLDS


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CRM Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load model ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if BEST_MODEL_PATH.exists():
        return joblib.load(BEST_MODEL_PATH)
    return None

pipeline = load_model()

# ─── Helper functions ─────────────────────────────────────────────────────────

def get_risk_level(prob: float) -> str:
    for level, (lo, hi) in CHURN_RISK_THRESHOLDS.items():
        if lo <= prob < hi:
            return level
    return "high"

RISK_COLORS = {"low": "#4CAF50", "medium": "#FF9800", "high": "#F44336"}

def risk_gauge(prob: float, title: str = "Churn Probability") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        delta={"reference": 50, "suffix": "%"},
        title={"text": title, "font": {"size": 20}},
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": RISK_COLORS[get_risk_level(prob)]},
            "steps": [
                {"range": [0,  33], "color": "#E8F5E9"},
                {"range": [33, 66], "color": "#FFF3E0"},
                {"range": [66, 100], "color": "#FFEBEE"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=30))
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/customer-insight.png", width=48)
    st.title("CRM Churn Intelligence")
    st.caption("Powered by XGBoost + MLflow")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🔍 Single Prediction", "📂 Bulk Scoring", "📈 Model Insights"],
        index=0
    )

    st.divider()
    if pipeline:
        st.success("✅ Model loaded")
    else:
        st.error("❌ Model not found\nRun `python run_pipeline.py` first")


# ─── Single Prediction Page ───────────────────────────────────────────────────

if page == "🔍 Single Prediction":
    st.title("🔍 Customer Churn Predictor")
    st.caption("Enter customer CRM data to predict churn risk in real time.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Account Info")
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges (€)", 18.0, 120.0, 65.5, step=0.5)
        total_charges   = st.number_input("Total Charges (€)", 0.0, 9000.0, float(tenure * monthly_charges))
        contract        = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment         = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col2:
        st.subheader("Demographics")
        gender      = st.selectbox("Gender", ["Male", "Female"])
        partner     = st.selectbox("Partner", ["Yes", "No"])
        dependents  = st.selectbox("Dependents", ["Yes", "No"])
        phone       = st.selectbox("Phone Service", ["Yes", "No"])
        multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    with col3:
        st.subheader("Internet Services")
        internet    = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security    = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup      = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device      = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech        = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv          = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies      = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.divider()
    predict_btn = st.button("🚀 Predict Churn Risk", use_container_width=True, type="primary")

    if predict_btn:
        if not pipeline:
            st.error("Model not loaded. Run `python run_pipeline.py` first.")
        else:
            input_data = pd.DataFrame([{
                "tenure": tenure, "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges, "gender": gender,
                "Partner": partner, "Dependents": dependents,
                "PhoneService": phone, "MultipleLines": multi_lines,
                "InternetService": internet, "OnlineSecurity": security,
                "OnlineBackup": backup, "DeviceProtection": device,
                "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
                "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment
            }])

            prob      = float(pipeline.predict_proba(input_data)[0, 1])
            risk      = get_risk_level(prob)
            color     = RISK_COLORS[risk]

            r1, r2 = st.columns([1, 1])
            with r1:
                st.plotly_chart(risk_gauge(prob), use_container_width=True)

            with r2:
                st.markdown(f"### Risk Level: :{risk} **{risk.upper()}**")
                st.markdown(f"**Churn Probability: {prob:.1%}**")
                st.divider()

                actions = {
                    "low":    "✅ No immediate action needed. Schedule quarterly check-in.",
                    "medium": "⚠️ Send retention offer. Assign to Customer Success.",
                    "high":   "🚨 URGENT: Escalate to retention team. Offer contract upgrade."
                }
                st.info(actions[risk])

                st.metric("Monthly Revenue at Risk",
                          f"€{monthly_charges:.2f}",
                          delta=f"{prob:.0%} probability of loss")


# ─── Bulk Scoring Page ────────────────────────────────────────────────────────

elif page == "📂 Bulk Scoring":
    st.title("📂 Bulk Customer Scoring")
    st.caption("Upload a CSV file with customer CRM data to score all records at once.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df_upload):,} customers")
        st.dataframe(df_upload.head())

        if pipeline and st.button("Score All Customers", type="primary"):
            with st.spinner("Scoring …"):
                feature_cols = [c for c in df_upload.columns if c != "Churn"]
                probs = pipeline.predict_proba(df_upload[feature_cols])[:, 1]
                df_upload["ChurnProbability"] = probs.round(4)
                df_upload["RiskLevel"] = df_upload["ChurnProbability"].apply(get_risk_level)

            st.success("Scoring complete!")

            # Summary KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Customers", f"{len(df_upload):,}")
            k2.metric("High Risk",   f"{(df_upload['RiskLevel']=='high').sum():,}",   delta=f"{(df_upload['RiskLevel']=='high').mean():.0%}")
            k3.metric("Medium Risk", f"{(df_upload['RiskLevel']=='medium').sum():,}", delta=f"{(df_upload['RiskLevel']=='medium').mean():.0%}")
            k4.metric("Low Risk",    f"{(df_upload['RiskLevel']=='low').sum():,}",    delta=f"{(df_upload['RiskLevel']=='low').mean():.0%}")

            # Distribution chart
            fig = px.histogram(
                df_upload, x="ChurnProbability", color="RiskLevel",
                color_discrete_map=RISK_COLORS, nbins=30,
                title="Churn Probability Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Downloadable results
            csv = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Scored Results",
                csv, "churn_scores.csv", "text/csv",
                use_container_width=True
            )


# ─── Model Insights Page ──────────────────────────────────────────────────────

elif page == "📈 Model Insights":
    st.title("📈 Model Performance & Insights")

    shap_path = Path("logs/plots/xgboost_shap_importance.png")
    roc_path  = Path("logs/plots/roc_curves_comparison.png")
    cm_path   = Path("logs/plots/xgboost_confusion_matrix.png")

    col1, col2 = st.columns(2)

    with col1:
        if roc_path.exists():
            st.subheader("ROC Curves")
            st.image(str(roc_path), use_column_width=True)
        else:
            st.info("Run `python run_pipeline.py` to generate plots.")

    with col2:
        if cm_path.exists():
            st.subheader("Confusion Matrix")
            st.image(str(cm_path), use_column_width=True)

    if shap_path.exists():
        st.subheader("SHAP Feature Importance")
        st.caption("Shows which features most influence the churn prediction.")
        st.image(str(shap_path), use_column_width=True)

    st.divider()
    st.subheader("📖 Business Interpretation")
    insights = {
        "Contract Type":     "Month-to-month customers churn 3× more than two-year contracts.",
        "Tenure":            "Customers in their first 6 months are the highest churn risk.",
        "Fiber Optic":       "Fiber optic customers churn more — likely due to price sensitivity.",
        "No Tech Support":   "Customers without tech support are 2× more likely to churn.",
        "Monthly Charges":   "Customers paying >€80/month show elevated churn probability.",
    }
    for feature, insight in insights.items():
        st.markdown(f"**{feature}**: {insight}")
