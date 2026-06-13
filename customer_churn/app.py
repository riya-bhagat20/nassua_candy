# ============================================
# app.py - Streamlit Churn Prediction App
# ============================================
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import subprocess
subprocess.run(["pip", "install", "xgboost"], capture_output=True)


# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1565c0 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { font-size: 2rem; margin-bottom: 0.3rem; }
    .main-header p  { font-size: 1rem; opacity: 0.85; }

    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .risk-high  { background:#ffebee; border-left:5px solid #c62828; border-radius:10px; padding:1.5rem; }
    .risk-med   { background:#fff8e1; border-left:5px solid #f57f17; border-radius:10px; padding:1.5rem; }
    .risk-low   { background:#e8f5e9; border-left:5px solid #2e7d32; border-radius:10px; padding:1.5rem; }
    .risk-high h2 { color:#c62828; }
    .risk-med  h2 { color:#f57f17; }
    .risk-low  h2 { color:#2e7d32; }

    .insight-box {
        background:#e3f2fd;
        border-left:4px solid #1565c0;
        border-radius:6px;
        padding:0.8rem 1rem;
        margin:0.4rem 0;
        font-size:0.9rem;
    }
    .stButton>button {
        background: linear-gradient(135deg,#1565c0,#283593);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover { opacity:0.9; }
</style>
""", unsafe_allow_html=True)


# Load model & scaler
model = joblib.load('best_churn_model.pkl')
scaler = joblib.load('scaler.pkl')

THRESHOLD = 0.50  # apna best_threshold yahan daalo


st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 Bank Customer Churn Predictor</h1>
    <p>ML-powered churn risk scoring | European Central Bank · Retail Analytics</p>
</div>
""", unsafe_allow_html=True)

# ============================================
with st.sidebar:
    st.markdown("## 🧾 Customer Profile")
    st.markdown("---")

    credit_score = st.slider("Credit Score", 300, 900, 650, 10)
    geography    = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender       = st.selectbox("Gender", ["Male", "Female"])
    age          = st.slider("Age", 18, 80, 38)
    tenure       = st.slider("Tenure (years)", 0, 10, 5)
    balance      = st.number_input("Account Balance (€)", 0.0, 300000.0, 60000.0, 1000.0)
    salary       = st.number_input("Estimated Salary (€)", 10000.0, 200000.0, 80000.0, 1000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card  = st.checkbox("Has Credit Card", value=True)
    is_active    = st.checkbox("Is Active Member", value=True)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Churn Risk")

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Risk Calculator",
    "📊 Model Performance",
    "🔍 Feature Analysis",
    "🎮 Scenario Simulator"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — RISK CALCULATOR
# ════════════════════════════════════════════════════════════
with tab1:
    if predict_btn:
        X_input = build_features(credit_score, geography, gender, age, tenure,
                                  balance, salary, num_products, has_cr_card, is_active)
        if model_name == 'Logistic Regression':
            X_input_scaled = scaler.transform(X_input)
            prob = model.predict_proba(X_input_scaled)[0][1]
        else:
            prob = model.predict_proba(X_input)[0][1]

        prob_pct = prob * 100
        risk_score = round(prob_pct)

        # Risk card
        if prob_pct >= 60:
            st.markdown(f"""<div class="risk-high">
                <h2>🔴 HIGH CHURN RISK</h2>
                <h1 style="font-size:3rem">{risk_score}%</h1>
                <p>Churn probability: <strong>{prob:.4f}</strong></p>
                <p>⚠️ Immediate retention action required</p>
            </div>""", unsafe_allow_html=True)
        elif prob_pct >= 35:
            st.markdown(f"""<div class="risk-med">
                <h2>🟡 MEDIUM CHURN RISK</h2>
                <h1 style="font-size:3rem">{risk_score}%</h1>
                <p>Churn probability: <strong>{prob:.4f}</strong></p>
                <p>📋 Proactive engagement recommended</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="risk-low">
                <h2>🟢 LOW CHURN RISK</h2>
                <h1 style="font-size:3rem">{risk_score}%</h1>
                <p>Churn probability: <strong>{prob:.4f}</strong></p>
                <p>✅ Customer likely to be retained</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

# TAB 4 — SCENARIO SIMULATOR
# ════════════════════════════════════════════════════════════
with tab4:
# ════════════════════════════════════════════════════════════
# TAB 1 — RISK CALCULATOR
# ════════════════════════════════════════════════════════════
    if predict_btn:
        X_input = build_features(credit_score, geography, gender, age, tenure,
                                  balance, salary, num_products, has_cr_card, is_active)
        if model_name == 'Logistic Regression':
            X_input_scaled = scaler.transform(X_input)
            prob = model.predict_proba(X_input_scaled)[0][1]
        else:
            prob = model.predict_proba(X_input)[0][1]

        prob_pct = prob * 100
        risk_score = round(prob_pct)

        # Risk card
        if prob_pct >= 60:
            st.markdown(f"""<div class="risk-high">
                <h2>🔴 HIGH CHURN RISK</h2>
                <h1 style="font-size:3rem">{risk_score}%</h1>
                <p>Churn probability: <strong>{prob:.4f}</strong></p>
                <p>⚠️ Immediate retention action required</p>
            </div>""", unsafe_allow_html=True)
        elif prob_pct >= 35:
            st.markdown(f"""<div class="risk-med">
                <h2>🟡 MEDIUM CHURN RISK</h2>
                <h1 style="font-size:3rem">{risk_score}%</h1>
                <p>Churn probability: <strong>{prob:.4f}</strong></p>
                <p>📋 Proactive engagement recommended</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="risk-low">
                <h2>🟢 LOW CHURN RISK</h2>
                <h1 style="font-size:3rem">{risk_score}%</h1>
                <p>Churn probability: <strong>{prob:.4f}</strong></p>
                <p>✅ Customer likely to be retained</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

# # ============================================
# # Feature Engineering (same as training)
# # ============================================
# def prepare_input():
#     bal_sal_ratio   = balance / salary if salary > 0 else 0
#     prod_density    = num_products / (tenure + 1)
#     engagement      = 1 if (is_active == "Yes" and num_products > 1) else 0
#     age_tenure      = age * tenure

#     data = {
#         'Year'                : 2025,
#         'CreditScore'         : credit_score,
#         'Age'                 : age,
#         'Tenure'              : tenure,
#         'Balance'             : balance,
#         'NumOfProducts'       : num_products,
#         'HasCrCard'           : 1 if has_cr_card == "Yes" else 0,
#         'IsActiveMember'      : 1 if is_active == "Yes" else 0,
#         'EstimatedSalary'     : salary,
#         'BalanceSalaryRatio'  : bal_sal_ratio,
#         'ProductDensity'      : prod_density,
#         'EngagementProduct'   : engagement,
#         'AgeTenure'           : age_tenure,
#         'Geography_Germany'   : geography == "Germany",
#         'Geography_Spain'     : geography == "Spain",
#         'Gender_Male'         : gender == "Male"
#     }
#     return pd.DataFrame([data])

# # ============================================
# # Predict Button
# # ============================================
# if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):

#     input_df     = prepare_input()
#     input_scaled = scaler.transform(input_df)
#     prob         = model.predict_proba(input_scaled)[:, 1][0]
#     prediction   = int(prob >= THRESHOLD)

#     st.divider()

#     # Result
#     if prediction == 1:
#         st.error(f"### ⚠️ Customer CHURN KAREGA")
#     else:
#         st.success(f"### ✅ Customer STAY KAREGA")

#     # Probability meter
#     st.metric("Churn Probability", f"{prob*100:.1f}%")
#     st.progress(float(prob))

#     # Risk level
#     if prob < 0.3:
#         st.info("🟢 Risk Level: LOW")
#     elif prob < 0.6:
#         st.warning("🟡 Risk Level: MEDIUM")
#     else:
#         st.error("🔴 Risk Level: HIGH")

#     # Details expander
#     with st.expander("📊 Input Details dekho"):
#         st.dataframe(input_df.T.rename(columns={0: 'Value'}))