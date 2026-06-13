"""
Bank Customer Churn Prediction - Streamlit Web App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "models/churn_model.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

artifacts = load_model()

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 Bank Customer Churn Predictor</h1>
    <p>ML-powered churn risk scoring | European Central Bank · Retail Analytics</p>
</div>
""", unsafe_allow_html=True)

if artifacts is None:
    st.error("⚠️ Model not found! Please run `python train_model.py` first.")
    st.code("python train_model.py", language="bash")
    st.stop()

model        = artifacts['model']
scaler       = artifacts['scaler']
feature_cols = artifacts['feature_cols']
model_name   = artifacts['model_name']
all_metrics  = artifacts['metrics']

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
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
# FEATURE BUILDER
# ─────────────────────────────────────────────────────────────
def build_features(cs, geo, gen, age, ten, bal, sal, nprod, hcc, isact):
    row = {
        'CreditScore':        cs,
        'Age':                age,
        'Tenure':             ten,
        'Balance':            bal,
        'NumOfProducts':      nprod,
        'HasCrCard':          int(hcc),
        'IsActiveMember':     int(isact),
        'EstimatedSalary':    sal,
        'BalanceToSalaryRatio': bal / (sal + 1),
        'AgeTenureInteraction': age * ten,
        'ProductDensity':     nprod / (ten + 1),
        'EngagementScore':    int(isact) * nprod,
        'HasBalance':         int(bal > 0),
    }
    for g in ['France', 'Germany', 'Spain']:
        row[f'Geography_{g}'] = int(geo == g)
    for g in ['Female', 'Male']:
        row[f'Gender_{g}'] = int(gen == g)

    df_row = pd.DataFrame([row])
    for col in feature_cols:
        if col not in df_row.columns:
            df_row[col] = 0
    return df_row[feature_cols]

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
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Risk Factors")
            factors = []
            if age > 45:          factors.append(("🔴 Age > 45", "High churn tendency in older customers"))
            if geography == 'Germany': factors.append(("🔴 Germany region", "Highest churn rate geography"))
            if num_products == 1:  factors.append(("🔴 Single product", "Low engagement with bank"))
            if not is_active:      factors.append(("🟡 Inactive member", "No recent product activity"))
            if balance > 100000 and num_products == 1:
                                   factors.append(("🟡 High balance, low engagement", "At risk of moving funds"))
            if credit_score < 500: factors.append(("🟡 Low credit score", "Financial stress indicator"))
            if tenure < 2:         factors.append(("🟢 Short tenure", "Still building relationship"))

            if factors:
                for icon_label, desc in factors:
                    st.markdown(f"""<div class="insight-box">
                        <strong>{icon_label}</strong><br><small>{desc}</small>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("No major risk factors detected!")

        with col2:
            st.subheader("💡 Recommended Actions")
            if prob_pct >= 60:
                st.error("**Immediate Retention Required**")
                st.write("• Assign a relationship manager")
                st.write("• Offer personalized loyalty package")
                if num_products == 1:
                    st.write("• Cross-sell investment / savings product")
                if not is_active:
                    st.write("• Re-engagement email/SMS campaign")
                st.write("• Offer fee waiver or cashback reward")
            elif prob_pct >= 35:
                st.warning("**Proactive Engagement Needed**")
                st.write("• Include in outreach campaign")
                st.write("• Offer loyalty reward points")
                if num_products == 1:
                    st.write("• Introduce complementary product")
                st.write("• Schedule quarterly check-in call")
            else:
                st.success("**Retention Stable — Focus on Growth**")
                st.write("• Eligible for premium upsell offer")
                st.write("• Continue standard engagement")
                st.write("• Candidate for referral program")

        # Gauge chart
        st.markdown("---")
        st.subheader("📈 Risk Probability Gauge")
        fig, ax = plt.subplots(figsize=(8, 3))
        colors = ['#2e7d32', '#f9a825', '#c62828']
        thresholds = [0, 35, 60, 100]
        bar_colors = []
        for i in range(100):
            if i < 35:   bar_colors.append('#2e7d32')
            elif i < 60: bar_colors.append('#f9a825')
            else:        bar_colors.append('#c62828')
        ax.bar(range(100), [1]*100, color=bar_colors, width=1, alpha=0.4)
        ax.axvline(x=prob_pct, color='black', linewidth=3, linestyle='--', label=f'Score: {risk_score}%')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.5)
        ax.set_xlabel("Churn Risk Score (%)", fontsize=12)
        ax.set_yticks([])
        ax.text(prob_pct, 1.1, f'{risk_score}%', ha='center', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_title("Customer Churn Risk Gauge", fontsize=13, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    else:
        st.info("👈 Fill in customer details in the sidebar and click **Predict Churn Risk**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Model", model_name)
        c2.metric("Accuracy", f"{all_metrics[model_name]['accuracy']}%")
        c3.metric("ROC-AUC", f"{all_metrics[model_name]['roc_auc']}%")

# ════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Model Comparison")

    perf_df = pd.DataFrame([
        {
            'Model': k,
            'Accuracy (%)': v['accuracy'],
            'Precision (%)': v['precision'],
            'Recall (%)':    v['recall'],
            'F1-Score (%)':  v['f1'],
            'ROC-AUC (%)':   v['roc_auc'],
            'Best': '★' if k == model_name else ''
        }
        for k, v in all_metrics.items()
    ]).sort_values('ROC-AUC (%)', ascending=False)

    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    # Bar charts
    col1, col2 = st.columns(2)
    metrics_to_plot = ['Accuracy (%)', 'ROC-AUC (%)', 'F1-Score (%)', 'Recall (%)']
    colors_bar = ['#1565c0', '#283593', '#c62828', '#2e7d32']

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(perf_df))
        bars = ax.bar(x, perf_df['ROC-AUC (%)'], color=['#1565c0' if m == model_name else '#90a4ae'
                                                          for m in perf_df['Model']], width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(perf_df['Model'], rotation=15, ha='right', fontsize=9)
        ax.set_ylabel("ROC-AUC (%)")
        ax.set_title("ROC-AUC by Model", fontweight='bold')
        ax.set_ylim(50, 100)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(perf_df))
        width = 0.2
        metrics = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']
        clrs = ['#1565c0', '#283593', '#f57f17', '#2e7d32']
        for i, (metric, clr) in enumerate(zip(metrics, clrs)):
            ax.bar(x + i*width, perf_df[metric], width, label=metric.replace(' (%)', ''), color=clr, alpha=0.85)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(perf_df['Model'], rotation=15, ha='right', fontsize=9)
        ax.set_ylabel("%")
        ax.set_title("All Metrics Comparison", fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim(50, 100)
        st.pyplot(fig)
        plt.close()

    # Key metrics cards
    st.subheader(f"📌 Best Model: {model_name}")
    m = all_metrics[model_name]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{m['accuracy']}%")
    c2.metric("Precision", f"{m['precision']}%")
    c3.metric("Recall",    f"{m['recall']}%")
    c4.metric("F1-Score",  f"{m['f1']}%")
    c5.metric("ROC-AUC",   f"{m['roc_auc']}%")

# ════════════════════════════════════════════════════════════
# TAB 3 — FEATURE ANALYSIS
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔍 Feature Importance Analysis")

    if hasattr(model, 'feature_importances_'):
        fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
        top_n = fi.tail(15)

        fig, ax = plt.subplots(figsize=(9, 6))
        colors_fi = ['#c62828' if v > top_n.quantile(0.75) else
                     '#f57f17' if v > top_n.quantile(0.50) else '#1565c0'
                     for v in top_n.values]
        bars = ax.barh(range(len(top_n)), top_n.values, color=colors_fi, edgecolor='white')
        ax.set_yticks(range(len(top_n)))
        ax.set_yticklabels(top_n.index, fontsize=10)
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Top Feature Importances — {model_name}", fontweight='bold', fontsize=13)
        for bar, val in zip(bars, top_n.values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2.,
                    f'{val:.4f}', va='center', fontsize=9)
        patches = [mpatches.Patch(color='#c62828', label='High impact'),
                   mpatches.Patch(color='#f57f17', label='Medium impact'),
                   mpatches.Patch(color='#1565c0', label='Low impact')]
        ax.legend(handles=patches, loc='lower right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Table
        fi_df = pd.DataFrame({'Feature': fi.index[::-1], 'Importance': fi.values[::-1]})
        fi_df['Importance (%)'] = (fi_df['Importance'] * 100).round(3)
        fi_df['Rank'] = range(1, len(fi_df)+1)
        st.dataframe(fi_df[['Rank','Feature','Importance (%)']].head(15),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Feature importance not available for this model type.")

    st.markdown("---")
    st.subheader("📐 Engineered Features Explained")
    eng_df = pd.DataFrame({
        'Feature': ['BalanceToSalaryRatio', 'AgeTenureInteraction',
                    'ProductDensity', 'EngagementScore', 'HasBalance'],
        'Formula': ['Balance / (EstimatedSalary + 1)', 'Age × Tenure',
                    'NumOfProducts / (Tenure + 1)', 'IsActiveMember × NumOfProducts',
                    '1 if Balance > 0 else 0'],
        'Purpose': ['Wealth relative to income', 'Loyalty adjusted for age',
                    'Products per year of tenure', 'Combined engagement signal',
                    'Whether customer uses savings account']
    })
    st.dataframe(eng_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# TAB 4 — SCENARIO SIMULATOR
# ════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🎮 What-If Scenario Simulator")
    st.info("Adjust sliders to see how changing customer attributes affects churn risk in real-time.")

    col1, col2 = st.columns(2)
    with col1:
        sim_age     = st.slider("Age",              18, 80, 40, key='sim_age')
        sim_tenure  = st.slider("Tenure (years)",    0, 10,  3, key='sim_ten')
        sim_cs      = st.slider("Credit Score",    300, 900, 620, 10, key='sim_cs')
        sim_nprod   = st.selectbox("Products",      [1, 2, 3, 4], key='sim_prod')
    with col2:
        sim_bal     = st.number_input("Balance (€)",  0.0, 300000.0, 50000.0, 5000.0, key='sim_bal')
        sim_sal     = st.number_input("Salary (€)",  10000.0, 200000.0, 70000.0, 5000.0, key='sim_sal')
        sim_geo     = st.selectbox("Geography",     ["France", "Spain", "Germany"], key='sim_geo')
        sim_active  = st.checkbox("Active Member",  value=True, key='sim_act')

    X_sim = build_features(sim_cs, sim_geo, 'Male', sim_age, sim_tenure,
                            sim_bal, sim_sal, sim_nprod, True, sim_active)
    if model_name == 'Logistic Regression':
        sim_prob = model.predict_proba(scaler.transform(X_sim))[0][1]
    else:
        sim_prob = model.predict_proba(X_sim)[0][1]
    sim_pct = round(sim_prob * 100)

    col_r, col_s = st.columns(2)
    with col_r:
        color = "#c62828" if sim_pct >= 60 else "#f57f17" if sim_pct >= 35 else "#2e7d32"
        label = "HIGH RISK" if sim_pct >= 60 else "MEDIUM RISK" if sim_pct >= 35 else "LOW RISK"
        st.markdown(f"""
        <div style="background:{color}22;border-left:5px solid {color};border-radius:10px;padding:1.5rem;text-align:center">
            <h2 style="color:{color}">{label}</h2>
            <h1 style="font-size:3.5rem;color:{color}">{sim_pct}%</h1>
            <p style="color:{color}">Churn probability: {sim_prob:.4f}</p>
        </div>""", unsafe_allow_html=True)

    with col_s:
        st.subheader("📋 Scenario Insights")
        if sim_nprod == 1:
            st.warning("Adding a 2nd product could significantly reduce churn risk")
        if not sim_active:
            st.warning("Activating membership reduces churn probability by ~13%")
        if sim_age > 50:
            st.info("Customers above 50 show higher churn tendency")
        if sim_geo == 'Germany':
            st.warning("Germany has the highest churn rate (~32%)")
        if sim_cs < 550:
            st.error("Low credit score is a strong churn predictor")
        if sim_tenure > 6:
            st.success("Long tenure reduces churn risk significantly")
        if sim_pct < 30:
            st.success("This customer profile has excellent retention likelihood")

    # Sensitivity analysis
    st.markdown("---")
    st.subheader("🧪 Sensitivity Analysis — How Each Factor Affects Risk")
    sensitivity_data = []
    base = sim_prob

    tweaks = [
        ("Age +10",            build_features(sim_cs, sim_geo, 'Male', min(sim_age+10,80),
                                              sim_tenure, sim_bal, sim_sal, sim_nprod, True, sim_active)),
        ("Add 1 Product",      build_features(sim_cs, sim_geo, 'Male', sim_age,
                                              sim_tenure, sim_bal, sim_sal, min(sim_nprod+1,4), True, sim_active)),
        ("Make Inactive",      build_features(sim_cs, sim_geo, 'Male', sim_age,
                                              sim_tenure, sim_bal, sim_sal, sim_nprod, True, False)),
        ("Switch to Germany",  build_features(sim_cs, 'Germany', 'Male', sim_age,
                                              sim_tenure, sim_bal, sim_sal, sim_nprod, True, sim_active)),
        ("Credit -100",        build_features(max(sim_cs-100,300), sim_geo, 'Male', sim_age,
                                              sim_tenure, sim_bal, sim_sal, sim_nprod, True, sim_active)),
        ("Tenure +3 years",    build_features(sim_cs, sim_geo, 'Male', sim_age,
                                              min(sim_tenure+3,10), sim_bal, sim_sal, sim_nprod, True, sim_active)),
    ]

    for label, X_twk in tweaks:
        if model_name == 'Logistic Regression':
            p = model.predict_proba(scaler.transform(X_twk))[0][1]
        else:
            p = model.predict_proba(X_twk)[0][1]
        delta = p - base
        sensitivity_data.append({'Change': label,
                                  'New Risk (%)': round(p*100, 1),
                                  'Delta (%)':    round(delta*100, 1)})

    sens_df = pd.DataFrame(sensitivity_data)
    fig, ax = plt.subplots(figsize=(8, 4))
    clrs = ['#c62828' if d > 0 else '#2e7d32' for d in sens_df['Delta (%)']]
    bars = ax.barh(sens_df['Change'], sens_df['Delta (%)'], color=clrs, edgecolor='white', height=0.5)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel("Change in Churn Risk (%)")
    ax.set_title("Sensitivity: Impact of Each Change on Churn Risk", fontweight='bold')
    for bar, val in zip(bars, sens_df['Delta (%)']):
        x_pos = val + 0.3 if val >= 0 else val - 0.3
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2., f'{val:+.1f}%', va='center', ha=ha, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.dataframe(sens_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>🏦 Bank Customer Churn Predictor &nbsp;|&nbsp; "
    "Powered by Scikit-Learn &nbsp;|&nbsp; "
    "European Central Bank Retail Analytics</small></center>",
    unsafe_allow_html=True
)
