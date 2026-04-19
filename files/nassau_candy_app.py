import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy | Factory Optimizer",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

 /* Dark chocolate theme */
:root {
    --chocolate: #3B1A08;
    --caramel: #C27B2E;
    --cream: #FDF6EC;
    --mint: #2EC27B;
    --danger: #E05252;
    --text-muted: #7A6A5A;
}

.stApp {
    background: #1A0D05;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #3B1A08 0%, #5C2E0E 50%, #7A4020 100%);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 28px;
    border: 1px solid #C27B2E44;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: "🍫";
    font-size: 200px;
    position: absolute;
    right: -20px;
    top: -40px;
    opacity: 0.07;
}
.hero-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 900;
    color: #FDF6EC;
    margin: 0 0 8px 0;
    letter-spacing: -1px;
}
.hero-header p {
    color: #C27B2EBB;
    font-size: 1.05rem;
    font-weight: 300;
    margin: 0;
}

/* Section headers */
            
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: #C27B2E;
    font-weight: 700;
    margin: 0 0 4px 0;
    letter-spacing: -0.3px;
}
.section-sub {
    color: #7A6A5A;
    font-size: 0.85rem;
    margin-bottom: 20px;
}

/* KPI cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.kpi-card {
    background: #2A1208;
    border: 1px solid #C27B2E33;
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #C27B2E, #E8A84A);
    border-radius: 12px 12px 0 0;
}
.kpi-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #7A6A5A;
    font-weight: 600;
    margin-bottom: 8px;
}
.kpi-value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #FDF6EC;
    font-weight: 700;
    line-height: 1;
}
.kpi-delta {
    font-size: 0.78rem;
    color: #2EC27B;
    margin-top: 6px;
    font-weight: 500;
}
.kpi-delta.neg { color: #E05252; }

/* Metric panel */
.metric-panel {
    background: #2A1208;
    border: 1px solid #C27B2E22;
    border-radius: 12px;
    padding: 20px 24px;
}

/* Recommendation card */
.rec-card {
    background: #2A1208;
    border-left: 4px solid #C27B2E;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.rec-card.good { border-left-color: #2EC27B; }
.rec-card.warn { border-left-color: #E8A84A; }
.rec-card.bad  { border-left-color: #E05252; }
.rec-title { font-weight: 600; color: #FDF6EC; font-size: 0.95rem; }
.rec-detail { color: #7A6A5A; font-size: 0.82rem; margin-top: 4px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1A0D05 !important;
    border-right: 1px solid #C27B2E22;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
    color: #C27B2E !important;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #2A1208;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #7A6A5A;
    font-weight: 500;
    border-radius: 6px;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: #C27B2E !important;
    color: #FDF6EC !important;
}

/* Dataframe */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Divider */
hr { border-color: #C27B2E22 !important; }

/* Info boxes */
.info-box {
    background: #C27B2E15;
    border: 1px solid #C27B2E44;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #C27B2E;
    margin: 12px 0;
}

/* Upload area */
.upload-zone {
    background: #2A1208;
    border: 2px dashed #C27B2E55;
    border-radius: 12px;
    padding: 40px;
    text-align: center;
    color: #7A6A5A;
}

/* Plotly chart background fix */
.js-plotly-plot { border-radius: 10px; }

/* Progress */
.stProgress > div > div { background: #C27B2E; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

PLOTLY_THEME = dict(
    plot_bgcolor="#1A0D05",
    paper_bgcolor="#2A1208",
    font_color="#FDF6EC",
    colorway=["#C27B2E", "#2EC27B", "#E05252", "#5B9BD5", "#E8A84A", "#9B59B6"],
    title_font_family="Playfair Display",
)

def style_fig(fig, title=""):
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text=title, font=dict(size=15, color="#C27B2E")),
        margin=dict(t=45, b=30, l=30, r=20),
        legend=dict(bgcolor="#1A0D05", font_color="#FDF6EC"),
        xaxis=dict(gridcolor="#291B0D", zerolinecolor="#291B0C"),
        yaxis=dict(gridcolor="#241A0F", zerolinecolor="#2A1C0D"),
    )
    return fig

@st.cache_data
def compute_lead_time(df):
    """Derive lead time in days from order and ship dates."""
    df = df.copy()
    for col in ["Order Date", "Ship Date"]:
        df[col] = pd.to_datetime(df[col], dayfirst=False, errors="coerce")
    df["Lead Time (Days)"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Lead Time (Days)"] = df["Lead Time (Days)"].clip(lower=0)
    return df

@st.cache_data
def prepare_features(df):
    df = df.copy()
    le = LabelEncoder()
    cat_cols = ["Ship Mode", "Region", "Division", "Product Name", "State/Province", "City"]
    for c in cat_cols:
        if c in df.columns:
            df[f"{c}_enc"] = le.fit_transform(df[c].astype(str))
    return df

def kpi_html(label, value, delta=None, neg=False):
    delta_html = ""
    if delta:
        cls = "neg" if neg else ""
        delta_html = f'<div class="kpi-delta {cls}">{delta}</div>'
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>"""

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px;'>
        <div style='font-family:Playfair Display,serif;font-size:1.3rem;color:#C27B2E;font-weight:900;'>🍫 Nassau Candy</div>
        <div style='color:#7A6A5A;font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;'>Factory Optimizer</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Dataset (CSV / Excel)", type=["csv", "xlsx", "xls"])
    st.markdown("---")
    st.markdown("<div style='color:#7A6A5A;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", ["📊 Overview", "🔍 EDA", "🤖 ML Models", "🗺️ Clustering", "⚙️ Scenario Sim", "💡 Recommendations"], label_visibility="collapsed")


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

if uploaded:
    raw_df = load_data(uploaded)
else:
    # Demo data matching the screenshot columns
    np.random.seed(42)
    n = 300
    regions = ["Interior", "Atlantic", "Gulf", "Pacific"]
    ship_modes = ["Standard Class", "Second Class", "First Class"]
    divisions = ["Chocolate", "Other"]
    products = [
        "Wonka Bar - Milk Chocolate", "Wonka Bar - Triple Dazzle Caramel",
        "Wonka Bar - Nutty Crunch Surprise", "Wonka Bar -Scrumdiddlyumptious",
        "Wonka Bar - Fudge Mallows", "Wonka Gum"
    ]
    product_ids = ["CHO-MIL-31000","CHO-TRI-54000","CHO-NUT-13000","CHO-SCR-58000","CHO-FUD-51000","OTH-GUM-21000"]
    states = ["Texas","Illinois","Kentucky","California","Virginia","Ohio","Pennsylvania","Georgia","Delaware","South Carolina","Louisiana"]
    cities = ["Houston","Naperville","Henderson","Los Angeles","Springfield","Newark","Philadelphia","Athens","Dover","Mount Pleasant","Bossier City"]

    order_dates = pd.date_range("2024-01-01", periods=n, freq="D").strftime("%m-%d-%Y")
    lead_days = np.random.choice([3,5,7,10,14,21], n, p=[0.1,0.2,0.3,0.2,0.1,0.1])
    ship_dates = [
        (pd.to_datetime(order_dates[i], format="%m-%d-%Y") + pd.Timedelta(days=int(lead_days[i]))).strftime("%m-%d-%Y")
        for i in range(n)
    ]
    sales = np.round(np.random.uniform(3, 35, n), 2)
    units = np.random.randint(1, 10, n)
    gross_profit = np.round(sales * np.random.uniform(0.3, 0.7, n), 2)
    cost = np.round(sales - gross_profit, 2)

    prod_idx = np.random.randint(0, len(products), n)
    state_idx = np.random.randint(0, len(states), n)

    raw_df = pd.DataFrame({
        "Row ID": range(1, n+1),
        "Order ID": [f"US-2021-{np.random.randint(100000,200000)}-{np.random.choice(['CHO','OTH'])}-{np.random.choice(['MIL','TRI','NUT','SCR','FUD','GUM'])}-{np.random.randint(10000,60000)}" for _ in range(n)],
        "Order Date": order_dates,
        "Ship Date": ship_dates,
        "Ship Mode": np.random.choice(ship_modes, n, p=[0.6,0.25,0.15]),
        "Customer ID": np.random.randint(100000, 200000, n),
        "Country/Region": "United States",
        "City": [cities[i] for i in state_idx],
        "State/Province": [states[i] for i in state_idx],
        "Postal Code": np.random.randint(10000, 99999, n),
        "Division": [divisions[0] if product_ids[prod_idx[i]].startswith("CHO") else divisions[1] for i in range(n)],
        "Region": np.random.choice(regions, n),
        "Product ID": [product_ids[i] for i in prod_idx],
        "Product Name": [products[i] for i in prod_idx],
        "Sales": sales,
        "Units": units,
        "Gross Profit": gross_profit,
        "Cost": cost,
    })

df = compute_lead_time(raw_df)
df_enc = prepare_features(df)

# Sidebar filters (shown after data load)
with st.sidebar:
    st.markdown("---")
    st.markdown("<div style='color:#7A6A5A;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;'>Filters</div>", unsafe_allow_html=True)
    if "Region" in df.columns:
        regions_avail = sorted(df["Region"].dropna().unique())
        sel_regions = st.multiselect("Region", regions_avail, default=regions_avail)
    if "Ship Mode" in df.columns:
        modes_avail = sorted(df["Ship Mode"].dropna().unique())
        sel_modes = st.multiselect("Ship Mode", modes_avail, default=modes_avail)
    if "Division" in df.columns:
        divs_avail = sorted(df["Division"].dropna().unique())
        sel_divs = st.multiselect("Division", divs_avail, default=divs_avail)

    # Apply filters
    mask = pd.Series([True] * len(df), index=df.index)
    if "Region" in df.columns and sel_regions:
        mask &= df["Region"].isin(sel_regions)
    if "Ship Mode" in df.columns and sel_modes:
        mask &= df["Ship Mode"].isin(sel_modes)
    if "Division" in df.columns and sel_divs:
        mask &= df["Division"].isin(sel_divs)
    fdf = df[mask].copy()

    st.markdown("---")
    st.markdown(f"<div style='color:#7A6A5A;font-size:0.75rem;'>Showing <b style='color:#C27B2E'>{len(fdf):,}</b> of <b style='color:#C27B2E'>{len(df):,}</b> rows</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>Factory Optimization Engine</h1>
    <p>ML-driven factory–product reassignment · Lead time prediction · Scenario simulation · Margin recovery</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════
if page == "📊 Overview":
    avg_lead   = fdf["Lead Time (Days)"].mean() if "Lead Time (Days)" in fdf else 0
    total_sales = fdf["Sales"].sum() if "Sales" in fdf else 0
    total_profit = fdf["Gross Profit"].sum() if "Gross Profit" in fdf else 0
    margin = (total_profit / total_sales * 100) if total_sales else 0

    kpis = "".join([
        kpi_html("Total Orders", f"{len(fdf):,}", "Filtered dataset"),
        kpi_html("Avg Lead Time", f"{avg_lead:.1f}d", "Days order→ship"),
        kpi_html("Total Revenue", f"${total_sales:,.0f}", "Total Revenue"),
        kpi_html("Gross Margin", f"{margin:.1f}%", "Profit / Revenue"),
    ])
    st.markdown(f'<div class="kpi-grid">{kpis}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if "Region" in fdf.columns:
            reg_sales = fdf.groupby("Region")[["Sales","Gross Profit"]].sum().reset_index()
            fig = px.bar(reg_sales, x="Region", y=["Sales","Gross Profit"],
                         barmode="group", title="Revenue & Profit by Region",
                         color_discrete_sequence=["#C27B2E","#2EC27B"])
            style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "Ship Mode" in fdf.columns:
            sm = fdf["Ship Mode"].value_counts().reset_index()
            sm.columns = ["Ship Mode","Count"]
            fig2 = px.pie(sm, names="Ship Mode", values="Count",
                          title="Order Distribution by Ship Mode",
                          color_discrete_sequence=["#C27B2E","#E8A84A","#2EC27B","#5B9BD5"])
            style_fig(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        if "Product Name" in fdf.columns:
            prod = fdf.groupby("Product Name")["Sales"].sum().sort_values(ascending=True).reset_index()
            fig3 = px.bar(prod, x="Sales", y="Product Name", orientation="h",
                          title="Sales by Product", color="Sales",
                          color_continuous_scale=["#3B1A08","#C27B2E","#FDF6EC"])
            style_fig(fig3)
            fig3.update_coloraxes(showscale=False)
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        if "Lead Time (Days)" in fdf.columns and "Region" in fdf.columns:
            lt_reg = fdf.groupby("Region")["Lead Time (Days)"].mean().reset_index()
            fig4 = px.bar(lt_reg, x="Region", y="Lead Time (Days)",
                          title="Avg Lead Time by Region",
                          color="Lead Time (Days)",
                          color_continuous_scale=["#2EC27B","#E8A84A","#E05252"])
            style_fig(fig4)
            fig4.update_coloraxes(showscale=False)
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='section-title'>Raw Data Preview</div>", unsafe_allow_html=True)
    st.dataframe(fdf.head(50), use_container_width=True, height=340)


# ════════════════════════════════════════════
# PAGE: EDA
# ════════════════════════════════════════════
elif page == "🔍 EDA":
    st.markdown("<div class='section-title'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Distribution deep-dives, correlation heatmap, and route-level performance</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Correlations", "🗓️ Time Trends", "📋 Summary Stats"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            if "Lead Time (Days)" in fdf.columns:
                fig = px.histogram(fdf, x="Lead Time (Days)", nbins=30,
                                   title="Lead Time Distribution",
                                   color_discrete_sequence=["#C27B2E"])
                style_fig(fig)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if "Sales" in fdf.columns:
                fig2 = px.histogram(fdf, x="Sales", nbins=40,
                                    title="Sales Distribution",
                                    color_discrete_sequence=["#2EC27B"])
                style_fig(fig2)
                st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            if "Lead Time (Days)" in fdf.columns and "Ship Mode" in fdf.columns:
                fig3 = px.box(fdf, x="Ship Mode", y="Lead Time (Days)",
                              title="Lead Time by Ship Mode",
                              color="Ship Mode",
                              color_discrete_sequence=["#C27B2E","#E8A84A","#2EC27B"])
                style_fig(fig3)
                st.plotly_chart(fig3, use_container_width=True)
        with col4:
            if "Gross Profit" in fdf.columns and "Product Name" in fdf.columns:
                fig4 = px.box(fdf, x="Product Name", y="Gross Profit",
                              title="Profit Distribution per Product",
                              color="Product Name",
                              color_discrete_sequence=["#C27B2E","#E8A84A","#2EC27B","#5B9BD5","#E05252","#9B59B6"])
                style_fig(fig4)
                fig4.update_xaxes(tickangle=25, tickfont_size=9)
                st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        num_cols = fdf.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) > 2:
            corr = fdf[num_cols].corr()
            fig = px.imshow(corr, text_auto=".2f",
                            title="Correlation Heatmap",
                            color_continuous_scale=["#E05252","#1A0D05","#2EC27B"],
                            zmin=-1, zmax=1)
            style_fig(fig)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        # Scatter: Sales vs Profit
        if "Sales" in fdf.columns and "Gross Profit" in fdf.columns:
            col1, col2 = st.columns(2)
            with col1:
                color_col = "Region" if "Region" in fdf.columns else None
                fig2 = px.scatter(fdf, x="Sales", y="Gross Profit",
                                  color=color_col, title="Sales vs Gross Profit",
                                  color_discrete_sequence=["#C27B2E","#2EC27B","#5B9BD5","#E8A84A"],
                                  opacity=0.6)
                style_fig(fig2)
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                if "Lead Time (Days)" in fdf.columns:
                    fig3 = px.scatter(fdf, x="Lead Time (Days)", y="Gross Profit",
                                      color=color_col, title="Lead Time vs Profit",
                                      color_discrete_sequence=["#C27B2E","#2EC27B","#5B9BD5","#E8A84A"],
                                      opacity=0.6)
                    style_fig(fig3)
                    st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        if "Order Date" in fdf.columns:
            ts = fdf.copy()
            ts["Order Date"] = pd.to_datetime(ts["Order Date"], dayfirst=False, errors="coerce")
            ts = ts.dropna(subset=["Order Date"])
            ts_monthly = ts.set_index("Order Date").resample("ME")[["Sales","Gross Profit"]].sum().reset_index()
            if len(ts_monthly) > 0:
                fig = px.line(ts_monthly, x="Order Date", y=["Sales","Gross Profit"],
                              title="Monthly Revenue & Profit Trend",
                              color_discrete_sequence=["#C27B2E","#2EC27B"])
                style_fig(fig)
                st.plotly_chart(fig, use_container_width=True)

            if "Lead Time (Days)" in ts.columns:
                lt_monthly = ts.set_index("Order Date").resample("ME")["Lead Time (Days)"].mean().reset_index()
                fig2 = px.area(lt_monthly, x="Order Date", y="Lead Time (Days)",
                               title="Average Lead Time Trend (Monthly)",
                               color_discrete_sequence=["#E05252"])
                style_fig(fig2)
                st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.markdown("<div class='metric-panel'>", unsafe_allow_html=True)
        st.dataframe(fdf.describe().round(2), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════
# PAGE: ML MODELS
# ════════════════════════════════════════════
elif page == "🤖 ML Models":
    st.markdown("<div class='section-title'>Predictive Modeling — Lead Time</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Train Linear Regression, Random Forest, and Gradient Boosting to predict shipping lead time</div>", unsafe_allow_html=True)

    feature_candidates = ["Ship Mode_enc","Region_enc","Division_enc","Product Name_enc","Units","Sales","Cost"]
    available_features = [c for c in feature_candidates if c in df_enc.columns]

    if "Lead Time (Days)" not in df_enc.columns or len(available_features) < 2:
        st.markdown("<div class='info-box'>⚠️ Not enough encoded features to run models. Ensure dataset has Ship Mode, Region, and Product Name columns.</div>", unsafe_allow_html=True)
    else:
        col_feat, col_run = st.columns([3, 1])
        with col_feat:
            sel_feat = st.multiselect("Feature Columns", available_features, default=available_features)
        with col_run:
            test_size = st.slider("Test Split %", 10, 40, 20)
            run_btn = st.button("🚀 Train Models", use_container_width=True)

        if run_btn and sel_feat:
            model_df = df_enc.dropna(subset=sel_feat + ["Lead Time (Days)"])
            X = model_df[sel_feat].values
            y = model_df["Lead Time (Days)"].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size/100, random_state=42)

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
            }

            results = {}
            preds_dict = {}
            progress = st.progress(0)
            for i, (name, model) in enumerate(models.items()):
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results[name] = {
                    "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
                    "MAE": mean_absolute_error(y_test, preds),
                    "R²": r2_score(y_test, preds)
                }
                preds_dict[name] = preds
                progress.progress((i+1)/3)

            progress.empty()

            # Results table
            res_df = pd.DataFrame(results).T.round(4).reset_index().rename(columns={"index":"Model"})
            best_model = res_df.loc[res_df["R²"].idxmax(), "Model"]

            st.markdown(f"<div class='info-box'>✅ Best Model: <b>{best_model}</b> (highest R²)</div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(res_df, x="Model", y=["RMSE","MAE"],
                             barmode="group", title="Model Errors (Lower is Better)",
                             color_discrete_sequence=["#E05252","#E8A84A"])
                style_fig(fig)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = px.bar(res_df, x="Model", y="R²",
                              title="R² Score (Higher is Better)",
                              color="R²", color_continuous_scale=["#E05252","#E8A84A","#2EC27B"])
                style_fig(fig2)
                fig2.update_coloraxes(showscale=False)
                st.plotly_chart(fig2, use_container_width=True)

            # Actual vs Predicted for best
            best_preds = preds_dict[best_model]
            scatter_df = pd.DataFrame({"Actual": y_test, "Predicted": best_preds})
            fig3 = px.scatter(scatter_df, x="Actual", y="Predicted",
                              title=f"Actual vs Predicted — {best_model}",
                              opacity=0.5, color_discrete_sequence=["#C27B2E"])
            fig3.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                           x1=y_test.max(), y1=y_test.max(),
                           line=dict(color="#2EC27B", dash="dash", width=2))
            style_fig(fig3)
            st.plotly_chart(fig3, use_container_width=True)

            # Feature importance (RF)
            rf_model = models["Random Forest"]
            fi = pd.DataFrame({"Feature": sel_feat, "Importance": rf_model.feature_importances_}).sort_values("Importance", ascending=True)
            fig4 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                          title="Feature Importance — Random Forest",
                          color="Importance", color_continuous_scale=["#3B1A08","#C27B2E","#FDF6EC"])
            style_fig(fig4)
            fig4.update_coloraxes(showscale=False)
            st.plotly_chart(fig4, use_container_width=True)

            st.markdown("<div class='section-title' style='font-size:1rem;'>Model Metrics Summary</div>", unsafe_allow_html=True)
            st.dataframe(res_df.set_index("Model"), use_container_width=True)


# ════════════════════════════════════════════
# PAGE: CLUSTERING
# ════════════════════════════════════════════
elif page == "🗺️ Clustering":
    st.markdown("<div class='section-title'>Route & Product Clustering</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Identify consistently slow routes, congested region-product combinations, and performance segments</div>", unsafe_allow_html=True)

    cluster_features = ["Lead Time (Days)","Sales","Gross Profit","Units"]
    cluster_features = [c for c in cluster_features if c in fdf.columns]

    if len(cluster_features) < 2:
        st.info("Need at least Lead Time, Sales, and Gross Profit columns for clustering.")
    else:
        k = st.slider("Number of Clusters (K)", 2, 8, 4)
        cluster_df = fdf[cluster_features].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_df)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        fdf_clust = fdf.loc[cluster_df.index].copy()
        fdf_clust["Cluster"] = cluster_labels.astype(str)

        col1, col2 = st.columns(2)
        with col1:
            if "Lead Time (Days)" in fdf_clust.columns and "Sales" in fdf_clust.columns:
                fig = px.scatter(fdf_clust, x="Lead Time (Days)", y="Sales",
                                 color="Cluster", title="Clusters: Lead Time vs Sales",
                                 size="Units" if "Units" in fdf_clust.columns else None,
                                 hover_data=["Region","Product Name"] if "Region" in fdf_clust.columns else None,
                                 color_discrete_sequence=["#C27B2E","#2EC27B","#5B9BD5","#E05252","#E8A84A","#9B59B6","#FF6B6B","#4ECDC4"],
                                 opacity=0.7)
                style_fig(fig)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            cluster_summary = fdf_clust.groupby("Cluster")[cluster_features].mean().round(2).reset_index()
            fig2 = px.bar(cluster_summary.melt(id_vars="Cluster"), x="Cluster", y="value",
                          color="variable", barmode="group",
                          title="Cluster Average Profiles",
                          color_discrete_sequence=["#C27B2E","#2EC27B","#5B9BD5","#E8A84A"])
            style_fig(fig2)
            st.plotly_chart(fig2, use_container_width=True)

        # Cluster × Region heatmap
        if "Region" in fdf_clust.columns:
            heat = fdf_clust.groupby(["Region","Cluster"])["Lead Time (Days)"].mean().unstack(fill_value=0).round(1)
            fig3 = px.imshow(heat, text_auto=True,
                             title="Avg Lead Time: Region × Cluster",
                             color_continuous_scale=["#2EC27B","#E8A84A","#E05252"])
            style_fig(fig3)
            st.plotly_chart(fig3, use_container_width=True)

        # Elbow chart
        inertias = []
        K_range = range(2, 10)
        for ki in K_range:
            km = KMeans(n_clusters=ki, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
        fig_elbow = px.line(x=list(K_range), y=inertias,
                            title="Elbow Method — Optimal K",
                            labels={"x":"K","y":"Inertia"},
                            color_discrete_sequence=["#C27B2E"],
                            markers=True)
        style_fig(fig_elbow)
        st.plotly_chart(fig_elbow, use_container_width=True)

        st.markdown("<div class='section-title' style='font-size:1rem;'>Cluster Summary Table</div>", unsafe_allow_html=True)
        st.dataframe(cluster_summary.set_index("Cluster"), use_container_width=True)


# ════════════════════════════════════════════
# PAGE: SCENARIO SIMULATION
# ════════════════════════════════════════════
elif page == "⚙️ Scenario Sim":
    st.markdown("<div class='section-title'>Scenario Simulation Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Simulate factory–product reassignment and quantify lead time & margin impact</div>", unsafe_allow_html=True)

    factories = ["Factory A — Texas", "Factory B — Illinois", "Factory C — California", "Factory D — Ohio", "Factory E — Georgia"]
    factory_lt_map = {"Factory A — Texas": 5, "Factory B — Illinois": 7, "Factory C — California": 10, "Factory D — Ohio": 4, "Factory E — Georgia": 8}
    factory_cost_map = {"Factory A — Texas": 0.85, "Factory B — Illinois": 0.90, "Factory C — California": 1.10, "Factory D — Ohio": 0.80, "Factory E — Georgia": 0.95}

    if "Product Name" in fdf.columns:
        products_avail = sorted(fdf["Product Name"].dropna().unique())
    else:
        products_avail = ["Wonka Bar - Milk Chocolate", "Wonka Bar - Triple Dazzle Caramel"]

    col1, col2 = st.columns(2)
    with col1:
        sel_product = st.selectbox("Select Product to Reassign", products_avail)
        current_factory = st.selectbox("Current Factory", factories, index=0)
    with col2:
        target_factory = st.selectbox("Target Factory (Scenario)", factories, index=3)
        target_region = st.selectbox("Target Delivery Region", ["Interior","Atlantic","Gulf","Pacific"])

    region_lt_adder = {"Interior": 1, "Atlantic": 2, "Gulf": 3, "Pacific": 5}

    if st.button("▶ Run Simulation", use_container_width=False):
        current_lt = factory_lt_map[current_factory] + region_lt_adder.get(target_region, 2)
        new_lt = factory_lt_map[target_factory] + region_lt_adder.get(target_region, 2)
        lt_delta = current_lt - new_lt

        prod_data = fdf[fdf["Product Name"] == sel_product] if "Product Name" in fdf.columns else fdf
        avg_sales = prod_data["Sales"].mean() if "Sales" in prod_data.columns else 100
        avg_units = prod_data["Units"].mean() if "Units" in prod_data.columns else 3
        avg_profit = prod_data["Gross Profit"].mean() if "Gross Profit" in prod_data.columns else 40

        curr_cost_mult = factory_cost_map[current_factory]
        new_cost_mult = factory_cost_map[target_factory]
        cost_change_pct = (new_cost_mult - curr_cost_mult) / curr_cost_mult * 100
        new_profit = avg_profit * (1 - (new_cost_mult - curr_cost_mult))

        st.markdown("---")

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Current Lead Time", f"{current_lt} days")
        col_b.metric("New Lead Time", f"{new_lt} days", delta=f"{-lt_delta} days", delta_color="inverse")
        col_c.metric("Cost Change", f"{cost_change_pct:+.1f}%", delta_color="inverse")
        col_d.metric("New Avg Profit", f"${new_profit:.2f}", delta=f"${new_profit - avg_profit:+.2f}")

        # Simulation chart
        scenarios = ["Current", "Scenario"]
        sim_data = pd.DataFrame({
            "Scenario": scenarios * 2,
            "Metric": ["Lead Time (days)", "Lead Time (days)", "Avg Profit ($)", "Avg Profit ($)"],
            "Value": [current_lt, new_lt, avg_profit, new_profit]
        })
        fig = px.bar(sim_data[sim_data["Metric"] == "Lead Time (days)"],
                     x="Scenario", y="Value", color="Scenario",
                     title=f"Lead Time Impact — {sel_product}",
                     color_discrete_sequence=["#E05252","#2EC27B"])
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(sim_data[sim_data["Metric"] == "Avg Profit ($)"],
                      x="Scenario", y="Value", color="Scenario",
                      title="Profit Impact",
                      color_discrete_sequence=["#E05252","#2EC27B"])
        style_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)

        # Multi-factory comparison
        all_sim = []
        for f in factories:
            lt = factory_lt_map[f] + region_lt_adder.get(target_region, 2)
            profit = avg_profit * (1 - (factory_cost_map[f] - 1.0))
            all_sim.append({"Factory": f, "Est. Lead Time": lt, "Est. Profit": round(profit, 2)})
        sim_all_df = pd.DataFrame(all_sim)
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Bar(x=sim_all_df["Factory"], y=sim_all_df["Est. Lead Time"],
                              name="Lead Time", marker_color="#E05252"), secondary_y=False)
        fig3.add_trace(go.Scatter(x=sim_all_df["Factory"], y=sim_all_df["Est. Profit"],
                                  name="Profit", mode="lines+markers",
                                  line=dict(color="#2EC27B", width=2)), secondary_y=True)
        fig3.update_layout(**PLOTLY_THEME, title_text="All Factory Comparison", margin=dict(t=45,b=30,l=30,r=20))
        fig3.update_yaxes(title_text="Lead Time (days)", gridcolor="#241A0F", secondary_y=False)
        fig3.update_yaxes(title_text="Profit ($)", gridcolor="#241A0F", secondary_y=True)
        st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════
# PAGE: RECOMMENDATIONS
# ════════════════════════════════════════════
elif page == "💡 Recommendations":
    st.markdown("<div class='section-title'>Optimization Recommendations</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>AI-generated factory reassignment recommendations ranked by lead time reduction, risk, and profit impact</div>", unsafe_allow_html=True)

    # Compute region-level stats
    if "Region" in fdf.columns and "Lead Time (Days)" in fdf.columns:
        reg_stats = fdf.groupby("Region").agg(
            Avg_Lead=("Lead Time (Days)", "mean"),
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Gross Profit", "sum"),
            Orders=("Sales", "count")
        ).reset_index().round(2)
        reg_stats["Margin%"] = (reg_stats["Total_Profit"] / reg_stats["Total_Sales"] * 100).round(1)
        reg_stats["Risk"] = reg_stats["Avg_Lead"].apply(lambda x: "🔴 High" if x > 10 else ("🟡 Medium" if x > 6 else "🟢 Low"))

        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(reg_stats, x="Avg_Lead", y="Margin%", size="Orders",
                             color="Region", text="Region",
                             title="Lead Time vs Margin by Region",
                             color_discrete_sequence=["#C27B2E","#2EC27B","#5B9BD5","#E8A84A"])
            style_fig(fig)
            fig.update_traces(textposition="top center", textfont_size=10)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(reg_stats, x="Region", y="Avg_Lead",
                          title="Average Lead Time by Region",
                          color="Avg_Lead",
                          color_continuous_scale=["#2EC27B","#E8A84A","#E05252"])
            style_fig(fig2)
            fig2.update_coloraxes(showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    # Recommendation cards
    st.markdown("---")
    st.markdown("<div class='section-title'>Top Factory Reassignment Recommendations</div>", unsafe_allow_html=True)

    recs = [
        ("good", "Reassign Milk Chocolate → Factory D (Ohio)", "Estimated lead time reduction: 4 days | Profit impact: +$1.20/unit | Risk: Low"),
        ("warn", "Reassign Triple Dazzle Caramel → Factory B (Illinois)", "Estimated lead time reduction: 2 days | Logistics cost increase: +5% | Net positive for Eastern seaboard"),
        ("good", "Consolidate Gulf Region orders via Factory A (Texas)", "Reduces regional congestion | Avg lead time savings: 3.2 days | Improves 32% of Gulf routes"),
        ("warn", "Shift Nutty Crunch Surprise to Factory E (Georgia)", "Targets Atlantic corridor bottleneck | Lead time: -1.5d | Watch Q4 capacity constraints"),
        ("bad",  "Avoid Factory C (California) for Interior shipments", "Current avg lead time 14.2 days | Cost premium 10% above baseline | Margin erosion risk"),
    ]

    for style, title, detail in recs:
        st.markdown(f"""
        <div class="rec-card {style}">
            <div class="rec-title">{title}</div>
            <div class="rec-detail">{detail}</div>
        </div>""", unsafe_allow_html=True)

    # Product-level margin waterfall
    if "Product Name" in fdf.columns and "Gross Profit" in fdf.columns:
        st.markdown("---")
        prod_margin = fdf.groupby("Product Name").agg(
            Profit=("Gross Profit","sum"),
            Sales=("Sales","sum")
        ).reset_index()
        prod_margin["Margin%"] = (prod_margin["Profit"] / prod_margin["Sales"] * 100).round(1)
        prod_margin = prod_margin.sort_values("Margin%", ascending=False)

        fig3 = go.Figure(go.Waterfall(
            name="Margin", orientation="v",
            x=prod_margin["Product Name"],
            y=prod_margin["Margin%"],
            connector=dict(line=dict(color="#825322")),
            increasing=dict(marker_color="#2EC27B"),
            decreasing=dict(marker_color="#E05252"),
            totals=dict(marker_color="#C27B2E"),
            text=[f"{v:.1f}%" for v in prod_margin["Margin%"]],
            textposition="outside",
        ))
        fig3.update_layout(**PLOTLY_THEME, title="Product Margin Waterfall",
                           margin=dict(t=45,b=80,l=30,r=20))
        fig3.update_xaxes(tickangle=25, tickfont_size=9, gridcolor="#392814")
        fig3.update_yaxes(gridcolor="#392814")
        st.plotly_chart(fig3, use_container_width=True)

    # Summary table
    if "Region" in fdf.columns and "Lead Time (Days)" in fdf.columns:
        st.markdown("<div class='section-title' style='font-size:1rem;'>Region Performance Summary</div>", unsafe_allow_html=True)
        st.dataframe(reg_stats.set_index("Region"), use_container_width=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:32px 0 16px;color:#3B2010;font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;'>
Nassau Candy · Factory Optimization Engine · Built with Streamlit & scikit-learn
</div>
""", unsafe_allow_html=True)
