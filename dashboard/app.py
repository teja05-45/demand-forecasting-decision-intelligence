# ============================================================
# PATH FIX — REQUIRED FOR STREAMLIT + SRC PACKAGE
# ============================================================
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import joblib

from src.decision.capacity_model import estimate_capacity
from src.decision.risk_detection import (
    detect_capacity_risk,
    detect_risk_with_uncertainty,
    suppress_redundant_alerts,
    assign_root_cause
)
from src.decision.cost_analysis import calculate_expected_cost
from src.decision.what_if_simulation import backtest_resource_decision

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Demand Forecasting & Decision Intelligence",
    layout="wide"
)

# ============================================================
# ENTERPRISE UI THEME
# ============================================================
st.markdown(
    """
    <style>
    .stApp { background-color: #f8fafc; }
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    h1, h2, h3 { color: #0f172a; }
    button[data-baseweb="tab"] {
        font-size: 15px;
        font-weight: 600;
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================
st.title("📊 Demand Forecasting & Decision Intelligence")
st.subheader("Enterprise ML System for Proactive Planning & Risk Control")
st.caption("Forecast demand • Detect risk early • Simulate decisions • Optimize cost")

# ============================================================
# LOAD DATA & MODEL
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/forecast_features.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_resource
def load_model():
    return joblib.load("models/demand_forecast_model.pkl")

base_df = load_data()
model = load_model()

# ============================================================
# FORECAST + UNCERTAINTY
# ============================================================
def add_forecast_with_uncertainty(df, model, confidence=0.9):
    features = [
        "demand_lag_1", "demand_lag_7", "demand_lag_14",
        "rolling_mean_7", "rolling_std_7", "rolling_mean_14",
        "day_of_week", "is_weekend",
        "avg_resolution_time", "active_resources",
        "backlog", "demand_growth_rate"
    ]

    df = df.copy()
    df["forecast"] = model.predict(df[features])

    residual_std = (df["demand"] - df["forecast"]).std()
    z = 1.65 if confidence == 0.9 else 1.96

    df["forecast_lower"] = df["forecast"] - z * residual_std
    df["forecast_upper"] = df["forecast"] + z * residual_std

    return df

# ============================================================
# RECOMPUTE PIPELINE
# ============================================================
def recompute_pipeline(df, model, demand_change, resource_change):
    df = df.copy()

    df["demand"] = df["demand"] * (1 + demand_change)
    df["active_resources"] = df["active_resources"] + resource_change

    df = add_forecast_with_uncertainty(df, model)
    df = estimate_capacity(df)
    df = detect_capacity_risk(df)
    df = detect_risk_with_uncertainty(df)
    df = suppress_redundant_alerts(df)
    df = assign_root_cause(df)
    df = calculate_expected_cost(df)

    return df

# ============================================================
# SAFE KPI HELPER
# ============================================================
def safe_int(value, default=0):
    if pd.isna(value):
        return default
    return int(value)

# ============================================================
# SIDEBAR — CONTROLS
# ============================================================
st.sidebar.title("🧠 Decision Controls")

date_range = st.sidebar.date_input(
    "📅 Date Range",
    [base_df["date"].min(), base_df["date"].max()],
    min_value=base_df["date"].min(),
    max_value=base_df["date"].max()
)

risk_filter = st.sidebar.multiselect(
    "⚠️ Risk Severity",
    ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    default=["HIGH", "CRITICAL"]
)

scenario = st.sidebar.selectbox(
    "🎯 Scenario Preset",
    ["Baseline", "Conservative", "Aggressive"]
)

demand_change = st.sidebar.slider("📈 Demand Change (%)", -30, 50, 20) / 100
resource_change = st.sidebar.slider("👥 Resource Change", -5, 10, 3)

if scenario == "Conservative":
    demand_change = max(demand_change, 0.1)
    resource_change = max(resource_change, 2)
elif scenario == "Aggressive":
    demand_change = max(demand_change, 0.3)
    resource_change = max(resource_change, 5)

apply_clicked = st.sidebar.button("✅ Apply Scenario")

# ============================================================
# SESSION STATE
# ============================================================
if "df_scenario" not in st.session_state:
    st.session_state.df_scenario = recompute_pipeline(
        base_df, model, 0.0, 0
    )

if apply_clicked:
    with st.spinner("Recomputing forecasts & decisions..."):
        st.session_state.df_scenario = recompute_pipeline(
            base_df, model, demand_change, resource_change
        )

df = st.session_state.df_scenario

# ============================================================
# FILTERED VIEW
# ============================================================
df_view = df[
    (df["date"] >= pd.to_datetime(date_range[0])) &
    (df["date"] <= pd.to_datetime(date_range[1])) &
    (df["risk_severity"].isin(risk_filter))
]

# ============================================================
# KPIs (SAFE)
# ============================================================
k1, k2, k3, k4 = st.columns(4)

avg_demand = safe_int(df_view["demand"].mean())
critical_days = safe_int((df_view["risk_severity"] == "CRITICAL").sum())
total_cost = safe_int(df_view["total_expected_cost"].sum())
utilization = (
    df_view["demand"].sum() / df_view["estimated_capacity"].sum()
    if len(df_view) > 0 else 0
)

k1.metric("Avg Daily Demand", avg_demand)
k2.metric("Critical Risk Days", critical_days)
k3.metric("Total Est. Cost (₹)", f"{total_cost:,}")
k4.metric("Capacity Utilization", f"{int(utilization * 100)}%")

if df_view.empty:
    st.warning(
        "No data available for selected filters. "
        "Adjust date range or risk severity."
    )

st.divider()

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Executive Overview",
    "📈 Forecast & Uncertainty",
    "⚠️ Risk & Root Cause",
    "🧪 Scenario Impact",
    "💰 Cost & Backtesting"
])

# ------------------------------------------------------------
with tab1:
    st.subheader("Demand vs Capacity")
    st.line_chart(df_view.set_index("date")[["demand", "estimated_capacity"]])

    st.subheader("Risk Distribution")
    st.bar_chart(df_view["risk_severity"].value_counts())

# ------------------------------------------------------------
with tab2:
    st.subheader("Forecast with Confidence Bounds")
    st.line_chart(
        df_view.set_index("date")[["forecast", "forecast_lower", "forecast_upper"]]
    )
    st.info("Decisions are based on worst-case (upper-bound) forecasts.")

# ------------------------------------------------------------
with tab3:
    st.subheader("Risk Details")
    st.dataframe(
        df_view[
            ["date", "demand", "estimated_capacity",
             "risk_severity", "root_cause", "alert_allowed"]
        ].tail(20),
        use_container_width=True
    )

# ------------------------------------------------------------
with tab4:
    st.subheader("Scenario Summary")
    st.success(
        f"Scenario: {scenario} | "
        f"Demand Change: {int(demand_change*100)}% | "
        f"Resource Change: {resource_change}"
    )

# ------------------------------------------------------------
with tab5:
    st.subheader("Historical Decision Backtesting")

    backtest = backtest_resource_decision(df_view, resource_change=3)

    c1, c2, c3 = st.columns(3)
    c1.metric("Actual Risk Days", backtest["actual_risk_days"])
    c2.metric("Simulated Risk Days", backtest["simulated_risk_days"])
    c3.metric("Risk Days Avoided", backtest["risk_days_avoided"])

    st.info("Backtesting quantifies how proactive decisions reduce risk.")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Enterprise ML System • Demand Forecasting • Decision Intelligence • Risk Management"
)
