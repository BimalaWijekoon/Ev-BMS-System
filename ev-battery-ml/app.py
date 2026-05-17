"""
EV Battery Intelligence System — Streamlit Web Interface
Premium UI for real-time battery health prediction and safety monitoring.
"""
import sys, os
# sys.path is already configured, no need to add parent
# We're at root level now

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json

from src.predict import load_models, predict_single, predict_batch, check_model_health
from src.simulator import BatterySimulator
from src.insights import BatteryInsights

# ─── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="EV Battery Intelligence",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 2rem; }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(99,102,241,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }
    .hero-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        position: relative;
        z-index: 1;
    }
    .hero-header p {
        color: #a5b4fc;
        font-size: 1.05rem;
        margin: 0;
        position: relative;
        z-index: 1;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #2a2a3e 100%);
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid rgba(99,102,241,0.2);
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99,102,241,0.15);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.2rem 0;
    }
    .metric-unit {
        font-size: 0.85rem;
        color: #64748b;
    }

    /* Status badges */
    .status-safe {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        color: #34d399;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        text-align: center;
        border: 1px solid rgba(52,211,153,0.3);
    }
    .status-warn {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        color: #fca5a5;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        text-align: center;
        border: 1px solid rgba(252,165,165,0.3);
        animation: glow 2s ease-in-out infinite;
    }
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 5px rgba(220,38,38,0.3); }
        50% { box-shadow: 0 0 20px rgba(220,38,38,0.5); }
    }

    /* Info box */
    .info-box {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #6366f1;
        margin: 1rem 0;
    }
    .info-box p { color: #cbd5e1; margin: 0; font-size: 0.9rem; line-height: 1.6; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h2 { color: #e2e8f0; }
    [data-testid="stSidebar"] label { color: #94a3b8 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15,23,42,0.5);
        padding: 4px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 500;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #475569;
        font-size: 0.8rem;
        border-top: 1px solid rgba(99,102,241,0.1);
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Models ───────────────────────────────────────────────────────
@st.cache_resource
def get_models():
    try:
        return load_models(), True
    except Exception as e:
        return None, False

MODELS, MODELS_LOADED = get_models()

# ─── Helper: Gauge Chart ──────────────────────────────────────────────
def make_gauge(value, title, min_val, max_val, color, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"size": 28, "color": "#e2e8f0"}},
        title={"text": title, "font": {"size": 14, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": "#475569",
                     "tickfont": {"color": "#64748b", "size": 10}},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [min_val, min_val + (max_val-min_val)*0.6], "color": "rgba(99,102,241,0.08)"},
                {"range": [min_val + (max_val-min_val)*0.6, max_val], "color": "rgba(239,68,68,0.08)"}
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 2},
                "thickness": 0.8,
                "value": min_val + (max_val-min_val)*0.8
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        height=220,
        margin=dict(l=20, r=20, t=40, b=10)
    )
    return fig

# ─── Hero Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🔋 EV Battery Intelligence System</h1>
    <p>Real-time battery health prediction & safety monitoring powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

if not MODELS_LOADED:
    st.error("⚠️ Models not loaded. Run notebooks 01–06 first to generate model artifacts.")
    st.stop()

# ─── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["⚡ Single Prediction", "📊 Batch Analysis", "📖 Model Info", "🏥 Model Health", "🔋 Charging Simulation"])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: Single Prediction
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Sensor Input Parameters")

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown("##### ⚡ Electrical Parameters")
        soc = st.slider("State of Charge (%)", 0.0, 100.0, 50.0, 0.1)
        soh = st.slider("State of Health", 0.50, 1.00, 0.95, 0.01)
        terminal_voltage = st.slider("Terminal Voltage (V)", 2.50, 4.30, 3.70, 0.01)
        battery_current = st.number_input("Battery Current (A)", value=15.0, step=0.5)
        action_current = st.number_input("Action Current (A)", value=15.0, step=0.5)
        action_voltage = st.number_input("Action Voltage (V)", value=3.70, step=0.01)

    with col_right:
        st.markdown("##### 🌡️ Thermal & Operational")
        battery_temp = st.number_input("Battery Temperature (°C)", value=35.0, step=0.5)
        ambient_temp = st.number_input("Ambient Temperature (°C)", value=25.0, step=0.5)
        internal_resistance = st.slider("Internal Resistance (Ω)", 0.010, 0.200, 0.050, 0.001)
        charging_efficiency = st.slider("Charging Efficiency", 0.70, 1.00, 0.85, 0.001)
        charging_time = st.number_input("Charging Time (s)", value=2500, step=50)

    st.markdown("---")

    if st.button("🔮  Run Prediction", type="primary", width='stretch'):
        raw_input = {
            'SOC': soc, 'SOH': soh, 'terminal_voltage': terminal_voltage,
            'battery_current': battery_current, 'battery_temp': battery_temp,
            'ambient_temp': ambient_temp, 'internal_resistance': internal_resistance,
            'action_current': action_current, 'action_voltage': action_voltage,
            'dT_dt': 0.0, 'dV_dt': 0.0, 'soc_delta': 0.0,
            'thermal_stress_index': min(battery_temp / 1129.0, 1.0),
            'aging_indicator': 1.0 - soh,
            'charging_efficiency': charging_efficiency,
            'charging_time': charging_time,
            'balancing_time': 15.0
        }

        with st.spinner("Running inference..."):
            result = predict_single(raw_input, MODELS)

        ir_pred = result['internal_resistance']
        temp_prob = result['over_temp_probability'] * 100
        temp_flag = result['over_temp_flag']
        volt_flag = result['over_voltage_flag']

        # ── Metric Cards ──
        st.markdown("### 📈 Prediction Results")
        c1, c2, c3, c4 = st.columns(4, gap="medium")

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Internal Resistance</div>
                <div class="metric-value" style="color: #818cf8;">{ir_pred:.6f}</div>
                <div class="metric-unit">Ohms (Ω)</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            prob_color = "#34d399" if temp_prob < 50 else "#fbbf24" if temp_prob < 75 else "#ef4444"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Over-Temp Probability</div>
                <div class="metric-value" style="color: {prob_color};">{temp_prob:.1f}%</div>
                <div class="metric-unit">Thermal Risk Score</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            status_cls = "status-safe" if temp_flag == 0 else "status-warn"
            status_txt = "🟢 SAFE" if temp_flag == 0 else "🔴 OVER-TEMP"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Temperature Status</div>
                <div class="{status_cls}" style="margin-top: 0.5rem;">{status_txt}</div>
            </div>""", unsafe_allow_html=True)

        with c4:
            status_cls_v = "status-safe" if volt_flag == 0 else "status-warn"
            status_txt_v = "🟢 SAFE" if volt_flag == 0 else "🔴 OVER-VOLT"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Voltage Status</div>
                <div class="{status_cls_v}" style="margin-top: 0.5rem;">{status_txt_v}</div>
            </div>""", unsafe_allow_html=True)

        # ── Gauges ──
        st.markdown("")
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(make_gauge(ir_pred * 1000, "IR (mΩ)", 10, 200, "#818cf8", " mΩ"),
                          width='stretch')
        with g2:
            gauge_color = "#34d399" if temp_prob < 50 else "#ef4444"
            st.plotly_chart(make_gauge(temp_prob, "Over-Temp Risk", 0, 100, gauge_color, "%"),
                          width='stretch')

        # ── Interpretation ──
        st.markdown("---")
        ir_ratio = ir_pred / internal_resistance if internal_resistance > 0 else 1.0
        if ir_ratio > 1.15:
            interp = "⚠️ **Predicted IR is significantly higher than measured** — potential rapid aging detected."
        elif ir_ratio < 0.85:
            interp = "ℹ️ **Predicted IR is lower than measured** — measurement noise possible, recheck sensor."
        else:
            interp = "✅ **Predicted IR aligns with measurement** — battery aging is within expected range."

        st.markdown(f"""
        <div class="info-box">
            <p>{interp}<br>
            <strong>Predicted:</strong> {ir_pred:.6f} Ω &nbsp;|&nbsp;
            <strong>Measured:</strong> {internal_resistance:.4f} Ω &nbsp;|&nbsp;
            <strong>Ratio:</strong> {ir_ratio:.3f}</p>
        </div>""", unsafe_allow_html=True)

        # ── INSIGHTS ANALYSIS ──
        st.markdown("---")
        st.markdown("### 📈 Battery Health Insights")
        
        # Initialize insights if needed
        if 'insights' not in st.session_state:
            try:
                st.session_state.insights = BatteryInsights(
                    csv_path=os.path.join(os.path.dirname(__file__), '..', 'data', 'nev_battery_charging.csv')
                )
            except Exception as e:
                st.warning(f"Could not load insights: {e}")
                st.session_state.insights = None
        
        if st.session_state.insights:
            # Get insights
            health_result = st.session_state.insights.assess_battery_health(
                ir_pred, temp_flag, volt_flag
            )
            fleet_comparison = st.session_state.insights.compare_to_fleet(
                ir_pred, charging_efficiency, temp_prob / 100.0
            )
            rul_result = st.session_state.insights.estimate_remaining_useful_life(ir_pred)
            recommendations = st.session_state.insights.get_recommendations(
                health_result['overall_health'], rul_result, health_result['alerts']
            )
            
            # Health Score
            health_val = health_result['overall_health']
            health_status = health_result['status']
            
            h1, h2 = st.columns([2, 1])
            
            with h1:
                st.plotly_chart(make_gauge(
                    health_val, "Battery Health", 0, 100,
                    "#ef4444" if health_val < 40 else "#fbbf24" if health_val < 60 else "#22c55e",
                    "%"
                ), width='stretch')
            
            with h2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Overall Status</div>
                    <div style="font-size: 1.3rem; margin-top: 0.8rem; font-weight: 700;">{health_status}</div>
                </div>""", unsafe_allow_html=True)
            
            # Component scores
            st.markdown("##### Health Components")
            comp_cols = st.columns(3)
            for idx, (comp_name, comp_score, _) in enumerate(health_result['component_scores']):
                with comp_cols[idx]:
                    color = "#22c55e" if comp_score >= 80 else "#fbbf24" if comp_score >= 60 else "#ef4444"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{comp_name}</div>
                        <div style="font-size: 1.5rem; color: {color}; font-weight: 700;">{comp_score:.0f}%</div>
                    </div>""", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Alerts
            if health_result['alerts']:
                st.markdown("##### 🚨 Active Alerts")
                for alert in health_result['alerts']:
                    if alert['severity'] == 'CRITICAL':
                        st.error(f"**{alert['type']}** — {alert['message']}")
                    else:
                        st.warning(f"**{alert['type']}** — {alert['message']}")
                st.markdown("")
            
            # RUL & Fleet Comparison
            rul_fleet_col1, rul_fleet_col2 = st.columns(2)
            
            with rul_fleet_col1:
                st.markdown("##### ⏱️ Remaining Useful Life")
                if 'estimated_cycles_remaining_conservative' in rul_result:
                    cycles = rul_result['estimated_cycles_remaining_conservative']
                    if cycles is not None:
                        cycle_color = "#ef4444" if cycles < 50 else "#fbbf24" if cycles < 500 else "#22c55e"
                        st.markdown(f"""
                        <div class="metric-card" style="border-left: 4px solid {cycle_color};">
                            <div class="metric-label">Cycles Remaining (Conservative)</div>
                            <div style="font-size: 1.8rem; color: {cycle_color}; font-weight: 700; margin: 0.5rem 0;">{cycles}</div>
                            <div style="font-size: 0.8rem; color: #94a3b8;">
                                Degradation: {rul_result['degradation_rate']:.6f} Ω/cycle
                            </div>
                        </div>""", unsafe_allow_html=True)
            
            with rul_fleet_col2:
                st.markdown("##### 📊 Fleet Comparison")
                fleet_ir = fleet_comparison['ir_vs_fleet']
                ir_status_color = "#ef4444" if fleet_ir['status'] == 'WORSE' else "#22c55e"
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {ir_status_color};">
                    <div class="metric-label">Battery vs Fleet</div>
                    <div style="font-size: 1.2rem; color: {ir_status_color}; font-weight: 700; margin: 0.3rem 0;">{fleet_ir['status']}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">
                        Percentile: {fleet_ir['percentile']:.1f}% ({fleet_ir['percentile']:.0f}th percentile)
                        <br>Deviation: {fleet_ir['deviation_sigmas']:+.2f}σ
                    </div>
                </div>""", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Recommendations
            st.markdown("##### 💡 Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")

# ═══════════════════════════════════════════════════════════════════════
# TAB 2: Batch Analysis
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Upload Battery Sensor Data")
    st.markdown("""
    <div class="info-box">
        <p>Upload a CSV file containing battery sensor readings. The file should include columns
        matching the training data format. Predictions will be appended as new columns.</p>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        df_input = pd.read_csv(uploaded)

        with st.expander("📋 Input Data Preview", expanded=True):
            st.dataframe(df_input.head(10), width='stretch')

        if st.button("🔮  Run Batch Prediction", type="primary", width='stretch'):
            with st.spinner("Processing batch..."):
                result_df = predict_batch(df_input, MODELS)

            st.success(f"✅ Predictions complete for {len(result_df)} rows")

            # Show predictions
            pred_cols = [c for c in result_df.columns if c.startswith('pred_')]
            st.dataframe(result_df[pred_cols].head(20), width='stretch')

            # Distribution charts
            if 'pred_internal_resistance' in result_df.columns:
                ch1, ch2 = st.columns(2)
                with ch1:
                    fig_ir = px.histogram(result_df, x='pred_internal_resistance',
                        nbins=40, title="Predicted Internal Resistance Distribution",
                        color_discrete_sequence=["#818cf8"])
                    fig_ir.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e2e8f0"), xaxis_title="IR (Ω)", yaxis_title="Count"
                    )
                    st.plotly_chart(fig_ir, width='stretch')

                with ch2:
                    if 'pred_over_temp_prob' in result_df.columns:
                        fig_tp = px.histogram(result_df, x='pred_over_temp_prob',
                            nbins=40, title="Over-Temp Probability Distribution",
                            color_discrete_sequence=["#f97316"])
                        fig_tp.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#e2e8f0"), xaxis_title="Probability", yaxis_title="Count"
                        )
                        st.plotly_chart(fig_tp, width='stretch')

            # Download
            csv_data = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions CSV", csv_data,
                file_name="ev_battery_predictions.csv", mime="text/csv",
                width='stretch')

# ═══════════════════════════════════════════════════════════════════════
# TAB 3: Model Info
# ═══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Model Architecture & Performance")

    # Load results
    try:
        with open(os.path.join(os.path.dirname(__file__), '..', 'models', 'regression_results.json')) as f:
            reg_results = json.load(f)
    except:
        reg_results = None

    try:
        with open(os.path.join(os.path.dirname(__file__), '..', 'models', 'classification_results.json')) as f:
            cls_results = json.load(f)
    except:
        cls_results = None

    r1, r2 = st.columns(2, gap="large")

    with r1:
        st.markdown("#### 📈 Regression — Internal Resistance")
        st.markdown("""
        <div class="info-box">
            <p><strong>Target:</strong> internal_resistance (Ω)<br>
            <strong>Why:</strong> cycle_degradation has r=0.038 (synthetic noise). IR achieves R²≈0.97.<br>
            <strong>Method:</strong> GridSearchCV + TimeSeriesSplit (chronological)</p>
        </div>""", unsafe_allow_html=True)

        if reg_results:
            models = list(reg_results.keys())
            r2_vals = [reg_results[m]['R2'] for m in models]
            rmse_vals = [reg_results[m]['RMSE'] for m in models]
            colors = ['#6366f1', '#22c55e', '#f97316']

            fig_reg = go.Figure()
            fig_reg.add_trace(go.Bar(x=models, y=r2_vals, name="R²",
                marker_color=colors, text=[f"{v:.4f}" for v in r2_vals], textposition='outside'))
            fig_reg.update_layout(
                title="R² Score by Model", yaxis_range=[0, 1.1],
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"), height=350
            )
            st.plotly_chart(fig_reg, width='stretch')

    with r2:
        st.markdown("#### 🎯 Classification — Over-Temperature")
        st.markdown("""
        <div class="info-box">
            <p><strong>Target:</strong> over_temp_flag (binary)<br>
            <strong>CV:</strong> StratifiedKFold (TimeSeriesSplit produces single-class folds)<br>
            <strong>Voltage:</strong> Rule-based fallback (action_voltage > 4.15V)</p>
        </div>""", unsafe_allow_html=True)

        if cls_results:
            models_cls = list(cls_results.keys())
            f1_vals = [cls_results[m]['F1'] for m in models_cls]
            recall_vals = [cls_results[m]['Recall'] for m in models_cls]

            fig_cls = go.Figure()
            fig_cls.add_trace(go.Bar(x=models_cls, y=f1_vals, name="F1",
                marker_color=['#6366f1','#22c55e','#f97316'],
                text=[f"{v:.3f}" for v in f1_vals], textposition='outside'))
            fig_cls.update_layout(
                title="F1 Score by Model", yaxis_range=[0, 1.1],
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"), height=350
            )
            st.plotly_chart(fig_cls, width='stretch')

    # Feature columns
    with st.expander("🔧 Selected Features"):
        try:
            with open(os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_columns.json')) as f:
                feats = json.load(f)
            for i, feat in enumerate(feats, 1):
                st.code(f"{i:2d}. {feat}", language=None)
        except:
            st.info("Feature columns not found.")

# ═══════════════════════════════════════════════════════════════════════
# TAB 4: Model Health Check
# ═══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🏥 Model Health Status")
    st.markdown("""
    <div class="info-box">
        <p>Real-time health check for all model components, preprocessor, and configuration files.</p>
    </div>""", unsafe_allow_html=True)
    
    if st.button("🔍 Run Health Check", type="primary", width='stretch'):
        with st.spinner("Checking model components..."):
            health = check_model_health(os.path.join(os.path.dirname(__file__), '..', 'models'))
        
        # Overall status
        status_icon = "✅" if health['status'] == 'healthy' else "⚠️"
        st.markdown(f"## {status_icon} Overall Status: **{health['status'].upper()}**")
        
        # Display components
        st.markdown("### Component Details")
        
        for comp_name, comp_data in health['components'].items():
            col1, col2 = st.columns([2, 3])
            
            with col1:
                if comp_data.get('status') == 'ok':
                    st.markdown(f"✅ **{comp_name}**")
                elif comp_data.get('status') == 'missing':
                    st.markdown(f"❌ **{comp_name}**")
                else:
                    st.markdown(f"⚠️ **{comp_name}**")
            
            with col2:
                if comp_data.get('status') == 'ok':
                    # Show details for OK components
                    details = []
                    for key, val in comp_data.items():
                        if key != 'status':
                            if isinstance(val, bool):
                                val = "✓" if val else "✗"
                            details.append(f"**{key}**: {val}")
                    st.text(" | ".join(details) if details else "Status: OK")
                elif comp_data.get('status') == 'missing':
                    st.error("File not found")
                else:
                    st.error(f"Error: {comp_data.get('error', 'Unknown error')}")
        
        # Show errors if any
        if health['errors']:
            st.markdown("### ❌ Errors Detected")
            for error in health['errors']:
                st.error(error)
        else:
            st.markdown("### ✅ All systems nominal!")
            st.success("All model components are healthy and ready for predictions.")

# ═══════════════════════════════════════════════════════════════════════
# TAB 5: Battery Charging Simulation
# ═══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🔋 Interactive Battery Charging Simulation")
    st.markdown("""
    <div class="info-box">
        <p>Simulate a complete battery charging cycle from 0% to 100%. Adjust input features 
        manually and watch real-time model predictions as the battery charges.</p>
    </div>""", unsafe_allow_html=True)
    
    # Initialize simulator if not already done
    if 'simulator' not in st.session_state:
        try:
            st.session_state.simulator = BatterySimulator(
                data_path=os.path.join(os.path.dirname(__file__), '..', 'data', 'nev_battery_charging.csv')
            )
            st.session_state.sim_state = st.session_state.simulator.get_initial_state()
            st.session_state.sim_step = 0
            st.session_state.sim_running = False
        except Exception as e:
            st.error(f"❌ Failed to initialize simulator: {e}")
            st.stop()
    
    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    
    with col_ctrl1:
        soc_increment = st.slider("SOC Increment per Step (%)", 0.1, 5.0, 1.0, 0.1)
    
    with col_ctrl2:
        speed = st.slider("Speed (steps per second)", 0.5, 5.0, 1.0, 0.5)
    
    with col_ctrl3:
        st.write("")  # Spacing
    
    # Buttons
    button_col1, button_col2, button_col3, button_col4 = st.columns(4)
    
    with button_col1:
        if st.button("▶️  Start" if not st.session_state.sim_running else "⏸️ Pause", width='stretch'):
            st.session_state.sim_running = not st.session_state.sim_running
    
    with button_col2:
        if st.button("⏭️  Step", width='stretch'):
            st.session_state.sim_step += 1
            st.session_state.sim_state = st.session_state.simulator.step(
                st.session_state.sim_state, soc_increment
            )
    
    with button_col3:
        if st.button("🔄 Reset", width='stretch'):
            st.session_state.sim_state = st.session_state.simulator.get_initial_state()
            st.session_state.sim_step = 0
            st.session_state.sim_running = False
    
    with button_col4:
        if st.button("🎯 Run to 100%", width='stretch'):
            with st.spinner("Simulating charge cycle..."):
                while st.session_state.sim_state['raw']['SOC'] < 99.0:
                    st.session_state.sim_state = st.session_state.simulator.step(
                        st.session_state.sim_state, soc_increment
                    )
                    st.session_state.sim_step += 1
            st.session_state.sim_running = False
    
    st.markdown("---")
    
    # Main simulation display
    st.markdown("### 🔋 Charging Progress")
    
    # Battery level progress bar
    soc_val = st.session_state.sim_state['raw']['SOC']
    
    st.progress(
        min(soc_val / 100.0, 1.0),
        text=f"SOC: {soc_val:.1f}%"
    )
    
    st.markdown("---")
    
    # Display engineered features
    st.markdown("### 📊 15 RFE-Selected Engineered Features (For Models)")
    st.write("These features were selected via Recursive Feature Elimination in Notebook 03:")
    
    # Show engineered features in a clean format
    eng_state = st.session_state.sim_state['engineered']
    cols = st.columns(3)
    
    for idx, (feat_name, feat_val) in enumerate(eng_state.items()):
        col_idx = idx % 3
        with cols[col_idx]:
            val_str = f"{feat_val:.6f}" if abs(feat_val) < 1 else f"{feat_val:.3f}"
            st.metric(feat_name, val_str)
    
    st.markdown("---")
    
    # Make prediction with engineered state
    st.markdown("### 🔮 Real-Time Model Predictions")
    
    try:
        # Use engineered state for prediction
        pred_result = predict_single(st.session_state.sim_state['engineered'], MODELS)
        
        # Prediction displays
        p1, p2, p3, p4 = st.columns(4)
        
        with p1:
            ir_pred = pred_result['internal_resistance']
            st.plotly_chart(make_gauge(
                ir_pred * 1000, 
                "Predicted IR", 10, 200, "#818cf8", "mΩ"
            ), width='stretch')
        
        with p2:
            temp_prob = pred_result['over_temp_probability'] * 100
            gauge_color = "#34d399" if temp_prob < 50 else "#fbbf24" if temp_prob < 80 else "#ef4444"
            st.plotly_chart(make_gauge(
                temp_prob, 
                "Over-Temp Risk", 0, 100, gauge_color, "%"
            ), width='stretch')
        
        with p3:
            temp_flag = pred_result['over_temp_flag']
            status_cls = "status-safe" if temp_flag == 0 else "status-warn"
            status_txt = "✅ SAFE" if temp_flag == 0 else "🔴 OVER-TEMP"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Temp Status</div>
                <div class="{status_cls}" style="margin-top: 0.5rem;">{status_txt}</div>
            </div>""", unsafe_allow_html=True)
        
        with p4:
            volt_flag = pred_result['over_voltage_flag']
            status_cls_v = "status-safe" if volt_flag == 0 else "status-warn"
            status_txt_v = "✅ SAFE" if volt_flag == 0 else "🔴 OVER-VOLT"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Voltage Status</div>
                <div class="{status_cls_v}" style="margin-top: 0.5rem;">{status_txt_v}</div>
            </div>""", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {str(e)[:100]}")
    
    st.markdown("---")
    
    # Info box
    st.markdown("""
    <div class="info-box">
        <p><strong>ℹ️ How it works:</strong>
        <br>1️⃣ <strong>Raw state:</strong> SOC increments each step (0→100%)
        <br>2️⃣ <strong>Engineering:</strong> Raw features are transformed to 15 engineered features
        <br>3️⃣ <strong>Prediction:</strong> Models make predictions on the 15 engineered features
        <br><br><strong>15 Engineered Features Selected by RFE:</strong>
        <br>SOH • battery_current • ambient_temp • action_current • action_voltage
        <br>dV_dt • soc_delta • charging_efficiency • charging_time • balancing_time
        <br>soc_range_rolling • thermal_acceleration • voltage_efficiency
        <br>internal_resistance_sq • aging_indicator_sq
        </p>
    </div>""", unsafe_allow_html=True)
    
    # Auto-step if running
    if st.session_state.sim_running and st.session_state.sim_state['raw']['SOC'] < 100.0:
        import time
        st.session_state.sim_state = st.session_state.simulator.step(
            st.session_state.sim_state, soc_increment
        )
        st.session_state.sim_step += 1
        time.sleep(1 / speed)
        st.rerun()

# ─── Footer ───────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    EV Battery Intelligence System • Built with Streamlit • scikit-learn • XGBoost<br>
    Regression: internal_resistance (R² ≈ 0.97) • Classification: StratifiedKFold CV
</div>
""", unsafe_allow_html=True)
