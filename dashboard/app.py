import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

st.set_page_config(page_title='CyberShield AI', layout='wide', page_icon='🛡️')

st.markdown("""
<style>
    /* Dark premium theme */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #101829 50%, #0d1321 100%);
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1f35 0%, #1e2740 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .stDataFrame {
        border-radius: 8px;
    }
    h1 {
        background: linear-gradient(90deg, #6366f1, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stMetric label {
        color: #94a3b8 !important;
    }
    .stMetric > div > div {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_alerts():
    """Load alerts from pipeline output."""
    alerts_path = config.DATA_DIR / "alerts.csv"
    if not os.path.exists(alerts_path):
        return None
    df = pd.read_csv(alerts_path)
    return df


@st.cache_data
def load_access_logs():
    """Load raw access logs."""
    if not os.path.exists(config.ACCESS_LOG_PATH):
        return None
    return pd.read_csv(config.ACCESS_LOG_PATH)


@st.cache_data
def load_ground_truth():
    """Load ground truth labels."""
    if not os.path.exists(config.GROUND_TRUTH_PATH):
        return None
    return pd.read_csv(config.GROUND_TRUTH_PATH)


# ─── Title ───
st.title("🛡️ CyberShield AI")
st.caption("Behavioral Anomaly Detection for Cybersecurity")

# ─── Load Data ───
alerts_df = load_alerts()
logs_df = load_access_logs()
truth_df = load_ground_truth()

if alerts_df is None or logs_df is None:
    st.warning(
        "⚠️ Data files not found. Please run **`python pipeline.py`** first "
        "to generate data and train models."
    )
    st.stop()

# ─── Sidebar Navigation ───
page = st.sidebar.radio(
    "📌 Navigation",
    ["System Overview", "Alert Queue", "Entity Investigation", "Alert Detail"],
    index=0,
)

# ─── Helpers ───
anomaly_df = alerts_df[alerts_df["true_label"] != "normal"].copy()
total_events = len(logs_df)
total_alerts = len(anomaly_df)
alert_rate = (total_alerts / total_events) * 100 if total_events > 0 else 0

# ═══════════════════════════════════════════════════════════════
# PAGE: System Overview
# ═══════════════════════════════════════════════════════════════
if page == "System Overview":
    st.header("📊 System Overview")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    top_attack = anomaly_df["predicted_attack_type"].mode().iloc[0] if len(anomaly_df) > 0 else "N/A"

    col1.metric("Total Events", f"{total_events:,}")
    col2.metric("Total Alerts", f"{total_alerts:,}")
    col3.metric("Alert Rate", f"{alert_rate:.2f}%")
    col4.metric("Top Attack", top_attack)

    st.divider()

    # Charts row
    colA, colB = st.columns(2)

    with colA:
        if len(anomaly_df) > 0:
            attack_counts = anomaly_df["predicted_attack_type"].value_counts().reset_index()
            attack_counts.columns = ["Attack Type", "Count"]
            fig_pie = px.pie(
                attack_counts, values="Count", names="Attack Type",
                title="Attack Type Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with colB:
        if "risk_score" in alerts_df.columns:
            fig_hist = px.histogram(
                alerts_df, x="risk_score", nbins=50,
                title="Risk Score Distribution",
                color_discrete_sequence=["#6366f1"],
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # Alert volume over time
    if "timestamp" in alerts_df.columns and len(anomaly_df) > 0:
        vol_df = anomaly_df.copy()
        vol_df["date"] = pd.to_datetime(vol_df["timestamp"]).dt.date
        vol = vol_df.groupby("date").size().reset_index(name="count")
        fig_line = px.area(
            vol, x="date", y="count",
            title="Alert Volume Over Time",
            color_discrete_sequence=["#06b6d4"],
        )
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
        )
        st.plotly_chart(fig_line, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: Alert Queue
# ═══════════════════════════════════════════════════════════════
elif page == "Alert Queue":
    st.header("🚨 Alert Queue")

    # Filters
    colF1, colF2 = st.columns(2)
    with colF1:
        attack_filter = st.multiselect(
            "Filter by Attack Type",
            options=config.ATTACK_TYPES,
            default=[],
        )
    with colF2:
        min_score = st.slider("Minimum Risk Score", 0.0, 1.0, 0.0, 0.05)

    # Filter alerts
    filtered = anomaly_df.copy()
    if attack_filter:
        filtered = filtered[filtered["predicted_attack_type"].isin(attack_filter)]
    filtered = filtered[filtered["risk_score"] >= min_score]
    filtered = filtered.sort_values("risk_score", ascending=False).head(
        config.DASHBOARD_PAGE_SIZE
    )

    if len(filtered) == 0:
        st.info("No alerts match the current filters.")
    else:
        filtered = filtered.reset_index(drop=True)
        filtered.insert(0, "Rank", range(1, len(filtered) + 1))

        display_cols = [
            "Rank", "entity_id", "timestamp",
            "predicted_attack_type", "risk_score", "top_reason",
        ]
        available_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[available_cols],
            use_container_width=True,
            height=500,
        )

        st.caption(f"Showing {len(filtered)} alerts")


# ═══════════════════════════════════════════════════════════════
# PAGE: Entity Investigation
# ═══════════════════════════════════════════════════════════════
elif page == "Entity Investigation":
    st.header("🔍 Entity Investigation")

    # Entity search
    entity_ids = sorted(alerts_df["entity_id"].unique())
    entity_search = st.selectbox(
        "Search / Select Entity ID", options=[""] + list(entity_ids)
    )

    if entity_search:
        entity_events = alerts_df[alerts_df["entity_id"] == entity_search].copy()

        if entity_events.empty:
            st.warning("No events found for this entity.")
        else:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            n_events = len(entity_events)
            n_anomalies = (entity_events["true_label"] != "normal").sum()
            avg_risk = entity_events["risk_score"].mean()

            col1.metric("Total Events", n_events)
            col2.metric("Anomalous Events", n_anomalies)
            col3.metric("Avg Risk Score", f"{avg_risk:.3f}")

            st.divider()

            # Timeline
            if "timestamp" in entity_events.columns:
                entity_events["timestamp_dt"] = pd.to_datetime(
                    entity_events["timestamp"]
                )
                fig = px.scatter(
                    entity_events,
                    x="timestamp_dt", y="risk_score",
                    color="true_label",
                    title=f"Risk Score Timeline — {entity_search}",
                    color_discrete_sequence=px.colors.qualitative.Set1,
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Raw events table
            st.subheader("Event Log")
            st.dataframe(entity_events, use_container_width=True, height=300)


# ═══════════════════════════════════════════════════════════════
# PAGE: Alert Detail
# ═══════════════════════════════════════════════════════════════
elif page == "Alert Detail":
    st.header("🔬 Alert Detail")

    if len(anomaly_df) == 0:
        st.info("No anomaly alerts to display.")
    else:
        top_alerts = anomaly_df.sort_values("risk_score", ascending=False).head(20)
        selected_idx = st.selectbox(
            "Select an alert to investigate",
            range(len(top_alerts)),
            format_func=lambda i: (
                f"{top_alerts.iloc[i]['entity_id']} | "
                f"Score: {top_alerts.iloc[i]['risk_score']:.3f} | "
                f"{top_alerts.iloc[i]['predicted_attack_type']}"
            ),
        )

        alert = top_alerts.iloc[selected_idx]

        # Alert header
        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Score", f"{alert['risk_score']:.3f}")
        col2.metric("Attack Type", alert["predicted_attack_type"])
        col3.metric("True Label", alert["true_label"])

        st.divider()

        # Human-readable explanation
        st.subheader("🧠 Explanation")
        if "top_reason" in alert.index:
            st.info(alert["top_reason"])
        else:
            st.info("No explanation available.")

        # Feature values
        st.subheader("📊 Contributing Factors")
        feature_vals = {}
        for f in config.FEATURE_NAMES:
            if f in alert.index:
                val = alert[f]
                if abs(val) > 0.01:
                    feature_vals[f] = val

        if feature_vals:
            sorted_feats = dict(
                sorted(feature_vals.items(), key=lambda x: abs(x[1]), reverse=True)[
                    :10
                ]
            )
            fig_bar = px.bar(
                x=list(sorted_feats.keys()),
                y=list(sorted_feats.values()),
                title="Top Feature Values",
                color=list(sorted_feats.values()),
                color_continuous_scale="RdYlGn_r",
            )
            fig_bar.update_layout(
                xaxis_title="Feature",
                yaxis_title="Value",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Full event details
        st.subheader("📋 Full Event Details")
        st.json(alert.to_dict())
