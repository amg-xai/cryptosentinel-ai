"""
Reusable Streamlit UI components.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


def render_kpi_cards(health: dict, alerts: dict, graph_stats: dict):
    """Top row KPI metrics."""
    col1, col2, col3, col4 = st.columns(4)

    stats = alerts.get("stats", {})
    by_tier = stats.get("by_tier", {})

    with col1:
        st.metric(
            label="🔍 Active Alerts",
            value=stats.get("total_active", 0),
            delta=f"+{stats.get('total_created', 0)} total created",
        )
    with col2:
        critical = by_tier.get("EMERGENCY_ESCALATE", 0)
        st.metric(
            label="🚨 Critical Threats",
            value=critical,
            delta="Require immediate action" if critical > 0 else "None active",
            delta_color="inverse",
        )
    with col3:
        st.metric(
            label="🕸️ Graph Nodes",
            value=graph_stats.get("num_nodes", 0),
            delta=f"{graph_stats.get('num_edges', 0)} edges",
        )
    with col4:
        models = health.get("models_loaded", {})
        loaded = sum(1 for v in models.values() if v)
        total = len(models)
        st.metric(
            label="🤖 Models Active",
            value=f"{loaded}/{total}",
            delta="All systems operational" if loaded == total else "Some models offline",
        )


def severity_badge(severity: str) -> str:
    colors = {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🟢",
    }
    return colors.get(severity, "⚪") + " " + severity


def render_alert_table(alerts: list):
    """Render alert feed as styled table."""
    if not alerts:
        st.info("No active alerts. System monitoring...")
        return

    for alert in alerts[:20]:
        severity = alert.get("severity", "LOW")
        score = alert.get("composite_score", 0)
        address = alert.get("address", "")[:20] + "..."
        action = alert.get("action", "")
        value = alert.get("value_at_risk_eth", 0)

        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
        with col1:
            st.text(address)
        with col2:
            st.text(severity_badge(severity))
        with col3:
            st.progress(score, text=f"{score:.2f}")
        with col4:
            st.text(f"{value:.3f} ETH")
        with col5:
            st.text(action.replace("_", " "))


def render_risk_distribution(alerts: list):
    """Plotly histogram of risk score distribution."""
    if not alerts:
        return

    scores = [a.get("composite_score", 0) for a in alerts]
    fig = go.Figure(go.Histogram(
        x=scores,
        nbinsx=20,
        marker_color="#00D4AA",
        opacity=0.8,
    ))
    fig.update_layout(
        title="Risk Score Distribution",
        xaxis_title="Risk Score",
        yaxis_title="Count",
        paper_bgcolor="#161B22",
        plot_bgcolor="#0D1117",
        font_color="#E6EDF3",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_tier_breakdown(stats: dict):
    """Plotly pie chart of alert tiers."""
    by_tier = stats.get("by_tier", {})
    if not any(by_tier.values()):
        return

    labels = list(by_tier.keys())
    values = list(by_tier.values())
    colors = ["#ef4444", "#f97316", "#eab308", "#22c55e"]

    fig = go.Figure(go.Pie(
        labels=[l.replace("_", " ") for l in labels],
        values=values,
        marker_colors=colors,
        hole=0.4,
    ))
    fig.update_layout(
        title="Alert Tier Breakdown",
        paper_bgcolor="#161B22",
        font_color="#E6EDF3",
        height=300,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_vulnerability_chart(scan_result: dict):
    """Bar chart of vulnerability severity counts."""
    counts = scan_result.get("vulnerability_count", {})
    if not counts:
        return

    severities = list(counts.keys())
    values = list(counts.values())
    colors = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#6b7280"]

    fig = go.Figure(go.Bar(
        x=severities,
        y=values,
        marker_color=colors,
    ))
    fig.update_layout(
        title="Vulnerability Severity Breakdown",
        paper_bgcolor="#161B22",
        plot_bgcolor="#0D1117",
        font_color="#E6EDF3",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)
