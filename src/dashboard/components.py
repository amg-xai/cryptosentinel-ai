"""
Reusable Streamlit UI components.
"""

import plotly.graph_objects as go
import streamlit as st


def render_kpi_cards(health: dict, alerts: dict, graph_stats: dict):
    """Top row KPI metrics."""
    col1, col2, col3, col4 = st.columns(4)
    stats = alerts.get("stats", {})
    # DB-backed stats shape: {total, by_severity:{MEDIUM,HIGH,CRITICAL}, last_24h}
    by_sev = stats.get("by_severity", {})
    total_alerts = stats.get("total", alerts.get("total_active", 0))
    with col1:
        st.metric(
            label="🔍 Active Alerts",
            value=total_alerts,
            delta=f"+{stats.get('last_24h', 0)} in last 24h",
        )
    with col2:
        critical = by_sev.get("CRITICAL", 0)
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
            delta=(
                "All systems operational" if loaded == total else "Some models offline"
            ),
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
    fig = go.Figure(
        go.Histogram(
            x=scores,
            nbinsx=20,
            marker_color="#00D4AA",
            opacity=0.8,
        )
    )
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

    fig = go.Figure(
        go.Pie(
            labels=[l.replace("_", " ") for l in labels],
            values=values,
            marker_colors=colors,
            hole=0.4,
        )
    )
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

    fig = go.Figure(
        go.Bar(
            x=severities,
            y=values,
            marker_color=colors,
        )
    )
    fig.update_layout(
        title="Vulnerability Severity Breakdown",
        paper_bgcolor="#161B22",
        plot_bgcolor="#0D1117",
        font_color="#E6EDF3",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pyvis_graph(
    nodes: list,
    edges: list,
    highlight_address: str = "",
) -> str:
    """
    Render an interactive PyVis network graph.
    Returns HTML string for embedding in Streamlit.

    nodes: list of {id, risk_score, label, tx_count}
    edges: list of {from, to, value_eth}
    """
    try:
        import os
        import tempfile

        from pyvis.network import Network

        net = Network(
            height="500px",
            width="100%",
            bgcolor="#0D1117",
            font_color="#ffffff",
            directed=True,
        )

        net.set_options("""
        {
            "physics": {
                "stabilization": {"iterations": 50},
                "barnesHut": {"gravitationalConstant": -5000}
            },
            "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
                "color": {"color": "#4b5563"}
            },
            "interaction": {"hover": true}
        }
        """)

        def risk_to_color(risk: float) -> str:
            if risk >= 0.85:
                return "#ef4444"
            if risk >= 0.70:
                return "#f97316"
            if risk >= 0.50:
                return "#eab308"
            return "#22c55e"

        for node in nodes:
            addr = node.get("id", "")
            risk = node.get("risk_score", 0.0)
            color = "#a855f7" if addr == highlight_address else risk_to_color(risk)
            size = 15 + int(risk * 25)
            net.add_node(
                addr,
                label=addr[:8] + "...",
                color=color,
                size=size,
                title=f"Address: {addr}<br>Risk: {risk:.3f}<br>TXs: {node.get('tx_count', 0)}",
            )

        for edge in edges:
            net.add_edge(
                edge["from"],
                edge["to"],
                width=max(1, min(5, edge.get("value_eth", 0.1) * 2)),
                title=f"{edge.get('value_eth', 0):.4f} ETH",
            )

        # Save to temp file and read HTML
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            net.save_graph(f.name)
            tmp_path = f.name

        with open(tmp_path) as f:
            html = f.read()

        os.unlink(tmp_path)
        return html

    except Exception as e:
        return f"<p style='color:red'>Graph error: {e}</p>"


def render_sankey(
    paths: list,
    address: str,
) -> None:
    """
    Render a Plotly Sankey diagram showing fund flow paths.
    paths: list of lists (laundering paths)
    """
    if not paths:
        st.info("No laundering paths detected for this address.")
        return

    # Build nodes and links from paths
    all_nodes = []
    links = {"source": [], "target": [], "value": []}

    for path in paths[:10]:  # limit to 10 paths
        for node in path:
            if node not in all_nodes:
                all_nodes.append(node)

        for i in range(len(path) - 1):
            src_idx = all_nodes.index(path[i])
            tgt_idx = all_nodes.index(path[i + 1])
            links["source"].append(src_idx)
            links["target"].append(tgt_idx)
            links["value"].append(1)

    if not all_nodes:
        return

    # Color nodes by position (source=red, intermediate=yellow, sink=green)
    node_colors = []
    for n in all_nodes:
        if n == address:
            node_colors.append("#ef4444")
        elif n == all_nodes[-1]:
            node_colors.append("#22c55e")
        else:
            node_colors.append("#f97316")

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                label=[n[:12] + "..." for n in all_nodes],
                color=node_colors,
            ),
            link=dict(
                source=links["source"],
                target=links["target"],
                value=links["value"],
                color="rgba(100,100,100,0.3)",
            ),
        )
    )

    fig.update_layout(
        title=f"Fund Flow from {address[:16]}...",
        paper_bgcolor="#161B22",
        font_color="#E6EDF3",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_shap_waterfall(waterfall_data: dict) -> None:
    """
    Render a SHAP waterfall chart using Plotly.
    Shows which features drive the risk score up or down.
    """
    if not waterfall_data or "features" not in waterfall_data:
        st.info("SHAP explanation not available for this transaction.")
        return

    features = waterfall_data["features"]
    shap_values = waterfall_data["shap_values"]
    base_value = waterfall_data.get("base_value", 0.5)

    colors = ["#ef4444" if v > 0 else "#22c55e" for v in shap_values]

    fig = go.Figure(
        go.Bar(
            x=shap_values,
            y=features,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in shap_values],
            textposition="outside",
        )
    )

    fig.add_vline(
        x=base_value,
        line_dash="dash",
        line_color="#6b7280",
        annotation_text=f"Base: {base_value:.3f}",
    )

    fig.update_layout(
        title="SHAP Feature Contributions to Risk Score",
        xaxis_title="SHAP Value (impact on risk score)",
        paper_bgcolor="#161B22",
        plot_bgcolor="#0D1117",
        font_color="#E6EDF3",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
