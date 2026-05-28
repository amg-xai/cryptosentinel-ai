"""
CryptoSentinel AI — SOC Dashboard
Dark theme, 4 pages, real-time data from FastAPI backend.
Run: python -m streamlit run src/dashboard/app.py
"""
import time
import streamlit as st

st.set_page_config(
    page_title="CryptoSentinel AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.dashboard.api_client import (
    get_health,
    get_alerts,
    get_critical_alerts,
    get_graph_stats,
    analyze_wallet,
    get_wallet_graph,
    scan_contract,
    acknowledge_alert,
)
from src.dashboard.components import (
    render_kpi_cards,
    render_alert_table,
    render_risk_distribution,
    render_tier_breakdown,
    render_vulnerability_chart,
    render_pyvis_graph,
    render_sankey,
    render_shap_waterfall,
    severity_badge,
)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🛡️ CryptoSentinel AI")
    st.markdown("*Quantum-resistant blockchain threat intelligence*")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 SOC Overview", "🕸️ Threat Graph", "📋 Contract Scanner", "📊 Model Monitor"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**System Status**")

    health = get_health()
    status = health.get("status", "unknown")
    color = "🟢" if status == "healthy" else "🔴"
    st.markdown(f"{color} API: {status.upper()}")

    models = health.get("models_loaded", {})
    for model, loaded in models.items():
        icon = "🟢" if loaded else "🔴"
        st.markdown(f"{icon} {model.replace('_', ' ').title()}")

    st.divider()
    uptime = health.get("uptime_seconds", 0)
    st.caption(f"Uptime: {uptime}s")


# =====================
# PAGE 1: SOC OVERVIEW
# =====================
if page == "🏠 SOC Overview":
    st.markdown("# 🛡️ Security Operations Center")
    st.markdown("*Real-time blockchain threat intelligence*")

    alerts_data = get_alerts(limit=50)
    graph_stats = get_graph_stats()

    render_kpi_cards(health, alerts_data, graph_stats)
    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### 🚨 Active Alert Feed")
        render_alert_table(alerts_data.get("alerts", []))

    with col_right:
        st.markdown("### 📊 Risk Distribution")
        render_risk_distribution(alerts_data.get("alerts", []))
        st.markdown("### 🎯 Alert Tiers")
        render_tier_breakdown(alerts_data.get("stats", {}))

    st.divider()
    if st.button("🔄 Refresh Data"):
        st.rerun()


# ========================
# PAGE 2: THREAT GRAPH
# ========================
elif page == "🕸️ Threat Graph":
    st.markdown("# 🕸️ Threat Graph Intelligence")
    st.markdown("*Laundering path detection, wallet clustering, fund flow analysis*")

    col1, col2 = st.columns([3, 1])
    with col1:
        address_input = st.text_input(
            "🔍 Search wallet address",
            placeholder="0x...",
        )
    with col2:
        max_depth = st.slider("Path depth", 1, 5, 3)

    graph_stats = get_graph_stats()
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Graph Nodes", graph_stats.get("num_nodes", 0))
    with col_b:
        st.metric("Graph Edges", graph_stats.get("num_edges", 0))

    if address_input:
        with st.spinner("Analyzing wallet..."):
            wallet_data = analyze_wallet(address_input)
            graph_data = get_wallet_graph(address_input)

        if "error" not in wallet_data:
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                score = wallet_data.get("composite_score", 0)
                st.metric("Risk Score", f"{score:.3f}")
            with col2:
                st.metric("Action", wallet_data.get("action", "N/A"))
            with col3:
                st.metric("Severity", wallet_data.get("severity", "N/A"))

            st.divider()

            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs([
                "🕸️ Network Graph",
                "🌊 Fund Flow (Sankey)",
                "📍 Path Analysis",
                "🧠 SHAP Explanation",
            ])

            with tab1:
                st.markdown("#### Interactive Wallet Network")
                st.caption("Node color: 🔴 Critical  🟠 High  🟡 Medium  🟢 Low  🟣 Selected")

                # Build nodes and edges from graph data
                wallet_stats = graph_data.get("wallet_stats", {})
                neighbors = wallet_stats.get("neighbors", [])

                nodes = [{"id": address_input,
                         "risk_score": wallet_data.get("composite_score", 0),
                         "tx_count": wallet_stats.get("tx_count", 0)}]
                edges = []

                for neighbor in neighbors[:10]:
                    nodes.append({
                        "id": neighbor,
                        "risk_score": 0.3,
                        "tx_count": 1,
                    })
                    edges.append({
                        "from": address_input,
                        "to": neighbor,
                        "value_eth": 1.0,
                    })

                if len(nodes) > 1:
                    html = render_pyvis_graph(
                        nodes, edges,
                        highlight_address=address_input,
                    )
                    st.components.v1.html(html, height=520, scrolling=False)
                else:
                    st.info("No neighbor data in graph yet. Run the scoring pipeline to populate the graph.")

            with tab2:
                st.markdown("#### Fund Flow Analysis")
                paths = graph_data.get("laundering_paths", [])
                render_sankey(paths, address_input)

            with tab3:
                st.markdown("#### Laundering Path Detection")
                paths = graph_data.get("laundering_paths", [])
                if paths:
                    st.success(f"Found {len(paths)} potential laundering paths")
                    for i, path in enumerate(paths[:5]):
                        st.code(f"Path {i+1}: " + " → ".join(
                            [p[:10] + "..." for p in path]
                        ))
                else:
                    st.info("No laundering paths detected")

                st.markdown("#### Fund Sources (Peel-back)")
                ancestors = graph_data.get("ancestors", [])
                if ancestors:
                    for anc in ancestors[:5]:
                        st.text(anc[:20] + "...")
                else:
                    st.info("No ancestor wallets found")

                st.markdown("#### Round Trips")
                round_trips = graph_data.get("round_trips", [])
                if round_trips:
                    for rt in round_trips[:5]:
                        st.warning(
                            f"Via {rt['intermediate'][:10]}... "
                            f"in {rt['round_trip_seconds']:.0f}s"
                        )
                else:
                    st.success("No round trips detected")

            with tab4:
                st.markdown("#### SHAP Feature Importance")
                st.caption("Shows which transaction features drove the risk score")

                # Fetch SHAP explanation
                import requests
                try:
                    shap_resp = requests.post(
                        "http://localhost:8000/explain/shap-waterfall",
                        json={
                            "tx_hash": "0xanalysis",
                            "from_addr": address_input,
                            "value_eth": wallet_data.get("value_at_risk_eth", 0),
                            "gas": 21000,
                            "gas_price": 1000000000,
                            "block_timestamp": 1000000.0,
                            "chain_name": "ethereum-sepolia",
                        },
                        timeout=30,
                    )
                    if shap_resp.status_code == 200:
                        waterfall_data = shap_resp.json().get("waterfall_data", {})
                        render_shap_waterfall(waterfall_data)
                    else:
                        st.info("SHAP explanation unavailable")
                except Exception as e:
                    st.info(f"SHAP explanation unavailable: {e}")

        else:
            st.error(f"Error: {wallet_data.get('error')}")
    else:
        st.info("Enter a wallet address above to begin investigation")


# ==========================
# PAGE 3: CONTRACT SCANNER
# ==========================
elif page == "📋 Contract Scanner":
    st.markdown("# 📋 Smart Contract Vulnerability Scanner")
    st.markdown("*Rule-based + ML bytecode analysis*")

    example_vuln = """pragma solidity ^0.8.0;
contract VulnerableBank {
    mapping(address => uint) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw() public {
        uint amount = balances[msg.sender];
        (bool success,) = msg.sender.call.value(amount)("");
        require(success);
        balances[msg.sender] = 0;
    }

    function kill() public {
        selfdestruct(payable(msg.sender));
    }
}"""

    source_code = st.text_area(
        "Paste Solidity source code",
        value=example_vuln,
        height=300,
    )

    if st.button("🔍 Scan Contract", type="primary"):
        if source_code.strip():
            with st.spinner("Scanning contract..."):
                result = scan_contract(source_code)

            if "error" not in result:
                risk = result.get("combined_risk_score", 0)
                col1, col2, col3 = st.columns(3)
                with col1:
                    color = "🔴" if risk > 0.7 else ("🟡" if risk > 0.3 else "🟢")
                    st.metric("Risk Score", f"{color} {risk:.2f}")
                with col2:
                    vuln_count = sum(result.get("vulnerability_count", {}).values())
                    st.metric("Vulnerabilities", vuln_count)
                with col3:
                    st.metric(
                        "Critical",
                        "YES 🚨" if result.get("has_critical") else "None ✅",
                    )

                st.divider()
                render_vulnerability_chart(result)

                st.markdown("### 🐛 Vulnerabilities Found")
                vulns = result.get("vulnerabilities", [])
                if vulns:
                    for v in vulns:
                        severity = v.get("severity", "LOW")
                        with st.expander(
                            f"{severity_badge(severity)} {v.get('vuln_type')} "
                            f"— Line {v.get('line', '?')} "
                            f"(confidence: {v.get('confidence', 0):.0%})"
                        ):
                            st.markdown(f"**Description:** {v.get('description')}")
                            st.markdown(f"**Recommendation:** {v.get('recommendation')}")
                            if v.get("code_snippet"):
                                st.code(v["code_snippet"], language="solidity")
                else:
                    st.success("✅ No vulnerabilities detected")


# ==========================
# PAGE 4: MODEL MONITOR
# ==========================
elif page == "📊 Model Monitor":
    st.markdown("# 📊 ML Model Monitoring")
    import plotly.express as px
    st.markdown("*Model performance, drift detection, and system health*")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🤖 Model Status")
        for model, loaded in models.items():
            status_icon = "🟢 Loaded" if loaded else "🔴 Not loaded"
            st.markdown(f"**{model.replace('_', ' ').title()}:** {status_icon}")

        st.divider()
        st.markdown("### 📈 Elliptic Benchmark Results")
        import pandas as pd
        df = pd.DataFrame({
            "Model": ["Isolation Forest", "VAE Autoencoder", "GNN (GraphSAGE+GAT)"],
            "F1 Score": [0.001, 0.004, 0.676],
            "PR-AUC": [0.036, 0.038, 0.650],
            "ROC-AUC": [0.168, 0.198, 0.899],
        })
        st.dataframe(df, use_container_width=True)

    with col2:
        st.markdown("### ⚡ PQC Benchmark")
        df_pqc = pd.DataFrame({
            "Algorithm": ["ECDSA-secp256k1", "ML-DSA-65 (Dilithium3)", "ML-KEM-768 (Kyber)"],
            "Sign/Encap (ms)": [0.697, 0.250, 0.021],
            "Size (bytes)": [71, 3309, 1088],
            "Quantum Safe": ["❌", "✅", "✅"],
        })
        st.dataframe(df_pqc, use_container_width=True)

        st.divider()
        st.markdown("### 🌐 System Info")
        st.json({
            "service": "cryptosentinel-api",
            "version": "0.1.0",
            "uptime_seconds": health.get("uptime_seconds", 0),
            "environment": health.get("environment", "development"),
        })

    st.divider()
    st.markdown("### 📉 GNN Training History")
    import json
    from pathlib import Path
    history_path = Path("data/models/gnn_history.json")
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        df_hist = pd.DataFrame(history)
        fig = px.line(
            df_hist, x="epoch", y="loss",
            title="GNN Training Loss",
            color_discrete_sequence=["#00D4AA"],
        )
        fig.update_layout(
            paper_bgcolor="#161B22",
            plot_bgcolor="#0D1117",
            font_color="#E6EDF3",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Training history not available")

    st.divider()
    st.markdown("### ⚡ Feature Engineering Benchmarks")
    df_bench = pd.DataFrame({
        "Implementation": ["Python loop", "NumPy vectorized", "Numba JIT (parallel)"],
        "N=100 (ms)": [2.69, 0.06, 0.03],
        "N=1000 (ms)": [100.69, 5.71, 7.83],
        "N=5000 (ms)": ["~2500", "92.17", "9.76"],
        "Speedup vs Python": ["1x", "43x", "83x+"],
    })
    st.dataframe(df_bench, use_container_width=True)
