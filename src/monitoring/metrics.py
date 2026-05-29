"""
Prometheus metrics — domain-specific business metrics.
"""

from prometheus_client import Counter, Gauge, Histogram

TRANSACTIONS_SCANNED = Counter(
    "cryptosentinel_transactions_scanned_total",
    "Total transactions scanned by chain",
    ["chain"],
)

THREATS_DETECTED = Counter(
    "cryptosentinel_threats_detected_total",
    "Threats detected by severity and action tier",
    ["severity", "action_tier"],
)

RISK_SCORE_HISTOGRAM = Histogram(
    "cryptosentinel_risk_score",
    "Distribution of risk scores across all scored transactions",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
)

GNN_INFERENCE_LATENCY = Histogram(
    "cryptosentinel_gnn_inference_seconds",
    "GNN inference latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

KAFKA_CONSUMER_LAG = Gauge(
    "cryptosentinel_kafka_consumer_lag",
    "Kafka consumer lag by topic",
    ["topic"],
)

KAFKA_MESSAGES_PRODUCED = Counter(
    "cryptosentinel_kafka_messages_produced_total",
    "Total Kafka messages produced by topic",
    ["topic"],
)

KAFKA_MESSAGES_CONSUMED = Counter(
    "cryptosentinel_kafka_messages_consumed_total",
    "Total Kafka messages consumed by topic",
    ["topic"],
)

GRAPH_QUERY_LATENCY = Histogram(
    "cryptosentinel_graph_query_seconds",
    "Graph query latency by query type",
    ["query_type"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

THREAT_GRAPH_NODES = Gauge(
    "cryptosentinel_threat_graph_nodes_total",
    "Total nodes in the threat graph",
)

THREAT_GRAPH_EDGES = Gauge(
    "cryptosentinel_threat_graph_edges_total",
    "Total edges in the threat graph",
)

ACTIVE_WATCHLIST_ADDRESSES = Gauge(
    "cryptosentinel_watchlist_addresses_total",
    "Number of addresses currently on the watchlist",
)

PQC_SIGNING_LATENCY = Histogram(
    "cryptosentinel_pqc_signing_seconds",
    "Post-quantum signing latency in seconds",
    ["algorithm"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
)

CONTRACTS_SCANNED = Counter(
    "cryptosentinel_contracts_scanned_total",
    "Smart contracts scanned by result",
    ["result"],
)


# --- Model drift detection (Day 37) ---
MODEL_SCORE_DRIFT_PSI = Gauge(
    "cryptosentinel_model_score_drift_psi",
    "Population Stability Index between live risk-score window and baseline. "
    "PSI < 0.1 stable, 0.1-0.25 moderate drift, > 0.25 significant drift.",
)
MODEL_SCORE_DRIFT_SAMPLES = Gauge(
    "cryptosentinel_model_score_drift_samples",
    "Number of live scores in the current drift-detection window.",
)
