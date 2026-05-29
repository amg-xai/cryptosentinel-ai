"""
Collect a drift baseline from the LIVE scoring path.

WHY this replaces the Elliptic-score baseline:
  The pipeline scores transactions via padded live features ->
  ensemble models -> CompositeRiskScorer -> composite_score, and the
  drift detector observes that composite score. The old baseline used
  the GNN's scores on the Elliptic test set — a completely different
  distribution — so PSI was pinned high and meaningless.

  This collector runs the SAME components on real recent blocks and
  saves the resulting composite scores. PSI then compares live-vs-live,
  so it reads ~0 when traffic is stable and spikes only on genuine drift.

Usage:
  python -m src.monitoring.collect_live_baseline --target 600
"""
import argparse
import numpy as np

from config.logging_config import get_logger
from config.settings import settings
from src.blockchain.ingester import BlockIngester
from src.ml.inference_engine import engine
from src.ml.tabular.feature_engineer import FeatureEngineer
from src.response.risk_scorer import CompositeRiskScorer, ModelScores
from src.graph.threat_graph import ThreatGraph
from src.monitoring.drift_detector import save_baseline

logger = get_logger(__name__)


def _score_tx(tx, fe, scorer, graph) -> float:
    """Run one transaction through the live scoring path; return composite."""
    payload = {
        "tx_hash": tx.tx_hash,
        "from_addr": tx.from_addr,
        "to_addr": tx.to_addr,
        "value_eth": tx.value_eth,
        "gas": tx.gas,
        "gas_price": tx.gas_price,
        "is_contract_call": tx.is_contract_call,
        "is_contract_creation": tx.is_contract_creation,
        "input_data": tx.input_data,
        "block_timestamp": float(tx.block_timestamp),
        "chain_name": tx.chain_name,
    }
    features = fe.extract(payload)
    arr = features.to_numpy()
    tabular = engine.score_features(arr)
    gnn = engine.score_with_gnn(arr)

    graph.add_transaction(
        tx_hash=tx.tx_hash,
        from_addr=tx.from_addr,
        to_addr=tx.to_addr or "",
        value_eth=tx.value_eth,
        timestamp=float(tx.block_timestamp),
        block_number=tx.block_number,
        is_contract_call=tx.is_contract_call,
    )
    gstats = graph.get_wallet_stats(tx.from_addr)
    centrality = 0.0
    if gstats:
        deg = gstats.get("in_degree", 0) + gstats.get("out_degree", 0)
        centrality = min(1.0, deg / 100)

    ms = ModelScores(
        gnn=gnn,
        autoencoder=tabular.get("autoencoder", -1.0),
        isolation_forest=tabular.get("isolation_forest", -1.0),
        graph_centrality=centrality,
        velocity_flag=features.tx_count_1h > 20,
    )
    assessment = scorer.score(
        address=tx.from_addr,
        tx_hash=tx.tx_hash,
        model_scores=ms,
        value_eth=tx.value_eth,
    )
    return assessment.composite_score


def collect(target: int = 600) -> np.ndarray:
    """Fetch recent blocks and score until `target` composite scores gathered."""
    engine.load()
    fe = FeatureEngineer()
    scorer = CompositeRiskScorer()
    graph = ThreatGraph()

    ingester = BlockIngester(
        http_url=settings.eth_http_url,
        ws_url=settings.eth_ws_url,
        chain_id=11155111,
        chain_name="ethereum-sepolia",
    )

    scores = []
    latest = ingester.get_latest_block()
    block_num = latest
    while len(scores) < target and block_num > latest - 5000:
        block = ingester._w3_http.eth.get_block(block_num, full_transactions=True)
        ts = block["timestamp"]
        for raw in block["transactions"]:
            tx = ingester._normalize_transaction(raw, ts)
            try:
                scores.append(_score_tx(tx, fe, scorer, graph))
            except Exception as e:
                logger.debug("baseline_score_skip", error=str(e))
        if len(scores) % 100 < len(block["transactions"]):
            print(f"  collected {len(scores)}/{target}...")
        block_num -= 1

    return np.asarray(scores[:target])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=600)
    args = parser.parse_args()

    print(f"Collecting {args.target} live composite scores...")
    scores = collect(args.target)
    if len(scores) == 0:
        print("No scores collected — check RPC connectivity.")
        return
    save_baseline(scores)
    print(f"Live baseline saved: {len(scores)} scores, "
          f"mean={scores.mean():.4f}, std={scores.std():.4f}, "
          f"min={scores.min():.4f}, max={scores.max():.4f}")


if __name__ == "__main__":
    main()
