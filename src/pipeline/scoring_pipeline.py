"""
Scoring pipeline — consumes transactions from Kafka and scores them.
"""

import asyncio
import time

import uvloop

from config.logging_config import get_logger
from config.settings import settings
from src.blockchain.ingester import BlockIngester
from src.blockchain.models import Transaction
from src.ml.inference_engine import engine
from src.ml.tabular.feature_engineer import FeatureEngineer
from src.graph.threat_graph import ThreatGraph
from src.monitoring.metrics import TRANSACTIONS_SCANNED
from src.db.session import init_db, is_available as db_available
from src.db.alert_repository import save_alert
from src.response.alert_manager import AlertManager
from src.response.risk_scorer import CompositeRiskScorer, ModelScores
from src.monitoring.drift_detector import DriftDetector
from src.graph.cross_chain import CrossChainAnalyzer
from src.streaming.producer import ThreatIntelProducer

logger = get_logger(__name__)

# Load models and initialize DB at module startup (after all imports)
engine.load()
init_db()

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

_feature_engineer = FeatureEngineer()
_risk_scorer = CompositeRiskScorer()
_drift_detector = DriftDetector(window_size=500, min_samples=100)
_cross_chain = CrossChainAnalyzer()
_alert_manager = AlertManager()
_threat_graph = ThreatGraph()

QUEUE_MAX_SIZE = 10_000
PIPELINE_METRICS_PORT = 8001


async def scoring_worker(
    queue: asyncio.Queue,
    producer: ThreatIntelProducer,
) -> None:
    processed = 0
    start_time = time.time()
    loop = asyncio.get_event_loop()

    logger.info("scoring_worker_started")

    while True:
        tx: Transaction = await queue.get()
        processed += 1

        try:
            # 1. Extract features
            tx_payload = {
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
            features = _feature_engineer.extract(tx_payload)
            features_array = features.to_numpy()

            # 2. Score tabular models in thread pool (non-blocking)
            tabular_scores = await loop.run_in_executor(
                None, engine.score_features, features_array
            )

            # 3. Score GNN in thread pool (CUDA + asyncio safe)
            gnn_score = await loop.run_in_executor(
                None, engine.score_with_gnn, features_array
            )

            # 4. Add to threat graph
            _threat_graph.add_transaction(
                tx_hash=tx.tx_hash,
                from_addr=tx.from_addr,
                to_addr=tx.to_addr or "",
                value_eth=tx.value_eth,
                timestamp=float(tx.block_timestamp),
                block_number=tx.block_number,
                is_contract_call=tx.is_contract_call,
            )

            # 5. Graph centrality
            graph_stats = _threat_graph.get_wallet_stats(tx.from_addr)
            graph_centrality = 0.0
            if graph_stats:
                total_deg = graph_stats.get("in_degree", 0) + graph_stats.get(
                    "out_degree", 0
                )
                graph_centrality = min(1.0, total_deg / 100)

            # 6. Velocity check
            velocity_flag = features.tx_count_1h > 20
            # 6b. Cross-chain correlation
            cc_flags = _cross_chain.observe(
                tx.from_addr, tx.to_addr, tx.chain_name
            )
            cc_boost = _cross_chain.cross_chain_risk_boost(tx.from_addr)

            # 7. Build model scores
            model_scores = ModelScores(
                gnn=gnn_score,
                autoencoder=tabular_scores.get("autoencoder", -1.0),
                isolation_forest=tabular_scores.get("isolation_forest", -1.0),
                graph_centrality=graph_centrality,
                velocity_flag=velocity_flag,
            )

            # 8. Composite risk assessment
            assessment = _risk_scorer.score(
                address=tx.from_addr,
                tx_hash=tx.tx_hash,
                model_scores=model_scores,
                value_eth=tx.value_eth,
                cross_chain_boost=cc_boost,
            )

            # 9. Prometheus
            # 9. Prometheus + drift detection
            TRANSACTIONS_SCANNED.labels(chain=tx.chain_name).inc()
            _drift_detector.observe(assessment.composite_score)
            # Recompute PSI every 50 transactions (cheap, windowed)
            if processed % 50 == 0:
                _drift_detector.update_metrics()

            # 10. Handle threats
            if assessment.is_threat:
                _alert_manager.add_or_update(assessment)
                severity = assessment.severity.lower()
                topic = "critical" if severity == "critical" else "high"
                producer.produce_alert(assessment.to_dict(), severity=topic)
                if db_available():
                    save_alert(assessment.to_dict())

                logger.warning(
                    "threat_detected",
                    address=tx.from_addr[:12] + "...",
                    score=round(assessment.composite_score, 4),
                    action=assessment.action.value,
                    gnn=round(gnn_score, 4) if gnn_score >= 0 else "N/A",
                    if_score=round(tabular_scores.get("isolation_forest", 0), 4),
                )

            # 11. Progress log every 100 txs
            if processed % 100 == 0:
                elapsed = time.time() - start_time
                alert_stats = _alert_manager.get_stats()
                logger.info(
                    "pipeline_progress",
                    processed=processed,
                    tps=round(processed / elapsed, 1),
                    queue_size=queue.qsize(),
                    active_alerts=alert_stats["total_active"],
                    graph_nodes=_threat_graph.num_nodes,
                    block=tx.block_number,
                    gnn_last=round(gnn_score, 4) if gnn_score >= 0 else "N/A",
                )

        except Exception as e:
            import traceback

            logger.error(
                "scoring_error",
                error=str(e),
                tb=traceback.format_exc(),
                tx_hash=tx.tx_hash,
            )

        finally:
            queue.task_done()


async def run_scoring_pipeline() -> None:
    logger.info(
        "scoring_pipeline_starting",
        models={
            "isolation_forest": engine.isolation_forest is not None,
            "autoencoder": engine.autoencoder is not None,
            "gnn": engine.gnn_trainer is not None,
        },
    )
    # Expose pipeline metrics so Prometheus can scrape this process.
    # The API and pipeline are separate processes with separate metric
    # registries; without this, pipeline-incremented metrics (transactions
    # scanned, PSI drift, cross-chain) never reach Prometheus.
    try:
        from prometheus_client import start_http_server
        start_http_server(PIPELINE_METRICS_PORT)
        logger.info("pipeline_metrics_server_started",
                    port=PIPELINE_METRICS_PORT)
    except Exception as e:
        logger.warning("pipeline_metrics_server_failed", error=str(e))

    producer = ThreatIntelProducer()
    queue: asyncio.Queue[Transaction] = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)

    eth_ingester = BlockIngester(
        http_url=settings.eth_http_url,
        ws_url=settings.eth_ws_url,
        chain_id=11155111,
        chain_name="ethereum-sepolia",
    )
    if not eth_ingester.test_connection():
        logger.error("pipeline_aborted", reason="connection_failed")
        return
    logger.info(
        "pipeline_connected",
        chain="ethereum-sepolia",
        latest_block=eth_ingester.get_latest_block(),
    )

    # Tasks that always run: Ethereum stream + 2 scoring workers
    tasks = [
        eth_ingester.stream(queue),
        scoring_worker(queue, producer),
        scoring_worker(queue, producer),
    ]

    # Optionally add Polygon mainnet if a real RPC URL is configured.
    # Both chains feed the SAME queue + workers, so the cross-chain
    # analyzer sees addresses from both and can fire its detection.
    polygon_url = settings.polygon_http_url
    if polygon_url and "localhost" not in polygon_url:
        polygon_ingester = BlockIngester(
            http_url=polygon_url,
            ws_url=settings.polygon_ws_url,
            chain_id=137,
            chain_name="polygon-mainnet",
            sample_rate=0.20,  # ~2s blocks, 100+ tx each — sample 20%
        )
        if polygon_ingester.test_connection():
            logger.info(
                "pipeline_connected",
                chain="polygon-mainnet",
                latest_block=polygon_ingester.get_latest_block(),
            )
            tasks.append(polygon_ingester.stream(queue))
        else:
            logger.warning("polygon_connection_failed_skipping")
    else:
        logger.info("polygon_not_configured_single_chain")

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(run_scoring_pipeline())
