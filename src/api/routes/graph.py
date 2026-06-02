"""Threat graph query routes."""

from fastapi import APIRouter, Depends

from config.logging_config import get_logger
from src.api.dependencies import get_threat_graph, get_cross_chain
from src.api.schemas import GraphQueryResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/graph", tags=["Graph"])


@router.get("/{address}", response_model=GraphQueryResponse)
async def get_wallet_graph(
    address: str,
    max_depth: int = 3,
    threat_graph=Depends(get_threat_graph),
):
    """Get threat graph intelligence for a wallet address.

    Graph analysis (laundering paths, peel-back, round-trips) degrades
    gracefully: if the graph layer fails, we still return wallet stats with
    empty analysis and degraded=True, rather than a 500. A single graph
    hiccup must not take down the endpoint.
    """
    stats = threat_graph.get_wallet_stats(address)
    paths: list = []
    ancestors: list = []
    round_trips: list = []
    degraded = False
    try:
        paths = threat_graph.find_laundering_paths(address, max_depth=max_depth)
        ancestors = list(threat_graph.peel_back(address, max_depth=max_depth))
        round_trips = threat_graph.detect_round_trips(address)
    except Exception as e:
        degraded = True
        logger.warning("graph_analysis_degraded", address=address[:12], error=str(e))

    return GraphQueryResponse(
        address=address,
        wallet_stats=stats,
        laundering_paths=paths,
        ancestors=ancestors,
        round_trips=round_trips,
        cluster_id=stats.get("cluster_id") if stats else None,
        degraded=degraded,
    )


@router.get("/stats/summary")
async def get_graph_stats(threat_graph=Depends(get_threat_graph)):
    """Get overall threat graph statistics."""
    return {
        "num_nodes": threat_graph.num_nodes,
        "num_edges": threat_graph.num_edges,
    }

@router.get("/cross-chain/stats")
async def get_cross_chain_stats(analyzer=Depends(get_cross_chain)):
    """
    Cross-chain correlation stats: tracked addresses, multi-chain actors,
    bridge and bridged-asset interactions.

    NOTE: the API and pipeline are separate processes, so this reflects
    observations made within the API process plus the static registry.
    Production would back the analyzer with shared state (Redis).
    """
    return analyzer.stats()


@router.get("/cross-chain/{address}")
async def get_address_cross_chain(address: str, analyzer=Depends(get_cross_chain)):
    """Cross-chain profile for one address."""
    return {
        "address": address,
        "chains": analyzer.chains_for(address),
        "is_cross_chain_actor": analyzer.is_cross_chain_actor(address),
        "bridge_interactions": analyzer.bridge_interaction_count(address),
        "risk_boost": analyzer.cross_chain_risk_boost(address),
    }