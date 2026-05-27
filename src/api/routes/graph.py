"""Threat graph query routes."""
from fastapi import APIRouter, Depends
from src.api.schemas import GraphQueryResponse
from src.api.dependencies import get_threat_graph
from config.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/graph", tags=["Graph"])


@router.get("/{address}", response_model=GraphQueryResponse)
async def get_wallet_graph(
    address: str,
    max_depth: int = 3,
    threat_graph=Depends(get_threat_graph),
):
    """Get threat graph intelligence for a wallet address."""
    stats = threat_graph.get_wallet_stats(address)
    paths = threat_graph.find_laundering_paths(address, max_depth=max_depth)
    ancestors = list(threat_graph.peel_back(address, max_depth=max_depth))
    round_trips = threat_graph.detect_round_trips(address)

    return GraphQueryResponse(
        address=address,
        wallet_stats=stats,
        laundering_paths=paths,
        ancestors=ancestors,
        round_trips=round_trips,
        cluster_id=stats.get("cluster_id") if stats else None,
    )


@router.get("/stats/summary")
async def get_graph_stats(threat_graph=Depends(get_threat_graph)):
    """Get overall threat graph statistics."""
    return {
        "num_nodes": threat_graph.num_nodes,
        "num_edges": threat_graph.num_edges,
    }
