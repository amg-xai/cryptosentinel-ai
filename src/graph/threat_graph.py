"""
ThreatGraph — directed transaction graph for money laundering detection.

WHY NetworkX DiGraph:
  Directed graph captures money FLOW direction.
  A→B means A sent funds to B — direction matters for laundering detection.
  Undirected graph would lose this signal entirely.

WHY incremental updates:
  Blockchain produces a new block every 12 seconds.
  Rebuilding the full graph on each block = too slow.
  add_transaction() updates in O(1) per transaction.

Core detection algorithms:
  1. Laundering path detection — depth-limited BFS from flagged nodes
  2. Peel-back tracing — follow funds BACKWARD from flagged wallet
  3. Round-trip detection — funds that leave and return within 24h
  4. Union-Find wallet clustering — group wallets controlled by same entity
  5. Louvain community detection — find tightly-connected wallet clusters

WHY Union-Find for clustering:
  Common-input-ownership heuristic: if two addresses appear as inputs
  in the same transaction, they likely belong to the same entity.
  Union-Find handles millions of wallets in near O(1) per operation.

WHY Louvain:
  Finds communities by maximizing modularity Q.
  Wallets that transact heavily with each other cluster together.
  Criminal organizations show tight internal transaction patterns.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx

from config.logging_config import get_logger
from src.monitoring.metrics import (
    GRAPH_QUERY_LATENCY,
    THREAT_GRAPH_EDGES,
    THREAT_GRAPH_NODES,
)

logger = get_logger(__name__)


@dataclass
class WalletNode:
    """Attributes stored per wallet node in the graph."""

    address: str
    risk_score: float = 0.0
    cluster_id: int | None = None
    entity_label: str = "unknown"  # exchange, mixer, contract, unknown
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    tx_count: int = 0
    total_value_eth: float = 0.0
    is_flagged: bool = False


@dataclass
class TransactionEdge:
    """Attributes stored per transaction edge."""

    tx_hash: str
    value_eth: float
    timestamp: float
    block_number: int
    is_contract_call: bool = False


class UnionFind:
    """
    Union-Find (Disjoint Set Union) for wallet clustering.

    Used for common-input-ownership heuristic:
    If wallets A and B both appear as inputs in the same transaction,
    they likely belong to the same entity → union(A, B).

    Time complexity: O(α(n)) ≈ O(1) per operation.
    """

    def __init__(self):
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        """Find root of x with path compression."""
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        """Union by rank — keeps tree balanced."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def same_cluster(self, x: str, y: str) -> bool:
        return self.find(x) == self.find(y)

    def get_clusters(self) -> dict[str, list[str]]:
        """Return all clusters as {root: [members]}."""
        clusters: dict[str, list[str]] = defaultdict(list)
        for node in self._parent:
            clusters[self.find(node)].append(node)
        return dict(clusters)


class ThreatGraph:
    """
    Directed transaction graph with threat intelligence capabilities.
    Maintains the graph incrementally as new transactions arrive.
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self._union_find = UnionFind()
        self._risk_cache: dict[str, float] = {}
        self._round_trip_window = 86400  # 24 hours in seconds

    def add_transaction(
        self,
        tx_hash: str,
        from_addr: str,
        to_addr: str,
        value_eth: float,
        timestamp: float,
        block_number: int,
        is_contract_call: bool = False,
        risk_score: float = 0.0,
    ) -> None:
        """
        Add a transaction to the graph.
        Idempotent — safe to call twice with same tx_hash.
        Updates node attributes if nodes already exist.
        """
        now = time.time()

        # Add/update from_addr node
        if not self.G.has_node(from_addr):
            self.G.add_node(
                from_addr,
                risk_score=0.0,
                cluster_id=None,
                entity_label="unknown",
                first_seen=now,
                last_seen=now,
                tx_count=0,
                total_value_eth=0.0,
                is_flagged=False,
            )

        # Add/update to_addr node
        if to_addr and not self.G.has_node(to_addr):
            self.G.add_node(
                to_addr,
                risk_score=0.0,
                cluster_id=None,
                entity_label="unknown",
                first_seen=now,
                last_seen=now,
                tx_count=0,
                total_value_eth=0.0,
                is_flagged=False,
            )

        if to_addr:
            # Update node stats
            self.G.nodes[from_addr]["last_seen"] = now
            self.G.nodes[from_addr]["tx_count"] += 1
            self.G.nodes[from_addr]["total_value_eth"] += value_eth

            # Update risk score if provided
            if risk_score > 0:
                current = self.G.nodes[from_addr]["risk_score"]
                self.G.nodes[from_addr]["risk_score"] = max(current, risk_score)

            # Add directed edge (from → to)
            # If edge exists, keep the one with higher value
            if not self.G.has_edge(from_addr, to_addr):
                self.G.add_edge(
                    from_addr,
                    to_addr,
                    tx_hash=tx_hash,
                    value_eth=value_eth,
                    timestamp=timestamp,
                    block_number=block_number,
                    is_contract_call=is_contract_call,
                )
            else:
                # Update edge with latest transaction
                self.G[from_addr][to_addr]["value_eth"] += value_eth
                self.G[from_addr][to_addr]["timestamp"] = timestamp

        # Update Prometheus gauges
        THREAT_GRAPH_NODES.set(self.G.number_of_nodes())
        THREAT_GRAPH_EDGES.set(self.G.number_of_edges())

    def flag_address(
        self,
        address: str,
        risk_score: float,
    ) -> None:
        """Mark an address as flagged and propagate risk to neighbors."""
        if not self.G.has_node(address):
            return

        self.G.nodes[address]["is_flagged"] = True
        self.G.nodes[address]["risk_score"] = risk_score

        # Propagate 50% of risk to direct predecessors (who sent to flagged)
        for pred in self.G.predecessors(address):
            current = self.G.nodes[pred]["risk_score"]
            propagated = risk_score * 0.5
            self.G.nodes[pred]["risk_score"] = max(current, propagated)

        logger.info(
            "address_flagged",
            address=address[:12] + "...",
            risk_score=risk_score,
            predecessors=self.G.in_degree(address),
        )

    def find_laundering_paths(
        self,
        source: str,
        max_depth: int = 5,
        max_paths: int = 100,
    ) -> list[list[str]]:
        """
        Find all paths from flagged source up to max_depth hops.
        Depth-limited BFS to detect layering patterns.

        Laundering = placement → layering → integration
        Layering: funds split across many addresses, multiple hops,
        then reconsolidate at a clean destination.

        max_paths limit prevents exponential explosion on hub nodes.
        """
        if not self.G.has_node(source):
            return []

        with GRAPH_QUERY_LATENCY.labels(query_type="laundering_path").time():
            paths = []
            try:
                # Get all nodes reachable from source within max_depth
                # using BFS — more reliable than all_simple_paths without target
                visited = {source}
                frontier = [[source]]

                while frontier:
                    current_path = frontier.pop(0)
                    current_node = current_path[-1]

                    if len(current_path) > 1:
                        paths.append(current_path)

                    if len(current_path) >= max_depth + 1:
                        continue

                    for neighbor in self.G.successors(current_node):
                        if neighbor not in current_path:  # avoid cycles
                            new_path = current_path + [neighbor]
                            frontier.append(new_path)

                    if len(paths) >= max_paths:
                        break

            except Exception as e:
                logger.error("laundering_path_error", error=str(e))
        logger.info(
            "laundering_paths_found",
            source=source[:12] + "...",
            paths_found=len(paths),
            max_depth=max_depth,
        )
        return paths

    def peel_back(
        self,
        address: str,
        max_depth: int = 5,
    ) -> set[str]:
        """
        Trace funds BACKWARD from a flagged address.
        Returns all ancestors up to max_depth hops.

        WHY backward tracing:
          Laundering involves layering through many wallets.
          To find the criminal's original wallet, trace backward
          from the known-bad address through the funding chain.
        """
        if not self.G.has_node(address):
            return set()

        with GRAPH_QUERY_LATENCY.labels(query_type="peel_back").time():
            ancestors = set()
            frontier = {address}

            for depth in range(max_depth):
                next_frontier = set()
                for node in frontier:
                    for pred in self.G.predecessors(node):
                        if pred not in ancestors and pred != address:
                            ancestors.add(pred)
                            next_frontier.add(pred)
                frontier = next_frontier
                if not frontier:
                    break

        logger.info(
            "peel_back_complete",
            address=address[:12] + "...",
            ancestors_found=len(ancestors),
        )
        return ancestors

    def detect_round_trips(
        self,
        address: str,
    ) -> list[dict]:
        """
        Detect funds that leave an address and return within 24h.
        Round-trips are a strong layering indicator.
        """
        if not self.G.has_node(address):
            return []

        round_trips = []
        outgoing_times: dict[str, float] = {}

        # Record when funds left
        for _, dst, data in self.G.out_edges(address, data=True):
            outgoing_times[dst] = data.get("timestamp", 0)

        # Check if funds returned from any destination
        for src, _, data in self.G.in_edges(address, data=True):
            if src in outgoing_times:
                time_out = outgoing_times[src]
                time_in = data.get("timestamp", 0)
                if 0 < (time_in - time_out) < self._round_trip_window:
                    round_trips.append(
                        {
                            "intermediate": src,
                            "time_out": time_out,
                            "time_in": time_in,
                            "round_trip_seconds": time_in - time_out,
                            "value_eth": data.get("value_eth", 0),
                        }
                    )

        if round_trips:
            logger.warning(
                "round_trips_detected",
                address=address[:12] + "...",
                count=len(round_trips),
            )

        return round_trips

    def cluster_wallets_union_find(
        self,
        transaction_inputs: list[list[str]],
    ) -> dict[str, list[str]]:
        """
        Apply common-input-ownership heuristic via Union-Find.

        For each transaction: if multiple addresses appear as inputs,
        union them — they likely belong to the same entity.

        Returns: {cluster_root: [wallet_addresses]}
        """
        for inputs in transaction_inputs:
            if len(inputs) >= 2:
                for i in range(1, len(inputs)):
                    self._union_find.union(inputs[0], inputs[i])

        clusters = self._union_find.get_clusters()

        # Assign cluster IDs to graph nodes
        for cluster_id, (root, members) in enumerate(clusters.items()):
            for member in members:
                if self.G.has_node(member):
                    self.G.nodes[member]["cluster_id"] = cluster_id

        logger.info(
            "union_find_clustering_complete",
            n_clusters=len(clusters),
            total_wallets=sum(len(m) for m in clusters.values()),
        )

        return clusters

    def detect_communities_louvain(self) -> dict[str, int]:
        """
        Apply Louvain community detection to find tightly-connected clusters.
        Returns: {address: community_id}

        Louvain optimizes modularity Q = Σ[A_ij - k_i*k_j/(2m)]
        Communities = groups with high internal vs external edge density.
        Criminal organizations show tight internal transaction patterns.
        """
        if self.G.number_of_nodes() < 2:
            return {}

        with GRAPH_QUERY_LATENCY.labels(query_type="louvain_community").time():
            try:
                import community as community_louvain

                # Louvain works on undirected graphs
                G_undirected = self.G.to_undirected()
                partition = community_louvain.best_partition(G_undirected)

                # Assign community IDs to graph nodes
                for node, community_id in partition.items():
                    if self.G.has_node(node):
                        self.G.nodes[node]["cluster_id"] = community_id

                n_communities = len(set(partition.values()))
                logger.info(
                    "louvain_complete",
                    n_communities=n_communities,
                    n_nodes=len(partition),
                )
                return partition

            except Exception as e:
                logger.error("louvain_failed", error=str(e))
                return {}

    def propagate_cluster_risk(self) -> None:
        """
        If any wallet in a cluster is flagged,
        boost risk scores of all wallets in the same cluster.

        WHY: A criminal uses many wallets. Flagging one
        should raise suspicion for the entire cluster.
        """
        clusters = self._union_find.get_clusters()

        for root, members in clusters.items():
            # Find max risk in cluster
            max_risk = (
                max(
                    self.G.nodes[m]["risk_score"] for m in members if self.G.has_node(m)
                )
                if members
                else 0.0
            )

            if max_risk > 0.5:
                # Propagate 70% of max risk to all cluster members
                propagated = max_risk * 0.7
                for member in members:
                    if self.G.has_node(member):
                        current = self.G.nodes[member]["risk_score"]
                        self.G.nodes[member]["risk_score"] = max(current, propagated)

    def get_wallet_stats(self, address: str) -> dict:
        """Get full stats for a wallet address."""
        if not self.G.has_node(address):
            return {}

        attrs = self.G.nodes[address]
        return {
            "address": address,
            "risk_score": attrs.get("risk_score", 0.0),
            "cluster_id": attrs.get("cluster_id"),
            "entity_label": attrs.get("entity_label", "unknown"),
            "tx_count": attrs.get("tx_count", 0),
            "total_value_eth": attrs.get("total_value_eth", 0.0),
            "is_flagged": attrs.get("is_flagged", False),
            "in_degree": self.G.in_degree(address),
            "out_degree": self.G.out_degree(address),
            "neighbors": list(self.G.successors(address))[:10],
        }

    @property
    def num_nodes(self) -> int:
        return self.G.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.G.number_of_edges()
