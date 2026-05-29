"""
Cross-chain correlation and bridge-interaction detection.

THE THREAT:
  Cross-chain laundering is a top evasion technique. Funds enter a bridge
  on chain A and exit on chain B, breaking the on-chain trail for any
  monitor that only watches one chain. Mixers + bridges together are how
  most large exploit proceeds (e.g. bridge hacks) get laundered.

WHAT THIS DETECTS:
  1. Bridge interactions — transactions to/from known bridge contracts.
  2. Cross-chain actors — the SAME address (EVM addresses are identical
     across EVM chains) active on more than one monitored chain.
  3. Cross-chain laundering candidates — an address that both interacts
     with a bridge AND is active on multiple chains, especially when its
     risk score is already elevated.

HONESTY NOTE:
  The deployment monitors Ethereum Sepolia (testnet) + Polygon mainnet,
  which don't bridge to each other in reality. The bridge registry below
  uses REAL public Polygon PoS bridge addresses, so bridge detection on
  the live Polygon stream is genuine. In production you'd monitor
  Ethereum mainnet too, where these bridges actually connect.
"""
from collections import defaultdict
from typing import Optional

from config.logging_config import get_logger

logger = get_logger(__name__)


# Real, public bridge contract addresses (lowercased for matching).
# Polygon PoS bridge contracts live on Ethereum mainnet; the ERC20/Ether
# predicate + RootChainManager are the canonical deposit entrypoints.
KNOWN_BRIDGES = {
    # Polygon PoS bridge (Ethereum side)
    "0xa0c68c638235ee32657e8f720a23cec1bfc77c77": "Polygon PoS: RootChainManager",
    "0x40ec5b33f54e0e8a33a975908c5ba1c14e5bbbdf": "Polygon PoS: ERC20 Predicate",
    "0x8484ef722627bf18ca5ae6bcf031c23e6e922b30": "Polygon PoS: Ether Predicate",
    # Polygon Plasma bridge
    "0x401f6c983ea34274ec46f84d70b31c151321188b": "Polygon: Plasma DepositManager",
    # Common third-party bridges seen across EVM chains
    # Common third-party bridges seen across EVM chains
    "0x3ee18b2214aff97000d974cf647e7c347e8fa585": "Wormhole: Token Bridge",
    "0xa7d7079b0fead91f3e65f86e8915cb59c1a4c664": "Across: SpokePool",
}

# Bridged-asset token contracts on Polygon mainnet. These tokens only
# exist because value was bridged from Ethereum, so interacting with them
# is a genuine (if weaker) cross-chain signal. Kept SEPARATE from bridge
# entrypoints above so we don't overstate — touching bridged USDC is not
# the same as making a bridge deposit.
BRIDGED_ASSETS = {
    "0x2791bca1f2de4661ed88a30c99a7a9449aa84174": "USDC.e (PoS-bridged USDC)",
    "0x7ceb23fd6bc0add59e62ac25578270cff1b9f619": "WETH (PoS-bridged)",
    "0xc2132d05d31c914a87c6611c10748aeb04b58e8f": "USDT (PoS-bridged)",
}


def is_bridged_asset(address: Optional[str]) -> bool:
    if not address:
        return False
    return address.lower() in BRIDGED_ASSETS


def bridged_asset_name(address: Optional[str]) -> Optional[str]:
    if not address:
        return None
    return BRIDGED_ASSETS.get(address.lower())

def is_bridge(address: Optional[str]) -> bool:
    """True if the address is a known bridge contract."""
    if not address:
        return False
    return address.lower() in KNOWN_BRIDGES


def bridge_name(address: Optional[str]) -> Optional[str]:
    """Human-readable bridge name, or None."""
    if not address:
        return None
    return KNOWN_BRIDGES.get(address.lower())


class CrossChainAnalyzer:
    """
    Tracks per-address activity across chains and bridge interactions.
    Lightweight in-memory index updated as transactions flow.
    """

    def __init__(self):
        # address -> set of chain names it has appeared on
        self._chains_by_address = defaultdict(set)
        # address -> count of bridge interactions
        self._bridge_hits = defaultdict(int)
        # address -> set of bridge names touched
        self._bridges_touched = defaultdict(set)
        # address -> count of bridged-asset token interactions
        self._bridged_asset_hits = defaultdict(int)

    def observe(
        self,
        from_addr: str,
        to_addr: Optional[str],
        chain_name: str,
    ) -> dict:
        """
        Record one transaction's chain + bridge signals.
        Returns a dict of any cross-chain flags raised for this tx.
        """
        flags = {}

        if from_addr:
            self._chains_by_address[from_addr].add(chain_name)
        if to_addr:
            self._chains_by_address[to_addr].add(chain_name)

        # Bridge interaction: either side is a known bridge
        bname = bridge_name(to_addr) or bridge_name(from_addr)
        if bname:
            actor = from_addr if is_bridge(to_addr) else to_addr
            if actor:
                self._bridge_hits[actor] += 1
                self._bridges_touched[actor].add(bname)
            flags["bridge_interaction"] = bname
            flags["bridge_actor"] = actor
            # Bridged-asset interaction: weaker cross-chain signal
        aname = bridged_asset_name(to_addr)
        if aname and from_addr:
            self._bridged_asset_hits[from_addr] += 1
            flags["bridged_asset"] = aname

        # Cross-chain actor: from_addr active on 2+ chains
        if from_addr and len(self._chains_by_address[from_addr]) > 1:
            flags["cross_chain_actor"] = sorted(
                self._chains_by_address[from_addr]
            )

        return flags

    def chains_for(self, address: str) -> list:
        """Which chains has this address been seen on?"""
        return sorted(self._chains_by_address.get(address, set()))

    def is_cross_chain_actor(self, address: str) -> bool:
        return len(self._chains_by_address.get(address, set())) > 1

    def bridge_interaction_count(self, address: str) -> int:
        return self._bridge_hits.get(address, 0)

    def cross_chain_risk_boost(self, address: str) -> float:
        """
        Multiplicative risk boost for cross-chain laundering signals.
        - Active on multiple chains:        x1.15
        - Has interacted with a bridge:     x1.25
        - Both (classic cross-chain hop):   x1.4 (capped)
        Returns 1.0 (no boost) for single-chain, non-bridge addresses.
        """
        cross = self.is_cross_chain_actor(address)
        bridged = self.bridge_interaction_count(address) > 0
        asset = self._bridged_asset_hits.get(address, 0) > 0
        if cross and bridged:
            return 1.4   # classic cross-chain hop via bridge entrypoint
        if bridged:
            return 1.25
        if cross:
            return 1.15
        if asset:
            return 1.08  # touched a bridged asset — mild signal
        return 1.0

    def get_cross_chain_actors(self) -> list:
        """All addresses active on more than one chain."""
        return [
            addr for addr, chains in self._chains_by_address.items()
            if len(chains) > 1
        ]

    def stats(self) -> dict:
        """Summary stats for the dashboard / API."""
        cross_actors = self.get_cross_chain_actors()
        return {
            "total_addresses_tracked": len(self._chains_by_address),
            "cross_chain_actors": len(cross_actors),
            "addresses_touching_bridges": len(self._bridge_hits),
           "total_bridge_interactions": sum(self._bridge_hits.values()),
            "bridged_asset_interactions": sum(self._bridged_asset_hits.values()),
            "known_bridges": len(KNOWN_BRIDGES),
            "known_bridged_assets": len(BRIDGED_ASSETS),
        }