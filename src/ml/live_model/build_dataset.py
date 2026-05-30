"""
Build a labeled dataset in the 16 live-feature space.

Positives: known-bad addresses (OFAC + MEW darklist). For each, pull recent
outgoing transfers from mainnet and replay them through the SAME
FeatureEngineer the live pipeline uses, so features match production exactly.

Negatives: transactions from random recent mainnet blocks (weak-negative
assumption: random mainnet traffic is overwhelmingly legitimate).

Output: data/processed/live_X.npy, live_y.npy
"""
import time
import random
import numpy as np
import requests
from pathlib import Path

from config.logging_config import get_logger
from config.settings import settings
from src.ml.tabular.feature_engineer import FeatureEngineer
from src.ml.live_model.label_sources import get_known_bad_addresses

logger = get_logger(__name__)
OUT_DIR = Path("data/processed")


def _rpc(method: str, params: list) -> dict:
    payload = {"id": 1, "jsonrpc": "2.0", "method": method, "params": params}
    r = requests.post(settings.eth_mainnet_http_url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json().get("result", {})


def _transfers_from(address: str, max_count: int = 30) -> list:
    """Outgoing transfers for an address via Alchemy getAssetTransfers."""
    try:
        res = _rpc("alchemy_getAssetTransfers", [{
            "fromAddress": address,
            "category": ["external", "erc20"],
            "maxCount": hex(max_count),
            "withMetadata": True,
            "excludeZeroValue": False,
            "order": "desc",
        }])
        return res.get("transfers", [])
    except Exception as e:
        logger.debug("transfers_fetch_failed", addr=address[:10], error=str(e))
        return []


def _transfer_to_payload(t: dict, chain="ethereum-mainnet") -> dict:
    """Convert an Alchemy transfer record into a FeatureEngineer payload."""
    # Parse timestamp from metadata
    ts = time.time()
    meta = t.get("metadata", {})
    bts = meta.get("blockTimestamp")
    if bts:
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(bts.replace("Z", "+00:00")).timestamp()
        except Exception:
            pass
    value = t.get("value") or 0.0
    return {
        "tx_hash": t.get("hash", ""),
        "from_addr": t.get("from", ""),
        "to_addr": t.get("to") or "",
        "value_eth": float(value),
        "gas": 21000,
        "gas_price": 20_000_000_000,
        "is_contract_call": t.get("category") == "erc20",
        "is_contract_creation": False,
        "input_data": "0x",
        "block_timestamp": float(ts),
        "chain_name": chain,
    }


def _features_for_address(address: str, fe: FeatureEngineer) -> np.ndarray:
    """
    Replay an address's transfers through the FeatureEngineer and return
    the feature vector after its most recent transfer (its 'current' state).
    """
    transfers = _transfers_from(address, max_count=30)
    if not transfers:
        return None
    # Replay oldest -> newest so rolling history builds correctly
    transfers = sorted(
        transfers,
        key=lambda t: t.get("metadata", {}).get("blockTimestamp", ""),
    )
    last_features = None
    for t in transfers:
        payload = _transfer_to_payload(t)
        if not payload["from_addr"]:
            continue
        feats = fe.extract(payload)
        last_features = feats.to_numpy()
    return last_features


def build(n_pos: int = 200, n_neg: int = 400) -> tuple:
    """Build positive (bad) and negative (random) feature sets."""
    fe = FeatureEngineer()

    # ---- Positives: known-bad addresses ----
    bad = get_known_bad_addresses()
    random.shuffle(bad)
    X_pos = []
    print(f"Building positives (target {n_pos})...")
    for addr in bad:
        if len(X_pos) >= n_pos:
            break
        vec = _features_for_address(addr, fe)
        if vec is not None:
            X_pos.append(vec)
        if len(X_pos) % 25 == 0 and len(X_pos) > 0:
            print(f"  positives: {len(X_pos)}/{n_pos}")

    # ---- Negatives: random recent mainnet transactions ----
    print(f"Building negatives (target {n_neg})...")
    fe_neg = FeatureEngineer()
    X_neg = []
    latest_hex = requests.post(
        settings.eth_mainnet_http_url,
        json={"id": 1, "jsonrpc": "2.0", "method": "eth_blockNumber", "params": []},
        timeout=30,
    ).json()["result"]
    latest = int(latest_hex, 16)

    block_num = latest
    while len(X_neg) < n_neg and block_num > latest - 200:
        blk = _rpc("eth_getBlockByNumber", [hex(block_num), True])
        if blk and blk.get("transactions"):
            ts = int(blk["timestamp"], 16)
            for raw in blk["transactions"]:
                if len(X_neg) >= n_neg:
                    break
                frm = (raw.get("from") or "").lower()
                payload = {
                    "tx_hash": raw.get("hash", ""),
                    "from_addr": raw.get("from", ""),
                    "to_addr": raw.get("to") or "",
                    "value_eth": int(raw.get("value", "0x0"), 16) / 1e18,
                    "gas": int(raw.get("gas", "0x5208"), 16),
                    "gas_price": int(raw.get("gasPrice", "0x0"), 16),
                    "is_contract_call": bool(raw.get("input", "0x") != "0x"),
                    "is_contract_creation": raw.get("to") is None,
                    "input_data": raw.get("input", "0x")[:10],
                    "block_timestamp": float(ts),
                    "chain_name": "ethereum-mainnet",
                }
                if not payload["from_addr"]:
                    continue
                feats = fe_neg.extract(payload)
                X_neg.append(feats.to_numpy())
            if len(X_neg) % 50 < len(blk["transactions"]):
                print(f"  negatives: {len(X_neg)}/{n_neg}")
        block_num -= 1

    X_pos = np.array(X_pos, dtype=np.float32)
    X_neg = np.array(X_neg, dtype=np.float32)
    return X_pos, X_neg


def main():
    X_pos, X_neg = build(n_pos=200, n_neg=400)
    print(f"\nPositives: {X_pos.shape}, Negatives: {X_neg.shape}")
    if len(X_pos) == 0 or len(X_neg) == 0:
        print("Insufficient data — aborting save.")
        return
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))]).astype(int)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "live_X.npy", X)
    np.save(OUT_DIR / "live_y.npy", y)
    print(f"Saved live_X {X.shape}, live_y {y.shape} "
          f"({int(y.sum())} pos / {int((1-y).sum())} neg)")


if __name__ == "__main__":
    main()
