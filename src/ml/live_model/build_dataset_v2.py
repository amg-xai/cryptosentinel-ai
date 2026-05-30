"""
Rigorous labeled dataset in the 16 live-feature space (v2).

Fixes the source-artifact leakage in v1:
  - Both classes are built through the IDENTICAL pipeline:
      address -> getAssetTransfers (hashes) -> eth_getTransactionByHash
      (REAL gas/value/input) -> replay through FeatureEngineer.
  - Negatives are ACTIVITY-MATCHED: active addresses (>= K transfers),
    not one-off block senders, so the model can't cheat on activity level.
  - No placeholder constants anywhere.

Positives: OFAC + MEW darklist addresses.
Negatives: active recent-block senders not on any bad list.

Output: data/processed/live_X_v2.npy, live_y_v2.npy
"""
import time
import random
import numpy as np
import requests
from datetime import datetime
from pathlib import Path

from config.logging_config import get_logger
from config.settings import settings
from src.ml.tabular.feature_engineer import FeatureEngineer
from src.ml.live_model.label_sources import get_known_bad_addresses

logger = get_logger(__name__)
OUT = Path("data/processed")
URL = settings.eth_mainnet_http_url


def _rpc(method, params, retries=3):
    for attempt in range(retries):
        try:
            r = requests.post(URL, json={"id": 1, "jsonrpc": "2.0",
                                         "method": method, "params": params},
                              timeout=30)
            if r.status_code == 429:  # rate limited
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json().get("result")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.0 * (attempt + 1))
    return None


def _transfer_hashes(address, max_count=20):
    try:
        res = _rpc("alchemy_getAssetTransfers", [{
            "fromAddress": address,
            "category": ["external", "erc20", "erc721"],
            "maxCount": hex(max_count),
            "order": "desc",
            "withMetadata": True,
        }])
        out = []
        for t in (res or {}).get("transfers", []):
            if not t.get("hash"):
                continue
            meta = t.get("metadata") or {}
            out.append((t["hash"], meta.get("blockTimestamp")))
        return out
    except Exception as e:
        logger.debug("transfer_hashes_failed", addr=address[:10], error=str(e))
        return []


def _real_tx(tx_hash):
    try:
        return _rpc("eth_getTransactionByHash", [tx_hash])
    except Exception:
        return None


def _tx_to_payload(tx, ts):
    return {
        "tx_hash": tx.get("hash", ""),
        "from_addr": tx.get("from", ""),
        "to_addr": tx.get("to") or "",
        "value_eth": int(tx.get("value", "0x0"), 16) / 1e18,
        "gas": int(tx.get("gas", "0x5208"), 16),
        "gas_price": int(tx.get("gasPrice", "0x0"), 16),
        "is_contract_call": tx.get("input", "0x") not in ("0x", ""),
        "is_contract_creation": tx.get("to") is None,
        "input_data": tx.get("input", "0x"),
        "block_timestamp": float(ts),
        "chain_name": "ethereum-mainnet",
    }


def _features_for_address(address, min_txs=3):
    """Replay an address's REAL transactions through FeatureEngineer."""
    hashes = _transfer_hashes(address, max_count=20)
    if len(hashes) < min_txs:
        return None
    fe = FeatureEngineer()
    # oldest -> newest
    hashes_sorted = sorted(hashes, key=lambda x: x[1] or "")
    last_vec = None
    used = 0
    for h, bts in hashes_sorted:
        tx = _real_tx(h)
        if not tx or not tx.get("from"):
            continue
        ts = time.time()
        if bts:
            try:
                ts = datetime.fromisoformat(bts.replace("Z", "+00:00")).timestamp()
            except Exception:
                pass
        feats = fe.extract(_tx_to_payload(tx, ts))
        last_vec = feats.to_numpy()
        used += 1
    return last_vec if used >= min_txs else None


def _candidate_negative_addresses(n_blocks=60):
    """Collect distinct sender addresses from recent blocks."""
    latest = int(_rpc("eth_blockNumber", []), 16)
    addrs = set()
    bn = latest
    while bn > latest - n_blocks and len(addrs) < 600:
        blk = _rpc("eth_getBlockByNumber", [hex(bn), True])
        for raw in (blk or {}).get("transactions", []):
            frm = (raw.get("from") or "").lower()
            if frm:
                addrs.add(frm)
        bn -= 1
    return list(addrs)


def build(n_pos=120, n_neg=120, min_txs=3):
    bad_set = set(get_known_bad_addresses())
    bad = list(bad_set)
    random.shuffle(bad)

    # ---- Positives ----
    print(f"Positives (target {n_pos}, activity-matched >= {min_txs} txs)...")
    X_pos = []
    for addr in bad:
        if len(X_pos) >= n_pos:
            break
        vec = _features_for_address(addr, min_txs=min_txs)
        if vec is not None:
            X_pos.append(vec)
            if len(X_pos) % 20 == 0:
                print(f"  positives: {len(X_pos)}/{n_pos}")

    # ---- Negatives: activity-matched, not on bad list ----
    print(f"Negatives (target {n_neg}, same pipeline)...")
    candidates = _candidate_negative_addresses()
    random.shuffle(candidates)
    X_neg = []
    for addr in candidates:
        if len(X_neg) >= n_neg:
            break
        if addr in bad_set:
            continue
        vec = _features_for_address(addr, min_txs=min_txs)
        if vec is not None:
            X_neg.append(vec)
            if len(X_neg) % 20 == 0:
                print(f"  negatives: {len(X_neg)}/{n_neg}")

    return np.array(X_pos, dtype=np.float32), np.array(X_neg, dtype=np.float32)


def main():
    X_pos, X_neg = build()
    print(f"\nPositives: {X_pos.shape}, Negatives: {X_neg.shape}")
    if len(X_pos) < 20 or len(X_neg) < 20:
        print("Insufficient data — aborting save.")
        return
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))]).astype(int)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "live_X_v2.npy", X)
    np.save(OUT / "live_y_v2.npy", y)
    print(f"Saved live_X_v2 {X.shape} ({int(y.sum())} pos / {int((1-y).sum())} neg)")


if __name__ == "__main__":
    main()
