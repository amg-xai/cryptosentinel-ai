"""Known-bad address screening (OFAC SDN + community darklist).

Loaded once at import from a bundled snapshot so screening requires no
network call per request.
"""
import json
from pathlib import Path

from config.logging_config import get_logger

logger = get_logger(__name__)

_PATH = Path("data/known_bad_addresses.json")
_ADDRESSES: set[str] = set()


def _load() -> None:
    global _ADDRESSES
    try:
        data = json.loads(_PATH.read_text())
        _ADDRESSES = {a.lower() for a in data.get("addresses", [])}
        logger.info("watchlist_loaded", count=len(_ADDRESSES))
    except Exception as e:
        logger.warning("watchlist_unavailable", error=str(e))
        _ADDRESSES = set()


def is_known_bad(address: str) -> bool:
    if not address:
        return False
    return address.lower() in _ADDRESSES


def watchlist_size() -> int:
    return len(_ADDRESSES)


_load()
