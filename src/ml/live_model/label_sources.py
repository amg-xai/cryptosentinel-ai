"""
Weak-label sources for live-feature model training.

Positive (known-bad) addresses come from PUBLIC, citable lists:
  - OFAC sanctioned digital-currency addresses (US Treasury, via 0xB10C mirror)
  - MyEtherWallet community darklist (phishing/scam addresses)

These are weak labels: the lists are incomplete and ETH-mainnet-centric,
but they are real, attributable, and exactly the kind of threat-intel feed
a production system would ingest. Negatives are sampled from random recent
mainnet traffic (overwhelmingly legitimate) — a standard weak-negative
assumption in fraud detection.
"""
import json
import requests

from config.logging_config import get_logger

logger = get_logger(__name__)

OFAC_ETH_URL = (
    "https://raw.githubusercontent.com/0xB10C/"
    "ofac-sanctioned-digital-currency-addresses/lists/"
    "sanctioned_addresses_ETH.txt"
)
MEW_DARKLIST_URL = (
    "https://raw.githubusercontent.com/MyEtherWallet/"
    "ethereum-lists/master/src/addresses/addresses-darklist.json"
)


def fetch_ofac_addresses(timeout: int = 30) -> set:
    """Sanctioned ETH addresses (lowercased)."""
    try:
        r = requests.get(OFAC_ETH_URL, timeout=timeout)
        r.raise_for_status()
        addrs = {
            line.strip().lower()
            for line in r.text.splitlines()
            if line.strip().startswith("0x")
        }
        logger.info("ofac_addresses_fetched", count=len(addrs))
        return addrs
    except Exception as e:
        logger.error("ofac_fetch_failed", error=str(e))
        return set()


def fetch_mew_darklist(timeout: int = 30) -> set:
    """MEW community darklist addresses (lowercased)."""
    try:
        r = requests.get(MEW_DARKLIST_URL, timeout=timeout)
        r.raise_for_status()
        data = json.loads(r.text)
        addrs = {
            entry["address"].strip().lower()
            for entry in data
            if entry.get("address", "").startswith("0x")
        }
        logger.info("mew_darklist_fetched", count=len(addrs))
        return addrs
    except Exception as e:
        logger.error("mew_fetch_failed", error=str(e))
        return set()


def get_known_bad_addresses() -> list:
    """Union of all weak-label bad-address sources."""
    bad = fetch_ofac_addresses() | fetch_mew_darklist()
    logger.info("known_bad_total", count=len(bad))
    return sorted(bad)
