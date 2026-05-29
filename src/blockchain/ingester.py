"""
BlockIngester — connects to Ethereum via WebSocket and streams transactions.

Architecture:
  WebSocket subscription → block received → extract transactions →
  normalize to Transaction model → yield to async pipeline

Fallback:
  If WebSocket disconnects, falls back to HTTP polling every 12 seconds
  (Ethereum block time). Reconnects automatically.
"""
import random
import asyncio

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

from config.logging_config import get_logger
from src.blockchain.models import Transaction
from src.monitoring.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer()


class BlockIngester:
    """
    Streams transactions from an Ethereum-compatible chain.
    Supports both WebSocket (preferred) and HTTP polling (fallback).
    """

    def __init__(
        self,
        http_url: str,
        ws_url: str,
        chain_id: int,
        chain_name: str,
        sample_rate: float = 1.0,
    ):
        self.http_url = http_url
        self.ws_url = ws_url
        self.chain_id = chain_id
        self.chain_name = chain_name
        self.sample_rate = sample_rate
        self._w3_http = self._connect_http()
        self._running = False

    def _connect_http(self) -> Web3:
        """HTTP connection — used for polling fallback and one-off queries."""
        w3 = Web3(Web3.HTTPProvider(self.http_url))
        # Inject PoA middleware for chains like Polygon that use PoA consensus
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        return w3

    def _normalize_transaction(
        self,
        tx: dict,
        block_timestamp: int,
    ) -> Transaction:
        """
        Convert raw web3 transaction dict to our normalized Transaction model.
        This is the boundary between web3's raw types and our domain model.
        """
        to_addr = tx.get("to")

        # web3 v6+ returns input as bytes — convert to hex string
        raw_input = tx.get("input", b"")
        if isinstance(raw_input, bytes):
            input_data = "0x" + raw_input.hex()
        elif isinstance(raw_input, str):
            input_data = raw_input if raw_input.startswith("0x") else "0x" + raw_input
        else:
            input_data = "0x"

        # Normalize to_addr — can be None for contract creation
        to_addr_str = to_addr if isinstance(to_addr, str) else None

        return Transaction(
            tx_hash=tx["hash"].hex(),
            block_number=tx["blockNumber"],
            block_timestamp=block_timestamp,
            from_addr=tx["from"],
            to_addr=to_addr_str,
            value_wei=tx["value"],
            gas=tx["gas"],
            gas_price=tx.get("gasPrice", 0),
            input_data=input_data,
            is_contract_creation=to_addr is None,
            is_contract_call=to_addr is not None and input_data != "0x",
            chain_id=self.chain_id,
            chain_name=self.chain_name,
        )

    async def _stream_via_polling(
        self,
        queue: asyncio.Queue,
        poll_interval: float = 4.0,
    ) -> None:
        """
        HTTP polling fallback.
        Checks for new blocks every poll_interval seconds.
        Used when WebSocket is unavailable.
        """
        logger.warning(
            "using_http_polling_fallback",
            chain=self.chain_name,
            interval=poll_interval,
        )

        last_block = await asyncio.to_thread(
            lambda: self._w3_http.eth.block_number
        )
        while self._running:
            try:
                current_block = await asyncio.to_thread(
                    lambda: self._w3_http.eth.block_number
                )
                if current_block > last_block:
                    for block_num in range(last_block + 1, current_block + 1):
                        with tracer.start_as_current_span("poll_block") as span:
                            span.set_attribute("block.number", block_num)
                            span.set_attribute("chain.name", self.chain_name)
                            # Offload the blocking RPC to a thread so this
                            # ingester doesn't starve other chains / workers
                            # sharing the event loop.
                            block = await asyncio.to_thread(
                                self._w3_http.eth.get_block,
                                block_num,
                                full_transactions=True,
                            )
                            timestamp = block["timestamp"]
                            tx_count = len(block["transactions"])

                            logger.info(
                                "block_polled",
                                block=block_num,
                                tx_count=tx_count,
                                chain=self.chain_name,
                            )

                            
                            for raw_tx in block["transactions"]:
                                # Volume control: high-throughput chains
                                # (Polygon) sample a fraction so they don't
                                # drown out low-volume chains in the queue.
                                if self.sample_rate < 1.0 and \
                                        random.random() > self.sample_rate:
                                    continue
                                tx = self._normalize_transaction(raw_tx, timestamp)
                                await queue.put(tx)

                    last_block = current_block

                await asyncio.sleep(poll_interval)

            except Exception as e:
                import traceback

                logger.error(
                    "polling_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    traceback=traceback.format_exc(),
                    chain=self.chain_name,
                )
                await asyncio.sleep(5)

    async def stream(self, queue: asyncio.Queue) -> None:
        """
        Main entry point. Streams transactions into the provided queue.
        Tries WebSocket first, falls back to HTTP polling on failure.

        The queue is bounded (maxsize set by caller) — if downstream
        processing is slow, this naturally applies backpressure.
        """
        self._running = True
        logger.info(
            "ingester_starting",
            chain=self.chain_name,
            chain_id=self.chain_id,
        )

        try:
            await self._stream_via_polling(queue)
        except Exception as e:
            logger.error(
                "ingester_fatal_error",
                error=str(e),
                chain=self.chain_name,
            )
            raise
        finally:
            self._running = False
            logger.info("ingester_stopped", chain=self.chain_name)

    def stop(self) -> None:
        """Signal the ingester to stop after current block."""
        self._running = False
        logger.info("ingester_stop_requested", chain=self.chain_name)

    def get_latest_block(self) -> int:
        """One-off query — current block number."""
        return self._w3_http.eth.block_number

    def test_connection(self) -> bool:
        """Verify RPC connection is working. Used in health checks."""
        try:
            block = self._w3_http.eth.block_number
            logger.info(
                "connection_verified",
                chain=self.chain_name,
                latest_block=block,
            )
            return True
        except Exception as e:
            logger.error(
                "connection_failed",
                chain=self.chain_name,
                error=str(e),
            )
            return False
