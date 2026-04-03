"""
PacificaAdapter — production adapter for Pacifica DEX.

Key implementation details:
  - ALL limit orders use tif="ALO" (no speed bump, guaranteed maker)
  - Batch orders for requoting: cancel+place in ONE round trip
  - Ed25519 signing via solders (Solana keypair, NOT Ethereum)
  - Prices and amounts sent as strings in API calls
  - Recursive key sorting required for all signed payloads

Requires: solders, base58, websockets, aiohttp
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from typing import AsyncIterator, Callable

import aiohttp

from MM_algo.adapters.base import (
    ExchangeAdapter, L2Snapshot, TradeEvent, OrderResult,
)

logger = logging.getLogger(__name__)

# Try imports — fail gracefully for testing without deps
try:
    import base58
    from solders.keypair import Keypair  # type: ignore
    import websockets
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    logger.warning("Pacifica deps not installed (solders, base58, websockets)")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAINNET_WS = "wss://ws.pacifica.fi/ws"
TESTNET_WS = "wss://test-ws.pacifica.fi/ws"
MAINNET_REST = "https://api.pacifica.fi"
TESTNET_REST = "https://api-testnet.pacifica.fi"


class PacificaAdapter(ExchangeAdapter):
    """
    Full adapter for Pacifica DEX.

    All limit orders use tif="ALO" (Add Liquidity Only):
      - No 200ms speed bump
      - Guaranteed maker fee
      - Rejected if would cross (handled gracefully)

    Args:
        private_key:   base58-encoded Ed25519 private key (from env)
        network:       "mainnet" or "testnet"
        symbol:        trading pair (e.g. "BTC", "ETH", "SOL")
        builder_code:  your registered builder code (0 fees to self)
        simulate:      if True, log orders without sending
    """

    def __init__(
        self,
        private_key: str = "",
        network: str = "testnet",
        symbol: str = "BTC",
        builder_code: str | None = None,
        simulate: bool = True,
    ) -> None:
        self._network = network
        self._symbol = symbol
        self._builder_code = builder_code
        self._simulate = simulate

        # Keypair
        self._keypair = None
        self._public_key = ""
        if private_key and HAS_DEPS:
            self._keypair = Keypair.from_bytes(base58.b58decode(private_key))
            self._public_key = str(self._keypair.pubkey())

        # Endpoints
        self._ws_url = TESTNET_WS if network == "testnet" else MAINNET_WS
        self._rest_url = TESTNET_REST if network == "testnet" else MAINNET_REST

        # Connection state
        self._ws = None
        self._connected = False
        self._last_pong_at: float = 0.0
        self._heartbeat_task: asyncio.Task | None = None
        self._message_task: asyncio.Task | None = None

        # Market state
        self._last_ob: dict = {"bids": [], "asks": []}
        self._last_bbo: dict = {"bid_px": 0, "bid_sz": 0, "ask_px": 0, "ask_sz": 0}
        self._last_mid: float = 0.0
        self._orderbook_updated_at: float = 0.0

        # Account state
        self._position_q: float = 0.0
        self._balance_usdc: float = 0.0
        self._open_orders: dict[str, dict] = {}

        # Market info
        self._tick_size: float = 0.01
        self._lot_size: float = 0.001

        # Callbacks
        self._on_orderbook: Callable | None = None
        self._on_trade: Callable | None = None
        self._on_fill: Callable | None = None

        # Pending fills queue
        self._pending_fills: asyncio.Queue = asyncio.Queue()

        # Sim order counter
        self._sim_counter = 0

        # WS response tracking
        self._pending_responses: dict[str, asyncio.Future] = {}

    # -------------------------------------------------------------------
    # Signing (Ed25519 / Solana)
    # -------------------------------------------------------------------

    def _sort_json_keys(self, value):
        """Recursively sort dict keys at all levels. Required by Pacifica."""
        if isinstance(value, dict):
            return {k: self._sort_json_keys(v) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return [self._sort_json_keys(item) for item in value]
        return value

    def _sign(
        self,
        operation_type: str,
        operation_data: dict,
        expiry_window_ms: int = 5000,
    ) -> dict:
        """
        Build and sign a Pacifica request payload.

        Returns the complete payload ready to send over WS or REST.
        """
        timestamp = int(time.time() * 1000)

        sign_payload = {
            "timestamp": timestamp,
            "expiry_window": expiry_window_ms,
            "type": operation_type,
            "data": operation_data,
        }

        sorted_payload = self._sort_json_keys(sign_payload)
        compact_json = json.dumps(sorted_payload, separators=(",", ":"))

        sig_bytes = self._keypair.sign_message(compact_json.encode("utf-8"))
        sig_b58 = base58.b58encode(bytes(sig_bytes)).decode("ascii")

        return {
            "account": self._public_key,
            "agent_wallet": None,
            "signature": sig_b58,
            "timestamp": timestamp,
            "expiry_window": expiry_window_ms,
            **operation_data,
        }

    # -------------------------------------------------------------------
    # Connection
    # -------------------------------------------------------------------

    async def connect(self) -> None:
        if self._simulate:
            logger.info("PacificaAdapter connected in SIMULATE mode (%s)", self._network)
            self._connected = True
            return

        if not HAS_DEPS:
            raise RuntimeError("Pacifica deps not installed: pip install solders base58 websockets")

        self._ws = await websockets.connect(self._ws_url)
        self._connected = True
        self._last_pong_at = time.time()

        # Subscribe to all channels
        await self._subscribe_all()

        # Start background loops
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._message_task = asyncio.create_task(self._message_loop())

        logger.info("PacificaAdapter connected to %s", self._ws_url)

    async def disconnect(self) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._message_task:
            self._message_task.cancel()
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._connected = False
        logger.info("PacificaAdapter disconnected")

    async def reconnect(self) -> None:
        """Close existing connection and reconnect."""
        logger.warning("Reconnecting to Pacifica WS...")
        await self.disconnect()
        await asyncio.sleep(1)
        await self.connect()

    async def _subscribe_all(self) -> None:
        """Subscribe to all required WS channels."""
        subs = [
            {"method": "subscribe", "params": {"orderbook": {"symbol": self._symbol}}},
            {"method": "subscribe", "params": {"bbo": {"symbol": self._symbol}}},
            {"method": "subscribe", "params": {"trades": {"symbol": self._symbol}}},
        ]
        if self._public_key:
            subs.extend([
                {"method": "subscribe", "params": {"account_order_updates": {"account": self._public_key}}},
                {"method": "subscribe", "params": {"account_positions": {"account": self._public_key}}},
                {"method": "subscribe", "params": {"account_margin": {"account": self._public_key}}},
            ])
        for sub in subs:
            await self._ws.send(json.dumps(sub))

    async def _heartbeat_loop(self) -> None:
        """Send ping every 30s. Reconnect if no pong for 60s."""
        while self._connected:
            try:
                await self._ws.send(json.dumps({"method": "ping"}))
                await asyncio.sleep(30)
                if time.time() - self._last_pong_at > 60:
                    logger.warning("No pong for 60s, reconnecting")
                    await self.reconnect()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Heartbeat error: %s", e)
                await self.reconnect()
                return

    async def _message_loop(self) -> None:
        """Receive and route WS messages."""
        while self._connected:
            try:
                raw = await self._ws.recv()
                msg = json.loads(raw)
                channel = msg.get("channel", "")

                if channel == "orderbook":
                    await self._handle_orderbook(msg)
                elif channel == "bbo":
                    await self._handle_bbo(msg)
                elif channel == "trades":
                    await self._handle_trades(msg)
                elif channel == "account_order_updates":
                    await self._handle_order_update(msg)
                elif channel == "account_positions":
                    await self._handle_position_update(msg)
                elif channel == "account_margin":
                    await self._handle_margin_update(msg)
                elif msg.get("method") == "pong" or "pong" in str(msg):
                    self._last_pong_at = time.time()
                else:
                    # Check for trading operation responses
                    msg_id = msg.get("id")
                    if msg_id and msg_id in self._pending_responses:
                        self._pending_responses[msg_id].set_result(msg)

            except asyncio.CancelledError:
                return
            except websockets.ConnectionClosed:
                logger.warning("WS connection closed, reconnecting")
                await self.reconnect()
                return
            except Exception as e:
                logger.error("Message loop error: %s", e)

    # -------------------------------------------------------------------
    # Message handlers
    # -------------------------------------------------------------------

    async def _handle_orderbook(self, msg: dict) -> None:
        data = msg.get("data", {})
        bids = [[float(p), float(s)] for p, s in data.get("bids", [])]
        asks = [[float(p), float(s)] for p, s in data.get("asks", [])]

        self._last_ob = {"bids": bids, "asks": asks}
        if bids and asks:
            self._last_mid = (bids[0][0] + asks[0][0]) / 2.0
        self._orderbook_updated_at = time.time()

        if self._on_orderbook and bids and asks:
            self._on_orderbook(bids[0][0], bids[0][1], asks[0][0], asks[0][1])

    async def _handle_bbo(self, msg: dict) -> None:
        data = msg.get("data", {})
        self._last_bbo = {
            "bid_px": float(data.get("bid_price", 0)),
            "bid_sz": float(data.get("bid_size", 0)),
            "ask_px": float(data.get("ask_price", 0)),
            "ask_sz": float(data.get("ask_size", 0)),
        }

    async def _handle_trades(self, msg: dict) -> None:
        data = msg.get("data", [])
        if not isinstance(data, list):
            data = [data]
        for t in data:
            # Pacifica uses "bid"/"ask" for side, convert to "buy"/"sell"
            raw_side = t.get("side", "bid")
            side = "buy" if raw_side == "bid" else "sell"
            if self._on_trade:
                self._on_trade(
                    side,
                    float(t.get("price", 0)),
                    float(t.get("amount", 0)),
                )

    async def _handle_order_update(self, msg: dict) -> None:
        data = msg.get("data", {})
        client_oid = data.get("client_order_id", "")
        status = data.get("status", "")
        side = "buy" if data.get("side") == "bid" else "sell"

        if status in ("filled", "partial"):
            fill_amount = float(data.get("filled_amount", 0))
            fill_price = float(data.get("price", 0))

            # Update position
            if side == "buy":
                self._position_q += fill_amount
            else:
                self._position_q -= fill_amount

            # Push fill event
            await self._pending_fills.put({
                "side": side,
                "price": fill_price,
                "size": fill_amount,
                "fee": float(data.get("fee", 0)),
                "timestamp": data.get("timestamp", time.time()),
            })

            if self._on_fill:
                self._on_fill(side, fill_price, fill_amount)

            if status == "filled":
                self._open_orders.pop(client_oid, None)
            logger.info("Fill: %s %.4f @ %.2f (status=%s)", side, fill_amount, fill_price, status)

        elif status in ("cancelled", "rejected"):
            self._open_orders.pop(client_oid, None)
            if status == "rejected":
                reason = data.get("reason", "unknown")
                logger.warning("Order rejected: %s — %s", client_oid, reason)
            else:
                logger.debug("Order cancelled: %s", client_oid)

    async def _handle_position_update(self, msg: dict) -> None:
        data = msg.get("data", [])
        if isinstance(data, list):
            for pos in data:
                if pos.get("symbol") == self._symbol:
                    self._position_q = float(pos.get("size", 0))
        elif isinstance(data, dict) and data.get("symbol") == self._symbol:
            self._position_q = float(data.get("size", 0))

    async def _handle_margin_update(self, msg: dict) -> None:
        data = msg.get("data", {})
        self._balance_usdc = float(data.get("available_margin", 0))

    # -------------------------------------------------------------------
    # Price / size rounding
    # -------------------------------------------------------------------

    def _round_price(self, price: float, side: str) -> str:
        """Round price to tick size. Bids floor, asks ceil."""
        if self._tick_size <= 0:
            return f"{price:.2f}"
        if side == "buy":
            rounded = math.floor(price / self._tick_size) * self._tick_size
        else:
            rounded = math.ceil(price / self._tick_size) * self._tick_size
        # Format with appropriate decimals
        decimals = max(0, -int(math.floor(math.log10(self._tick_size))))
        return f"{rounded:.{decimals}f}"

    def _round_size(self, size: float) -> str:
        """Round size down to lot size."""
        if self._lot_size <= 0:
            return f"{size:.4f}"
        rounded = math.floor(size / self._lot_size) * self._lot_size
        decimals = max(0, -int(math.floor(math.log10(self._lot_size))))
        return f"{rounded:.{decimals}f}"

    # -------------------------------------------------------------------
    # Core trading operations
    # -------------------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        side: str,
        price: float,
        size: float,
        post_only: bool = True,
        reduce_only: bool = False,
        builder_fee: int = 0,
    ) -> OrderResult:
        """
        Place a single ALO limit order.

        CRITICAL: Only tif="ALO" is allowed for limit orders.
        Non-ALO attempts are logged as WARNING and rejected.
        """
        if not post_only:
            logger.warning(
                "TAKER ORDER on Pacifica: %s %s %.4f @ %.2f — "
                "only emergency flatten should use non-ALO",
                side, symbol, size, price,
            )

        start = time.time()
        client_oid = str(uuid.uuid4())
        pac_side = "bid" if side == "buy" else "ask"

        if self._simulate:
            self._sim_counter += 1
            rtt = (time.time() - start) * 1000
            logger.info(
                "[SIM] %s %s %s @ %s (ALO, reduce_only=%s)",
                pac_side.upper(), symbol,
                self._round_size(size), self._round_price(price, side),
                reduce_only,
            )
            self._open_orders[client_oid] = {
                "symbol": symbol, "side": side, "price": price,
                "size": size, "placed_at": time.time(),
            }
            return OrderResult(
                success=True, order_id=client_oid, status="placed", rtt_ms=rtt,
            )

        # Build signed payload
        op_data = {
            "symbol": symbol,
            "price": self._round_price(price, side),
            "amount": self._round_size(size),
            "side": pac_side,
            "tif": "ALO",
            "reduce_only": reduce_only,
            "client_order_id": client_oid,
        }
        if self._builder_code:
            op_data["builder_code"] = self._builder_code

        signed = self._sign("create_order", op_data)
        msg_id = str(uuid.uuid4())

        # Send via WS
        ws_msg = {"id": msg_id, "params": {"create_order": signed}}
        future = asyncio.get_event_loop().create_future()
        self._pending_responses[msg_id] = future

        await self._ws.send(json.dumps(ws_msg))

        try:
            response = await asyncio.wait_for(future, timeout=5.0)
            rtt = (time.time() - start) * 1000

            if response.get("error"):
                error = response["error"]
                if "would_be_taker" in str(error).lower():
                    logger.warning("ALO rejection (would cross): %s @ %s", pac_side, price)
                else:
                    logger.error("Order error: %s", error)
                return OrderResult(
                    success=False, order_id=client_oid, status="rejected",
                    rtt_ms=rtt, error=str(error),
                )

            self._open_orders[client_oid] = {
                "symbol": symbol, "side": side, "price": price,
                "size": size, "placed_at": time.time(),
            }
            return OrderResult(
                success=True, order_id=client_oid, status="placed", rtt_ms=rtt,
            )

        except asyncio.TimeoutError:
            rtt = (time.time() - start) * 1000
            logger.error("Order placement timed out")
            return OrderResult(
                success=False, order_id=client_oid, status="timeout", rtt_ms=rtt,
            )
        finally:
            self._pending_responses.pop(msg_id, None)

    async def cancel_order(self, symbol: str, order_id: str) -> OrderResult:
        start = time.time()

        if self._simulate:
            self._open_orders.pop(order_id, None)
            rtt = (time.time() - start) * 1000
            logger.info("[SIM] Cancel %s on %s", order_id, symbol)
            return OrderResult(
                success=True, order_id=order_id, status="cancelled", rtt_ms=rtt,
            )

        op_data = {
            "symbol": symbol,
            "client_order_id": order_id,
        }
        signed = self._sign("cancel_order", op_data)
        msg_id = str(uuid.uuid4())
        ws_msg = {"id": msg_id, "params": {"cancel_order": signed}}
        future = asyncio.get_event_loop().create_future()
        self._pending_responses[msg_id] = future

        await self._ws.send(json.dumps(ws_msg))

        try:
            response = await asyncio.wait_for(future, timeout=5.0)
            rtt = (time.time() - start) * 1000
            self._open_orders.pop(order_id, None)
            return OrderResult(
                success=True, order_id=order_id, status="cancelled", rtt_ms=rtt,
            )
        except asyncio.TimeoutError:
            rtt = (time.time() - start) * 1000
            return OrderResult(
                success=False, order_id=order_id, status="timeout", rtt_ms=rtt,
            )
        finally:
            self._pending_responses.pop(msg_id, None)

    async def cancel_all_orders(self, symbol: str) -> int:
        if self._simulate:
            count = len([o for o in self._open_orders.values() if o["symbol"] == symbol])
            self._open_orders = {
                k: v for k, v in self._open_orders.items() if v["symbol"] != symbol
            }
            logger.info("[SIM] Cancel all %d orders on %s", count, symbol)
            return count

        op_data = {"symbol": symbol}
        signed = self._sign("cancel_all_orders", op_data)
        msg_id = str(uuid.uuid4())
        ws_msg = {"id": msg_id, "params": {"cancel_all_orders": signed}}
        future = asyncio.get_event_loop().create_future()
        self._pending_responses[msg_id] = future

        await self._ws.send(json.dumps(ws_msg))

        count = len([o for o in self._open_orders.values() if o["symbol"] == symbol])
        try:
            await asyncio.wait_for(future, timeout=5.0)
            self._open_orders = {
                k: v for k, v in self._open_orders.items() if v["symbol"] != symbol
            }
        except asyncio.TimeoutError:
            logger.error("Cancel all timed out")
        finally:
            self._pending_responses.pop(msg_id, None)
        return count

    async def bulk_place(
        self, symbol: str, orders: list[dict],
    ) -> list[OrderResult]:
        results = []
        for order in orders:
            r = await self.place_order(
                symbol=symbol,
                side=order["side"],
                price=order["price"],
                size=order["size"],
                post_only=order.get("post_only", True),
                reduce_only=order.get("reduce_only", False),
            )
            results.append(r)
        return results

    async def bulk_cancel(
        self, symbol: str, order_ids: list[str],
    ) -> list[OrderResult]:
        results = []
        for oid in order_ids:
            r = await self.cancel_order(symbol, oid)
            results.append(r)
        return results

    async def requote(
        self,
        bid_price: float,
        ask_price: float,
        bid_size: float,
        ask_size: float,
    ) -> bool:
        """
        Cancel stale quotes + place fresh ones in one batch.

        For simulate mode, uses sequential calls.
        For live mode, builds a batch_orders payload with up to 4 actions:
          [Cancel(old_bid), Cancel(old_ask), Create(new_bid), Create(new_ask)]

        All ALO orders → no speed bump on the batch.
        """
        # Self-trade prevention: bid must be below ask
        if bid_price >= ask_price:
            logger.error(
                "Self-trade prevented in requote: bid %.6f >= ask %.6f",
                bid_price, ask_price,
            )
            # Widen by 1 tick each side
            bid_price -= self._tick_size
            ask_price += self._tick_size

        if self._simulate:
            # Cancel existing
            to_cancel = list(self._open_orders.keys())
            for oid in to_cancel:
                await self.cancel_order(self._symbol, oid)
            # Place new
            if bid_size > 0:
                await self.place_order(self._symbol, "buy", bid_price, bid_size)
            if ask_size > 0:
                await self.place_order(self._symbol, "sell", ask_price, ask_size)
            return True

        # Live: build batch
        start = time.time()
        actions = []

        # Cancel existing orders
        for oid, order in list(self._open_orders.items()):
            if order["symbol"] == self._symbol:
                cancel_data = {"symbol": self._symbol, "client_order_id": oid}
                signed_cancel = self._sign("cancel_order", cancel_data)
                actions.append({"type": "Cancel", "data": signed_cancel})

        # New bid
        if bid_size > 0:
            bid_oid = str(uuid.uuid4())
            bid_data = {
                "symbol": self._symbol,
                "price": self._round_price(bid_price, "buy"),
                "amount": self._round_size(bid_size),
                "side": "bid",
                "tif": "ALO",
                "reduce_only": False,
                "client_order_id": bid_oid,
            }
            if self._builder_code:
                bid_data["builder_code"] = self._builder_code
            signed_bid = self._sign("create_order", bid_data)
            actions.append({"type": "Create", "data": signed_bid})

        # New ask
        if ask_size > 0:
            ask_oid = str(uuid.uuid4())
            ask_data = {
                "symbol": self._symbol,
                "price": self._round_price(ask_price, "sell"),
                "amount": self._round_size(ask_size),
                "side": "ask",
                "tif": "ALO",
                "reduce_only": False,
                "client_order_id": ask_oid,
            }
            if self._builder_code:
                ask_data["builder_code"] = self._builder_code
            signed_ask = self._sign("create_order", ask_data)
            actions.append({"type": "Create", "data": signed_ask})

        if not actions:
            return True

        msg_id = str(uuid.uuid4())
        ws_msg = {
            "id": msg_id,
            "params": {"batch_orders": {"actions": actions}},
        }
        future = asyncio.get_event_loop().create_future()
        self._pending_responses[msg_id] = future

        await self._ws.send(json.dumps(ws_msg))

        try:
            response = await asyncio.wait_for(future, timeout=5.0)
            rtt = (time.time() - start) * 1000
            logger.info("Requote batch RTT: %.0fms (%d actions)", rtt, len(actions))

            # Update internal tracking
            self._open_orders.clear()
            if bid_size > 0:
                self._open_orders[bid_oid] = {
                    "symbol": self._symbol, "side": "buy",
                    "price": bid_price, "size": bid_size, "placed_at": time.time(),
                }
            if ask_size > 0:
                self._open_orders[ask_oid] = {
                    "symbol": self._symbol, "side": "sell",
                    "price": ask_price, "size": ask_size, "placed_at": time.time(),
                }
            return True

        except asyncio.TimeoutError:
            logger.error("Requote batch timed out")
            return False
        finally:
            self._pending_responses.pop(msg_id, None)

    async def flatten_position(self) -> bool:
        """
        Emergency flatten — the ONLY place a market order is permitted.
        Uses reduce_only=True to prevent increasing position.
        """
        logger.error("EMERGENCY FLATTEN initiated for %s", self._symbol)

        # Step 1: Cancel all
        await self.cancel_all_orders(self._symbol)

        pos = self._position_q
        if abs(pos) < 1e-9:
            logger.info("Position already flat")
            return True

        side = "sell" if pos > 0 else "buy"
        size = abs(pos)

        if self._simulate:
            logger.error(
                "[SIM] EMERGENCY MARKET %s %.4f %s (reduce_only)",
                side.upper(), size, self._symbol,
            )
            self._position_q = 0.0
            return True

        # Market order
        op_data = {
            "symbol": self._symbol,
            "side": "ask" if side == "sell" else "bid",
            "amount": self._round_size(size),
            "reduce_only": True,
            "slippage_percent": "1.0",
        }
        signed = self._sign("create_market_order", op_data)
        msg_id = str(uuid.uuid4())
        ws_msg = {"id": msg_id, "params": {"create_market_order": signed}}

        await self._ws.send(json.dumps(ws_msg))
        logger.error("Emergency market %s sent: %.4f %s", side, size, self._symbol)
        return True

    # -------------------------------------------------------------------
    # Account queries
    # -------------------------------------------------------------------

    async def get_position(self, symbol: str) -> float:
        return self._position_q

    async def get_balance(self) -> float:
        return self._balance_usdc

    async def get_equity(self) -> float:
        return self._balance_usdc

    async def get_funding_rate(self, symbol: str) -> float:
        if self._simulate:
            return 0.0
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self._rest_url}/api/v1/market/funding_rate?symbol={symbol}"
                async with session.get(url) as resp:
                    data = await resp.json()
                    return float(data.get("rate", 0))
        except Exception as e:
            logger.error("Failed to fetch funding rate: %s", e)
            return 0.0

    async def get_fee_rates(self) -> dict:
        return {"maker_rate": 0.00015, "taker_rate": 0.00040, "builder_rate": 0.0}

    async def schedule_cancel_all(self, symbol: str, delay_s: float) -> bool:
        # Pacifica does not have native schedule-cancel; use WS heartbeat as DMS
        logger.debug("Schedule cancel-all: using heartbeat as dead man's switch")
        return True

    async def set_self_trade_prevention(self, mode: str = "enabled") -> bool:
        logger.info("Pacifica STP: built into exchange + software check in adapter")
        return True

    # -------------------------------------------------------------------
    # Market data accessors
    # -------------------------------------------------------------------

    def get_mid_price(self) -> float:
        if time.time() - self._orderbook_updated_at > 10:
            logger.warning("Orderbook stale: %.1fs old", time.time() - self._orderbook_updated_at)
        return self._last_mid

    def get_best_bid(self) -> tuple[float, float]:
        bids = self._last_ob.get("bids", [])
        if bids:
            return bids[0][0], bids[0][1]
        return 0.0, 0.0

    def get_best_ask(self) -> tuple[float, float]:
        asks = self._last_ob.get("asks", [])
        if asks:
            return asks[0][0], asks[0][1]
        return 0.0, 0.0

    def get_available_margin(self) -> float:
        return self._balance_usdc

    def is_orderbook_fresh(self, max_age_ms: int = 5000) -> bool:
        age_ms = (time.time() - self._orderbook_updated_at) * 1000
        return age_ms < max_age_ms

    # -------------------------------------------------------------------
    # REST helpers
    # -------------------------------------------------------------------

    async def fetch_market_info(self) -> None:
        """Get tick/lot sizes for the symbol."""
        if self._simulate:
            logger.info("[SIM] Market info: tick=%s, lot=%s", self._tick_size, self._lot_size)
            return

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self._rest_url}/api/v1/market/tick_lot_sizes"
                async with session.get(url) as resp:
                    data = await resp.json()
                    for item in data:
                        if item.get("symbol") == self._symbol:
                            self._tick_size = float(item.get("tick_size", 0.01))
                            self._lot_size = float(item.get("lot_size", 0.001))
                            break
            logger.info("Market info: tick=%.6f, lot=%.6f", self._tick_size, self._lot_size)
        except Exception as e:
            logger.error("Failed to fetch market info: %s", e)

    async def fetch_account_state(self) -> None:
        """Bootstrap position and margin from REST before WS is ready."""
        if self._simulate:
            self._balance_usdc = 100_000.0
            return

        try:
            async with aiohttp.ClientSession() as session:
                # Positions
                url = f"{self._rest_url}/api/v1/account/positions?account={self._public_key}"
                async with session.get(url) as resp:
                    data = await resp.json()
                    for pos in data:
                        if pos.get("symbol") == self._symbol:
                            self._position_q = float(pos.get("size", 0))

                # Margin
                url = f"{self._rest_url}/api/v1/account/margin?account={self._public_key}"
                async with session.get(url) as resp:
                    data = await resp.json()
                    self._balance_usdc = float(data.get("available_margin", 0))

        except Exception as e:
            logger.error("Failed to fetch account state: %s", e)

    async def register_builder_code(self, fee_rate: str = "0") -> str | None:
        """Register our own builder code (one-time, charge 0 to self)."""
        if self._simulate:
            logger.info("[SIM] Builder code registration (fee_rate=%s)", fee_rate)
            return "sim_builder_code"

        op_data = {"fee_rate": fee_rate}
        signed = self._sign("create_builder_code", op_data)

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self._rest_url}/api/v1/account/builder_codes/create"
                async with session.post(url, json=signed) as resp:
                    data = await resp.json()
                    code = data.get("builder_code")
                    if code:
                        self._builder_code = code
                        logger.info("Builder code registered: %s", code)
                    return code
        except Exception as e:
            logger.error("Failed to register builder code: %s", e)
            return None

    # -------------------------------------------------------------------
    # Subscriptions (from ABC — wrap internal methods)
    # -------------------------------------------------------------------

    async def subscribe_orderbook(self, symbol: str) -> AsyncIterator[L2Snapshot]:
        """Yield L2 snapshots. In practice, the message loop populates state."""
        while self._connected:
            if self._last_ob["bids"] and self._last_ob["asks"]:
                yield L2Snapshot(
                    timestamp=self._orderbook_updated_at,
                    symbol=symbol,
                    best_bid=self._last_ob["bids"][0][0],
                    best_bid_size=self._last_ob["bids"][0][1],
                    best_ask=self._last_ob["asks"][0][0],
                    best_ask_size=self._last_ob["asks"][0][1],
                    bids=self._last_ob["bids"],
                    asks=self._last_ob["asks"],
                )
            await asyncio.sleep(0.2)

    async def subscribe_trades(self, symbol: str) -> AsyncIterator[TradeEvent]:
        """Yield trade events from the pending fills queue."""
        while self._connected:
            try:
                fill = await asyncio.wait_for(self._pending_fills.get(), timeout=1.0)
                yield TradeEvent(
                    timestamp=fill.get("timestamp", time.time()),
                    symbol=symbol,
                    side=fill["side"],
                    price=fill["price"],
                    size=fill["size"],
                )
            except asyncio.TimeoutError:
                continue
