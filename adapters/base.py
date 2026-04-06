"""
ExchangeAdapter — abstract base class for all exchange connectors.

Every exchange adapter must implement this interface. The algorithm
only interacts with exchanges through this ABC, making the strategy
exchange-agnostic.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class L2Snapshot:
    """Top-of-book L2 data."""
    timestamp: float
    symbol: str
    best_bid: float
    best_bid_size: float
    best_ask: float
    best_ask_size: float
    bids: list[list[float]]   # [[price, size], ...] sorted descending
    asks: list[list[float]]   # [[price, size], ...] sorted ascending


@dataclass
class TradeEvent:
    """A single trade on the exchange."""
    timestamp: float
    symbol: str
    side: str          # "buy" or "sell"
    price: float
    size: float
    trade_id: str = ""


@dataclass
class OrderResult:
    """Result from placing or cancelling an order."""
    success: bool
    order_id: str = ""
    status: str = ""       # "placed", "cancelled", "rejected", "filled"
    rtt_ms: float = 0.0    # round-trip latency
    error: str = ""
    fill_price: float = 0.0
    fill_size: float = 0.0


class ExchangeAdapter(abc.ABC):
    """
    Abstract interface to any orderbook DEX.

    All methods that touch the network are async.
    Data feeds are async generators.
    """

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish connection to the exchange."""

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Clean shutdown — cancel all orders, close connections."""

    @abc.abstractmethod
    async def subscribe_orderbook(self, symbol: str) -> AsyncIterator[L2Snapshot]:
        """Async generator yielding L2 snapshots on every update."""

    @abc.abstractmethod
    async def subscribe_trades(self, symbol: str) -> AsyncIterator[TradeEvent]:
        """Async generator yielding trade events."""

    @abc.abstractmethod
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
        Place a limit order.

        IMPORTANT: post_only defaults to True (ALO). A WARNING is logged
        if post_only=False is explicitly passed.
        """

    @abc.abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> OrderResult:
        """Cancel a single order."""

    @abc.abstractmethod
    async def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all open orders for a symbol. Returns count cancelled."""

    @abc.abstractmethod
    async def bulk_place(
        self,
        symbol: str,
        orders: list[dict],
    ) -> list[OrderResult]:
        """Place multiple orders in one batch."""

    @abc.abstractmethod
    async def bulk_cancel(
        self,
        symbol: str,
        order_ids: list[str],
    ) -> list[OrderResult]:
        """Cancel multiple orders in one batch."""

    @abc.abstractmethod
    async def get_position(self, symbol: str) -> float:
        """Get current net delta in base asset."""

    @abc.abstractmethod
    async def get_balance(self) -> float:
        """Get available USDC/quote balance."""

    @abc.abstractmethod
    async def get_equity(self) -> float:
        """Get total account equity (balance + unrealised P&L)."""

    @abc.abstractmethod
    async def get_funding_rate(self, symbol: str) -> float:
        """Get current hourly funding rate."""

    @abc.abstractmethod
    async def get_fee_rates(self) -> dict:
        """Get current fee rates: {maker_rate, taker_rate, builder_rate}."""

    @abc.abstractmethod
    async def schedule_cancel_all(self, symbol: str, delay_s: float) -> bool:
        """
        Dead man's switch: schedule a cancel-all N seconds from now.
        On Hyperliquid this uses the scheduleCancel action.
        """

    @abc.abstractmethod
    async def requote(
        self,
        symbol: str,
        bid_price: float,
        bid_size: float,
        ask_price: float,
        ask_size: float,
    ) -> tuple[list[OrderResult], list[OrderResult]]:
        """
        Cancel existing quotes and place new ones atomically.

        Returns: (cancel_results, place_results)
        """

    @abc.abstractmethod
    async def set_self_trade_prevention(self, mode: str) -> bool:
        """
        Enable native self-trade prevention.
        Hyperliquid: expire-maker mode.
        """
