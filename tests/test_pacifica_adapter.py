"""
Tests for PacificaAdapter.

Unit tests run without network dependencies (simulate mode).
Integration tests against testnet are marked with @pytest.mark.testnet
and skipped by default.

Covers:
  - Signing: recursive key sort, payload structure
  - Simulate mode: order placement, cancellation, requote
  - ALO enforcement: non-ALO requests logged as warning
  - Self-trade prevention in requote
  - Price/size rounding
  - Order tracking
  - Flatten position
  - Fee engine integration with Pacifica tiers
"""

import pytest
import asyncio
import json
from MM_algo.adapters.pacifica import PacificaAdapter
from MM_algo.core.fee_engine import FeeEngine, FeeConfig, PACIFICA_TIERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_adapter(**kwargs) -> PacificaAdapter:
    return PacificaAdapter(simulate=True, **kwargs)


@pytest.fixture
def adapter():
    a = make_adapter(symbol="BTC")
    asyncio.get_event_loop().run_until_complete(a.connect())
    return a


# ---------------------------------------------------------------------------
# Test 1: Signing — recursive key sorting
# ---------------------------------------------------------------------------

class TestSigning:

    def test_sort_json_keys_flat(self):
        a = make_adapter()
        result = a._sort_json_keys({"b": 1, "a": 2})
        keys = list(result.keys())
        assert keys == ["a", "b"]

    def test_sort_json_keys_nested(self):
        a = make_adapter()
        result = a._sort_json_keys({
            "z": {"b": 1, "a": 2},
            "a": [{"c": 3, "a": 1}],
        })
        outer_keys = list(result.keys())
        assert outer_keys == ["a", "z"]
        inner_keys = list(result["z"].keys())
        assert inner_keys == ["a", "b"]
        list_inner_keys = list(result["a"][0].keys())
        assert list_inner_keys == ["a", "c"]

    def test_sort_json_keys_primitives(self):
        a = make_adapter()
        assert a._sort_json_keys(42) == 42
        assert a._sort_json_keys("hello") == "hello"
        assert a._sort_json_keys(None) is None


# ---------------------------------------------------------------------------
# Test 2: Simulate mode connection
# ---------------------------------------------------------------------------

class TestSimulateMode:

    def test_connect_simulate(self, adapter):
        assert adapter._connected is True
        assert adapter._simulate is True

    @pytest.mark.asyncio
    async def test_place_order_simulate(self):
        a = make_adapter(symbol="BTC")
        await a.connect()
        result = await a.place_order("BTC", "buy", 50000.0, 0.1)
        assert result.success is True
        assert result.status == "placed"
        assert len(a._open_orders) == 1

    @pytest.mark.asyncio
    async def test_cancel_order_simulate(self):
        a = make_adapter(symbol="BTC")
        await a.connect()
        result = await a.place_order("BTC", "buy", 50000.0, 0.1)
        oid = result.order_id
        cancel = await a.cancel_order("BTC", oid)
        assert cancel.success is True
        assert len(a._open_orders) == 0

    @pytest.mark.asyncio
    async def test_cancel_all_simulate(self):
        a = make_adapter(symbol="BTC")
        await a.connect()
        await a.place_order("BTC", "buy", 50000.0, 0.1)
        await a.place_order("BTC", "sell", 51000.0, 0.1)
        await a.place_order("BTC", "buy", 49000.0, 0.2)
        count = await a.cancel_all_orders("BTC")
        assert count == 3
        assert len(a._open_orders) == 0


# ---------------------------------------------------------------------------
# Test 3: ALO enforcement
# ---------------------------------------------------------------------------

class TestALOEnforcement:

    @pytest.mark.asyncio
    async def test_alo_default(self):
        a = make_adapter(symbol="BTC")
        await a.connect()
        # Default should be post_only=True (ALO)
        result = await a.place_order("BTC", "buy", 50000.0, 0.1)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_non_alo_logs_warning(self, caplog):
        import logging
        a = make_adapter(symbol="BTC")
        await a.connect()
        with caplog.at_level(logging.WARNING):
            await a.place_order("BTC", "buy", 50000.0, 0.1, post_only=False)
        assert "TAKER ORDER" in caplog.text


# ---------------------------------------------------------------------------
# Test 4: Requote (cancel + place batch)
# ---------------------------------------------------------------------------

class TestRequote:

    @pytest.mark.asyncio
    async def test_requote_basic(self):
        a = make_adapter(symbol="BTC")
        await a.connect()
        # Place initial quotes
        await a.place_order("BTC", "buy", 49000.0, 0.1)
        await a.place_order("BTC", "sell", 51000.0, 0.1)
        assert len(a._open_orders) == 2

        # Requote
        result = await a.requote(49500.0, 50500.0, 0.15, 0.15)
        assert result is True
        assert len(a._open_orders) == 2

    @pytest.mark.asyncio
    async def test_requote_self_trade_prevention(self, caplog):
        import logging
        a = make_adapter(symbol="BTC")
        await a.connect()
        with caplog.at_level(logging.ERROR):
            # bid >= ask should be caught
            await a.requote(100.0, 99.0, 0.1, 0.1)
        assert "Self-trade prevented" in caplog.text


# ---------------------------------------------------------------------------
# Test 5: Price/size rounding
# ---------------------------------------------------------------------------

class TestRounding:

    def test_bid_price_floors(self):
        a = make_adapter()
        a._tick_size = 0.5
        price = a._round_price(100.7, "buy")
        assert float(price) == 100.5  # floor

    def test_ask_price_ceils(self):
        a = make_adapter()
        a._tick_size = 0.5
        price = a._round_price(100.3, "sell")
        assert float(price) == 100.5  # ceil

    def test_size_floors(self):
        a = make_adapter()
        a._lot_size = 0.01
        size = a._round_size(1.567)
        assert float(size) == 1.56


# ---------------------------------------------------------------------------
# Test 6: Flatten position
# ---------------------------------------------------------------------------

class TestFlatten:

    @pytest.mark.asyncio
    async def test_flatten_long(self):
        a = make_adapter(symbol="BTC")
        await a.connect()
        a._position_q = 1.5
        result = await a.flatten_position()
        assert result is True
        assert a._position_q == 0.0

    @pytest.mark.asyncio
    async def test_flatten_short(self):
        a = make_adapter(symbol="BTC")
        await a.connect()
        a._position_q = -0.8
        result = await a.flatten_position()
        assert result is True
        assert a._position_q == 0.0

    @pytest.mark.asyncio
    async def test_flatten_already_flat(self):
        a = make_adapter(symbol="BTC")
        await a.connect()
        a._position_q = 0.0
        result = await a.flatten_position()
        assert result is True


# ---------------------------------------------------------------------------
# Test 7: Orderbook freshness
# ---------------------------------------------------------------------------

class TestOrderbookFreshness:

    def test_stale_orderbook(self):
        a = make_adapter()
        a._orderbook_updated_at = 0  # very old
        assert a.is_orderbook_fresh(5000) is False

    def test_fresh_orderbook(self):
        import time
        a = make_adapter()
        a._orderbook_updated_at = time.time()
        assert a.is_orderbook_fresh(5000) is True


# ---------------------------------------------------------------------------
# Test 8: Pacifica fee tiers
# ---------------------------------------------------------------------------

class TestPacificaFees:

    def test_pacifica_tier1_rates(self):
        fe = FeeEngine(FeeConfig(exchange="pacifica", volume_tier=0))
        assert fe.get_maker_rate() == pytest.approx(0.000_15, abs=1e-9)
        assert fe.get_taker_rate() == pytest.approx(0.000_40, abs=1e-9)

    def test_pacifica_vip1_zero_maker(self):
        fe = FeeEngine(FeeConfig(exchange="pacifica", volume_tier=5))
        assert fe.get_maker_rate() == pytest.approx(0.0, abs=1e-9)

    def test_pacifica_rwa_50pct_discount(self):
        fe = FeeEngine(FeeConfig(
            exchange="pacifica", volume_tier=0, is_rwa_market=True,
        ))
        # Tier 1 maker 0.015% × 50% = 0.0075%
        assert fe.get_maker_rate() == pytest.approx(0.000_075, abs=1e-9)
        assert fe.get_taker_rate() == pytest.approx(0.000_20, abs=1e-9)

    def test_pacifica_rwa_symbol_detection(self):
        fe = FeeEngine(FeeConfig(
            exchange="pacifica", volume_tier=0, symbol="TSLA",
        ))
        # TSLA is in RWA list → 50% discount auto-applied
        assert fe.get_maker_rate() == pytest.approx(0.000_075, abs=1e-9)

    def test_pacifica_no_staking_no_rebate(self):
        """Pacifica has no staking discounts or maker-share rebates."""
        fe = FeeEngine(FeeConfig(
            exchange="pacifica", volume_tier=0,
            staking_tier="diamond",  # should be ignored
            maker_volume_share_pct=0.04,  # should be ignored
        ))
        # Should be raw tier 1 rates, no adjustments
        assert fe.get_maker_rate() == pytest.approx(0.000_15, abs=1e-9)

    def test_pacifica_round_trip(self):
        fe = FeeEngine(FeeConfig(exchange="pacifica", volume_tier=0))
        rt = fe.compute_round_trip(100_000, "standard")
        # 2 × 1.5bps × $100k = $30
        assert rt == pytest.approx(30.0, abs=0.01)

    def test_pacifica_fee_report(self):
        fe = FeeEngine(FeeConfig(exchange="pacifica", volume_tier=2))
        report = fe.fee_report()
        assert "T2" in report


# ---------------------------------------------------------------------------
# Test 9: Fee viability end-to-end
# ---------------------------------------------------------------------------

class TestFeeViabilityE2E:

    def test_min_spread_pacifica_tier1(self):
        fe = FeeEngine(FeeConfig(exchange="pacifica", volume_tier=0, builder_fee_bps=0))
        min_spread = fe.min_viable_spread_bps(10_000)
        # 2 × 0.015% / 0.8 = 0.0375% = 3.75 bps
        assert min_spread == pytest.approx(3.75, abs=0.1)

    def test_viability_check(self):
        fe = FeeEngine(FeeConfig(exchange="pacifica", volume_tier=0))
        assert fe.is_viable(5.0, 10_000) is True
        assert fe.is_viable(1.0, 10_000) is False


# ---------------------------------------------------------------------------
# Test 10: Message parsing
# ---------------------------------------------------------------------------

class TestMessageParsing:

    @pytest.mark.asyncio
    async def test_handle_orderbook(self):
        a = make_adapter(symbol="BTC")
        await a.connect()
        msg = {
            "channel": "orderbook",
            "data": {
                "symbol": "BTC",
                "bids": [["99000.00", "0.5"], ["98999.00", "1.2"]],
                "asks": [["99001.00", "0.3"], ["99002.00", "0.8"]],
            },
        }
        await a._handle_orderbook(msg)
        assert a._last_mid == pytest.approx(99000.5, abs=0.01)
        assert a._last_ob["bids"][0] == [99000.0, 0.5]
        assert a._last_ob["asks"][0] == [99001.0, 0.3]

    @pytest.mark.asyncio
    async def test_handle_bbo(self):
        a = make_adapter(symbol="BTC")
        await a.connect()
        msg = {
            "channel": "bbo",
            "data": {
                "symbol": "BTC",
                "bid_price": "99000.00",
                "bid_size": "0.5",
                "ask_price": "99001.00",
                "ask_size": "0.3",
            },
        }
        await a._handle_bbo(msg)
        assert a._last_bbo["bid_px"] == pytest.approx(99000.0)
        assert a._last_bbo["ask_sz"] == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_handle_trade_side_conversion(self):
        """Pacifica uses 'bid'/'ask' for trade side, we need 'buy'/'sell'."""
        a = make_adapter(symbol="BTC")
        await a.connect()
        received_sides = []
        a._on_trade = lambda side, price, size: received_sides.append(side)

        msg = {
            "channel": "trades",
            "data": [
                {"symbol": "BTC", "price": "99000", "amount": "0.1", "side": "bid"},
                {"symbol": "BTC", "price": "99001", "amount": "0.2", "side": "ask"},
            ],
        }
        await a._handle_trades(msg)
        assert received_sides == ["buy", "sell"]

    @pytest.mark.asyncio
    async def test_handle_position_update(self):
        a = make_adapter(symbol="BTC")
        await a.connect()
        msg = {
            "channel": "account_positions",
            "data": [{"symbol": "BTC", "size": "0.5"}],
        }
        await a._handle_position_update(msg)
        assert a._position_q == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_handle_margin_update(self):
        a = make_adapter(symbol="BTC")
        await a.connect()
        msg = {
            "channel": "account_margin",
            "data": {"available_margin": "8500.00"},
        }
        await a._handle_margin_update(msg)
        assert a._balance_usdc == pytest.approx(8500.0)
