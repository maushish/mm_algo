"""
Comprehensive tests for MarketSignals (signal engine).

Covers:
  - OFI z-score computation and direction
  - Toxicity score and thresholds
  - Consecutive adverse fill halt
  - Volatility regime detection
  - Funding rate pass-through
  - Backtest mode
"""

import pytest
import numpy as np
from MM_algo.core.signal_engine import MarketSignals, MarketState, _toxicity_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def feed_symmetric_book(signals: MarketSignals, mid: float, n: int = 50, ts_start: float = 0.0):
    """Feed N symmetric L2 snapshots around a stable mid."""
    for i in range(n):
        signals.on_l2_snapshot(
            best_bid=mid - 0.01,
            best_bid_size=100.0,
            best_ask=mid + 0.01,
            best_ask_size=100.0,
            timestamp=ts_start + i * 0.5,
        )


def feed_buy_pressure(signals: MarketSignals, mid: float, n: int = 30, ts_start: float = 0.0):
    """Feed snapshots where bid size accelerates (buy pressure).
    Accelerating growth creates non-zero std in OFI buffer so z-score is meaningful."""
    for i in range(n):
        signals.on_l2_snapshot(
            best_bid=mid - 0.01,
            best_bid_size=100.0 + i * i,    # accelerating growth
            best_ask=mid + 0.01,
            best_ask_size=100.0,             # stable
            timestamp=ts_start + i * 0.5,
        )


def feed_sell_pressure(signals: MarketSignals, mid: float, n: int = 30, ts_start: float = 0.0):
    """Feed snapshots where ask size accelerates (sell pressure)."""
    for i in range(n):
        signals.on_l2_snapshot(
            best_bid=mid - 0.01,
            best_bid_size=100.0,              # stable
            best_ask=mid + 0.01,
            best_ask_size=100.0 + i * i,      # accelerating growth
            timestamp=ts_start + i * 0.5,
        )


# ---------------------------------------------------------------------------
# OFI tests
# ---------------------------------------------------------------------------

class TestOFI:

    def test_symmetric_book_zero_ofi(self):
        sig = MarketSignals()
        feed_symmetric_book(sig, 100.0, n=50)
        state = sig.compute()
        # Constant sizes → delta=0 every tick → OFI z-score ≈ 0
        assert abs(state.ofi_zscore) < 0.5

    def test_buy_pressure_positive_ofi(self):
        sig = MarketSignals()
        feed_buy_pressure(sig, 100.0, n=50)
        state = sig.compute()
        # Growing bid size → positive delta_bid → OFI should be positive
        assert state.ofi_zscore > 0

    def test_sell_pressure_negative_ofi(self):
        sig = MarketSignals()
        feed_sell_pressure(sig, 100.0, n=50)
        state = sig.compute()
        # Growing ask size → positive delta_ask → OFI should be negative
        assert state.ofi_zscore < 0

    def test_ofi_reservation_adj_sign(self):
        sig = MarketSignals(ofi_alpha=0.1)
        feed_buy_pressure(sig, 100.0, n=50)
        state = sig.compute()
        # Positive OFI → positive reservation adjustment (raise mid)
        assert state.ofi_reservation_adj >= 0

    def test_ofi_insufficient_data(self):
        sig = MarketSignals()
        sig.on_l2_snapshot(100, 50, 101, 50, timestamp=0)
        state = sig.compute()
        # Only 1 snapshot → no delta → z-score = 0
        assert state.ofi_zscore == 0.0


# ---------------------------------------------------------------------------
# Toxicity tests
# ---------------------------------------------------------------------------

class TestToxicity:

    def test_balanced_trades_low_toxicity(self):
        sig = MarketSignals()
        feed_symmetric_book(sig, 100.0, n=5)
        # Feed equal buys and sells
        for i in range(50):
            side = "buy" if i % 2 == 0 else "sell"
            sig.on_trade(side, 100.0, 1.0, timestamp=i * 0.1)
        state = sig.compute()
        assert state.toxicity_score < 0.1

    def test_one_sided_buys_high_toxicity(self):
        sig = MarketSignals()
        feed_symmetric_book(sig, 100.0, n=5)
        for i in range(100):
            sig.on_trade("buy", 100.0, 1.0, timestamp=i * 0.1)
        state = sig.compute()
        assert state.toxicity_score == pytest.approx(1.0, abs=0.01)

    def test_one_sided_sells_high_toxicity(self):
        sig = MarketSignals()
        feed_symmetric_book(sig, 100.0, n=5)
        for i in range(100):
            sig.on_trade("sell", 100.0, 1.0, timestamp=i * 0.1)
        state = sig.compute()
        assert state.toxicity_score == pytest.approx(1.0, abs=0.01)

    def test_toxicity_response_table(self):
        # Normal
        mult, action = _toxicity_response(0.3)
        assert mult == 1.0 and action == "quote"

        # Caution
        mult, action = _toxicity_response(0.5)
        assert mult == 1.5 and action == "quote"

        # Danger
        mult, action = _toxicity_response(0.7)
        assert mult == 2.5 and action == "quote"

        # Halt
        mult, action = _toxicity_response(0.9)
        assert mult == 2.5 and action == "halt"

    def test_toxicity_triggers_halt(self):
        sig = MarketSignals(toxicity_halt_threshold=0.85)
        feed_symmetric_book(sig, 100.0, n=5)
        # All buys → toxicity = 1.0 > 0.85 → halt
        for i in range(100):
            sig.on_trade("buy", 100.0, 1.0, timestamp=i * 0.1)
        state = sig.compute()
        assert state.action == "halt"
        assert "toxicity" in state.halt_reason

    def test_no_trades_zero_toxicity(self):
        sig = MarketSignals()
        feed_symmetric_book(sig, 100.0, n=10)
        state = sig.compute()
        assert state.toxicity_score == 0.0


# ---------------------------------------------------------------------------
# Consecutive adverse fill tests
# ---------------------------------------------------------------------------

class TestConsecutiveAdverse:

    def test_consecutive_adverse_triggers_halt(self):
        sig = MarketSignals(consecutive_adverse_limit=5)
        feed_symmetric_book(sig, 100.0, n=10)

        # Simulate 5 adverse fills: we bought at 100, mid dropped each time
        for i in range(5):
            sig.on_own_fill("buy", 100.0, 99.5)

        state = sig.compute()
        assert state.action == "halt"
        assert "consecutive_adverse" in state.halt_reason

    def test_non_adverse_fill_resets_streak(self):
        sig = MarketSignals(consecutive_adverse_limit=5)
        feed_symmetric_book(sig, 100.0, n=10)

        # 4 adverse fills
        for _ in range(4):
            sig.on_own_fill("buy", 100.0, 99.5)
        # 1 non-adverse fill (bought at 100, mid rose)
        sig.on_own_fill("buy", 100.0, 100.5)

        state = sig.compute()
        assert state.action != "halt" or "consecutive" not in (state.halt_reason or "")

    def test_sell_adverse(self):
        sig = MarketSignals(consecutive_adverse_limit=3)
        feed_symmetric_book(sig, 100.0, n=10)

        # We sold at 100, mid rose → adverse
        for _ in range(3):
            sig.on_own_fill("sell", 100.0, 100.5)

        state = sig.compute()
        assert state.action == "halt"


# ---------------------------------------------------------------------------
# Volatility regime tests
# ---------------------------------------------------------------------------

class TestVolatility:

    def test_stable_market_normal_regime(self):
        sig = MarketSignals()
        # Feed stable prices for a while
        for i in range(200):
            mid = 100.0 + np.random.normal(0, 0.001)
            sig.on_l2_snapshot(mid - 0.01, 100, mid + 0.01, 100, timestamp=i * 1.0)
        state = sig.compute()
        assert state.vol_regime in ("normal", "low_vol")

    def test_volatile_market_high_vol_regime(self):
        sig = MarketSignals()
        # Feed stable prices first to set slow vol baseline
        for i in range(200):
            mid = 100.0 + np.random.normal(0, 0.001)
            sig.on_l2_snapshot(mid - 0.01, 100, mid + 0.01, 100, timestamp=i * 1.0)

        # Then inject high vol (large moves)
        for i in range(200, 400):
            mid = 100.0 + np.random.normal(0, 0.5)
            sig.on_l2_snapshot(mid - 0.01, 100, mid + 0.01, 100, timestamp=i * 1.0)

        state = sig.compute()
        # Fast vol should be much higher than slow vol
        assert state.sigma_fast > state.sigma_slow

    def test_sigma_values_non_negative(self):
        sig = MarketSignals()
        feed_symmetric_book(sig, 100.0, n=50)
        state = sig.compute()
        assert state.sigma_fast >= 0
        assert state.sigma_slow >= 0


# ---------------------------------------------------------------------------
# Funding rate tests
# ---------------------------------------------------------------------------

class TestFunding:

    def test_funding_rate_passthrough(self):
        sig = MarketSignals(funding_rate=0.0003)
        feed_symmetric_book(sig, 100.0, n=5)
        state = sig.compute()
        assert state.funding_rate == pytest.approx(0.0003)

    def test_update_funding_rate(self):
        sig = MarketSignals()
        feed_symmetric_book(sig, 100.0, n=5)
        sig.update_funding_rate(-0.0005)
        state = sig.compute()
        assert state.funding_rate == pytest.approx(-0.0005)


# ---------------------------------------------------------------------------
# Spread multiplier tests
# ---------------------------------------------------------------------------

class TestSpreadMultiplier:

    def test_normal_toxicity_1x(self):
        sig = MarketSignals()
        feed_symmetric_book(sig, 100.0, n=10)
        # Balanced trades
        for i in range(50):
            sig.on_trade("buy" if i % 2 == 0 else "sell", 100, 1, timestamp=i)
        state = sig.compute()
        assert state.spread_multiplier == 1.0

    def test_moderate_toxicity_1_5x(self):
        sig = MarketSignals()
        feed_symmetric_book(sig, 100.0, n=10)
        # 75 buys, 25 sells → toxicity = 50/100 = 0.50
        for i in range(75):
            sig.on_trade("buy", 100, 1, timestamp=i)
        for i in range(25):
            sig.on_trade("sell", 100, 1, timestamp=75 + i)
        state = sig.compute()
        assert state.spread_multiplier == 1.5

    def test_high_toxicity_2_5x(self):
        sig = MarketSignals()
        feed_symmetric_book(sig, 100.0, n=10)
        # 85 buys, 15 sells → toxicity = 70/100 = 0.70
        for i in range(85):
            sig.on_trade("buy", 100, 1, timestamp=i)
        for i in range(15):
            sig.on_trade("sell", 100, 1, timestamp=85 + i)
        state = sig.compute()
        assert state.spread_multiplier == 2.5


# ---------------------------------------------------------------------------
# Backtest mode tests
# ---------------------------------------------------------------------------

class TestBacktest:

    def test_backtest_returns_timeseries(self):
        sig = MarketSignals()
        snapshots = [
            {"ts": i * 0.5, "best_bid": 99.99, "best_bid_size": 100,
             "best_ask": 100.01, "best_ask_size": 100}
            for i in range(100)
        ]
        trades = [
            {"ts": i * 0.5 + 0.1, "side": "buy" if i % 2 == 0 else "sell",
             "price": 100.0, "size": 1.0}
            for i in range(100)
        ]
        results = sig.backtest(snapshots, trades)
        assert len(results) == len(snapshots)
        assert all(isinstance(r, MarketState) for r in results)

    def test_backtest_timestamps_ordered(self):
        sig = MarketSignals()
        snapshots = [
            {"ts": i, "best_bid": 99.99, "best_bid_size": 100,
             "best_ask": 100.01, "best_ask_size": 100}
            for i in range(20)
        ]
        trades = []
        results = sig.backtest(snapshots, trades)
        timestamps = [r.timestamp for r in results]
        assert timestamps == sorted(timestamps)

    def test_backtest_with_trending_price(self):
        """Price trending up → OFI should eventually detect buy pressure."""
        sig = MarketSignals()
        snapshots = []
        for i in range(100):
            mid = 100.0 + i * 0.01  # trending up
            snapshots.append({
                "ts": i * 0.5,
                "best_bid": mid - 0.01,
                "best_bid_size": 100 + i * i,  # accelerating bids
                "best_ask": mid + 0.01,
                "best_ask_size": 100,
            })
        trades = [
            {"ts": i * 0.5 + 0.1, "side": "buy", "price": 100 + i * 0.01, "size": 1}
            for i in range(100)
        ]
        results = sig.backtest(snapshots, trades)

        # Last few states should show buy pressure (positive OFI) and high toxicity
        final = results[-1]
        assert final.ofi_zscore > 0
        assert final.toxicity_score > 0.8

    def test_backtest_empty_inputs(self):
        sig = MarketSignals()
        results = sig.backtest([], [])
        assert results == []


# ---------------------------------------------------------------------------
# MarketState dataclass tests
# ---------------------------------------------------------------------------

class TestMarketState:

    def test_default_halt_reason_none(self):
        state = MarketState(
            timestamp=0, mid_price=100, ofi_zscore=0, ofi_reservation_adj=0,
            toxicity_score=0, vol_regime="normal", sigma_fast=0, sigma_slow=0,
            funding_rate=0, spread_multiplier=1.0, action="quote",
        )
        assert state.halt_reason is None

    def test_mid_price_set(self):
        sig = MarketSignals()
        sig.on_l2_snapshot(99.5, 100, 100.5, 100, timestamp=0)
        sig.on_l2_snapshot(99.5, 100, 100.5, 100, timestamp=1)
        state = sig.compute()
        assert state.mid_price == pytest.approx(100.0, abs=0.01)
