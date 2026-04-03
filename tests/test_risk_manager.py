"""
Tests for RiskManager circuit breakers.

Covers:
  - Inventory breach → HALTED_INVENTORY
  - Drawdown halt → HALTED_DRAWDOWN (manual restart)
  - Latency halt and reduced mode
  - Toxicity halt with cooldown
  - Dead man's switch
  - Self-trade prevention
  - Position sizing guard
  - State transitions
"""

import pytest
import time
from MM_algo.core.risk_manager import (
    RiskManager, RiskConfig, OrderProposal, RiskVerdict, BotState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_rm(**kwargs) -> RiskManager:
    cfg = RiskConfig(**kwargs)
    rm = RiskManager(cfg)
    rm.set_session_equity(100_000)
    rm.heartbeat()
    return rm


def make_proposal(side="buy", price=100.0, size=10.0, **kw) -> OrderProposal:
    return OrderProposal(
        side=side, price=price, size=size,
        notional=price * size, **kw,
    )


# ---------------------------------------------------------------------------
# Inventory breach tests
# ---------------------------------------------------------------------------

class TestInventoryBreach:

    def test_order_allowed_within_limits(self):
        rm = make_rm(q_max_usd=50_000)
        v = rm.check_order(make_proposal(), 100_000, inventory_q=10_000)
        assert v.allowed is True

    def test_inventory_breach_halts(self):
        rm = make_rm(q_max_usd=50_000)
        v = rm.check_order(make_proposal(), 100_000, inventory_q=55_000)
        assert v.allowed is False
        assert rm.state == BotState.HALTED_INVENTORY
        assert "cancel_all" in v.actions_required
        assert "flatten_inventory" in v.actions_required

    def test_emergency_flatten_allowed_during_halt(self):
        rm = make_rm(q_max_usd=50_000)
        # Trigger halt
        rm.check_order(make_proposal(), 100_000, inventory_q=55_000)
        assert rm.state == BotState.HALTED_INVENTORY

        # Emergency flatten should be allowed
        flatten = make_proposal(
            side="sell", price=100.0, size=550.0,
            is_emergency_flatten=True,
        )
        v = rm.check_order(flatten, 100_000, inventory_q=55_000)
        assert v.allowed is True

    def test_inventory_halt_recovery_after_cooldown(self):
        rm = make_rm(q_max_usd=50_000, inventory_cooldown_s=5.0)
        ts = time.time()

        # Trigger halt
        rm.check_order(make_proposal(), 100_000, inventory_q=55_000, timestamp=ts)
        assert rm.state == BotState.HALTED_INVENTORY

        # Mark as flattened
        rm.mark_inventory_flattened()

        # Before cooldown — still halted
        v = rm.check_order(make_proposal(), 100_000, inventory_q=0, timestamp=ts + 3)
        assert v.allowed is False

        # After cooldown — recovered
        v = rm.check_order(make_proposal(), 100_000, inventory_q=0, timestamp=ts + 6)
        assert v.allowed is True
        assert rm.state == BotState.RUNNING


# ---------------------------------------------------------------------------
# Drawdown halt tests
# ---------------------------------------------------------------------------

class TestDrawdownHalt:

    def test_no_halt_above_threshold(self):
        rm = make_rm(max_drawdown_pct=3.0)
        v = rm.check_order(make_proposal(), current_equity=98_000, inventory_q=0)
        assert v.allowed is True

    def test_halt_on_drawdown_breach(self):
        rm = make_rm(max_drawdown_pct=3.0)
        # 3% of 100k = 3k, threshold = 97k
        v = rm.check_order(make_proposal(), current_equity=96_000, inventory_q=0)
        assert v.allowed is False
        assert rm.state == BotState.HALTED_DRAWDOWN
        assert "cancel_all" in v.actions_required

    def test_drawdown_halt_never_auto_recovers(self):
        rm = make_rm(max_drawdown_pct=3.0)
        rm.check_order(make_proposal(), current_equity=96_000, inventory_q=0)
        assert rm.state == BotState.HALTED_DRAWDOWN

        # Even with full equity restored, can't auto-recover
        v = rm.check_order(make_proposal(), current_equity=200_000, inventory_q=0)
        assert v.allowed is False
        assert rm.state == BotState.HALTED_DRAWDOWN

    def test_drawdown_cannot_reset(self):
        rm = make_rm(max_drawdown_pct=3.0)
        rm.check_order(make_proposal(), current_equity=96_000, inventory_q=0)
        assert rm.reset_from_halt() is False


# ---------------------------------------------------------------------------
# Latency tests
# ---------------------------------------------------------------------------

class TestLatency:

    def test_normal_latency_ok(self):
        rm = make_rm(latency_warn_ms=200, latency_halt_ms=1000)
        for _ in range(20):
            rm.record_rtt(50.0)
        v = rm.check_order(make_proposal(), 100_000, inventory_q=0)
        assert v.allowed is True
        assert rm.state == BotState.RUNNING

    def test_elevated_latency_reduced_mode(self):
        rm = make_rm(latency_warn_ms=200, latency_halt_ms=1000)
        for _ in range(20):
            rm.record_rtt(250.0)
        rm.check_order(make_proposal(), 100_000, inventory_q=0)
        assert rm.state == BotState.REDUCED_MODE

    def test_extreme_latency_halts(self):
        rm = make_rm(latency_halt_ms=1000)
        for _ in range(20):
            rm.record_rtt(1200.0)
        v = rm.check_order(make_proposal(), 100_000, inventory_q=0)
        assert v.allowed is False

    def test_single_extreme_rtt_halts(self):
        rm = make_rm(latency_single_max_ms=3000)
        for _ in range(19):
            rm.record_rtt(50.0)
        rm.record_rtt(3500.0)  # one extreme spike
        v = rm.check_order(make_proposal(), 100_000, inventory_q=0)
        assert v.allowed is False

    def test_reduced_mode_spreads(self):
        rm = make_rm(latency_warn_ms=200)
        for _ in range(20):
            rm.record_rtt(250.0)
        rm.check_order(make_proposal(), 100_000, inventory_q=0)
        assert rm.get_spread_multiplier() == 3.0
        assert rm.get_requote_interval_s() == 5.0

    def test_latency_recovery_to_running(self):
        rm = make_rm(latency_warn_ms=200, rtt_recovery_s=5.0)
        ts = time.time()

        # Enter reduced mode
        for _ in range(20):
            rm.record_rtt(250.0)
        rm.heartbeat(timestamp=ts)
        rm.check_order(make_proposal(), 100_000, inventory_q=0, timestamp=ts)
        assert rm.state == BotState.REDUCED_MODE

        # Good latency — first call starts recovery timer
        for _ in range(20):
            rm.record_rtt(50.0)
        rm.heartbeat(timestamp=ts + 50)
        rm.check_order(make_proposal(), 100_000, inventory_q=0, timestamp=ts + 1)

        # Second call after recovery period → should transition to RUNNING
        rm.check_order(make_proposal(), 100_000, inventory_q=0, timestamp=ts + 7)
        assert rm.state == BotState.RUNNING


# ---------------------------------------------------------------------------
# Toxicity halt tests
# ---------------------------------------------------------------------------

class TestToxicityHalt:

    def test_toxicity_halt(self):
        rm = make_rm(toxicity_cooldown_s=60)
        v = rm.check_order(
            make_proposal(), 100_000, inventory_q=0,
            market_state_action="halt",
        )
        assert v.allowed is False
        assert rm.state == BotState.HALTED_TOXICITY
        assert "cancel_all" in v.actions_required

    def test_toxicity_does_not_flatten(self):
        rm = make_rm()
        v = rm.check_order(
            make_proposal(), 100_000, inventory_q=0,
            market_state_action="halt",
        )
        assert "flatten_inventory" not in v.actions_required

    def test_toxicity_recovers_after_cooldown(self):
        rm = make_rm(toxicity_cooldown_s=10)
        ts = time.time()

        rm.check_order(
            make_proposal(), 100_000, inventory_q=0,
            market_state_action="halt", timestamp=ts,
        )
        assert rm.state == BotState.HALTED_TOXICITY

        # Before cooldown
        v = rm.check_order(
            make_proposal(), 100_000, inventory_q=0,
            market_state_action="quote", timestamp=ts + 5,
        )
        assert v.allowed is False

        # After cooldown + market normalised
        v = rm.check_order(
            make_proposal(), 100_000, inventory_q=0,
            market_state_action="quote", timestamp=ts + 11,
        )
        assert v.allowed is True
        assert rm.state == BotState.RUNNING


# ---------------------------------------------------------------------------
# Dead man's switch
# ---------------------------------------------------------------------------

class TestDeadManSwitch:

    def test_heartbeat_extends_deadline(self):
        rm = make_rm(dead_man_switch_s=10)
        ts = time.time()
        rm.heartbeat(timestamp=ts)
        v = rm.check_order(make_proposal(), 100_000, inventory_q=0, timestamp=ts + 5)
        assert v.allowed is True

    def test_missed_heartbeat_halts(self):
        rm = make_rm(dead_man_switch_s=10)
        ts = time.time()
        rm.heartbeat(timestamp=ts)
        # Check way past deadline
        v = rm.check_order(make_proposal(), 100_000, inventory_q=0, timestamp=ts + 15)
        assert v.allowed is False
        assert "cancel_all" in v.actions_required


# ---------------------------------------------------------------------------
# Self-trade prevention
# ---------------------------------------------------------------------------

class TestSelfTradePrevention:

    def test_bid_crossing_own_ask_rejected(self):
        rm = make_rm()
        rm.register_order("ask_1", "sell", 100.0)
        proposal = make_proposal(side="buy", price=100.5)  # above our ask
        v = rm.check_order(proposal, 100_000, inventory_q=0)
        assert v.allowed is False
        assert "Self-trade" in v.reason

    def test_ask_crossing_own_bid_rejected(self):
        rm = make_rm()
        rm.register_order("bid_1", "buy", 100.0)
        proposal = make_proposal(side="sell", price=99.5)  # below our bid
        v = rm.check_order(proposal, 100_000, inventory_q=0)
        assert v.allowed is False

    def test_non_crossing_allowed(self):
        rm = make_rm()
        rm.register_order("ask_1", "sell", 101.0)
        rm.register_order("bid_1", "buy", 99.0)
        # Bid below our ask, ask above our bid
        v = rm.check_order(make_proposal(side="buy", price=100.0), 100_000, 0)
        assert v.allowed is True

    def test_unregister_clears_tracking(self):
        rm = make_rm()
        rm.register_order("ask_1", "sell", 100.0)
        rm.unregister_order("ask_1")
        proposal = make_proposal(side="buy", price=100.5)
        v = rm.check_order(proposal, 100_000, inventory_q=0)
        assert v.allowed is True


# ---------------------------------------------------------------------------
# Position sizing guard
# ---------------------------------------------------------------------------

class TestPositionSizing:

    def test_size_reduced_near_limit(self):
        rm = make_rm(q_max_usd=50_000)
        # Inventory at 30k, proposing to buy $20k more → total 50k > 90% of 50k (45k)
        # target_q = 40k, max_notional = 40000 - 30000 = $10k
        # Original notional = $20k, so size should be halved
        proposal = make_proposal(side="buy", price=100.0, size=200.0)  # $20k
        v = rm.check_order(proposal, 100_000, inventory_q=30_000)
        assert v.allowed is True
        assert v.adjusted_size is not None
        assert v.adjusted_size < proposal.size

    def test_size_rejected_if_too_small(self):
        rm = make_rm(q_max_usd=50_000, min_order_notional=10.0)
        # Inventory at 49.5k, proposing buy → almost no room
        proposal = make_proposal(side="buy", price=100.0, size=1000.0)
        v = rm.check_order(proposal, 100_000, inventory_q=49_500)
        # After reduction, notional too small → rejected
        # target = 40000, max_notional = 40000 - 49500 = negative → 0
        assert v.allowed is False

    def test_emergency_flatten_bypasses_sizing(self):
        rm = make_rm(q_max_usd=50_000)
        proposal = make_proposal(
            side="sell", price=100.0, size=1000.0,
            is_emergency_flatten=True,
        )
        v = rm.check_order(proposal, 100_000, inventory_q=60_000)
        # Emergency flatten always allowed (even during halt)
        # First it hits inventory breach, but flatten is allowed
        assert v.allowed is True


# ---------------------------------------------------------------------------
# State and operational tests
# ---------------------------------------------------------------------------

class TestState:

    def test_initial_state_running(self):
        rm = make_rm()
        assert rm.state == BotState.RUNNING

    def test_force_halt(self):
        rm = make_rm()
        rm.force_halt("test")
        assert rm.state == BotState.HALTED_MANUAL

    def test_manual_halt_blocks_orders(self):
        rm = make_rm()
        rm.force_halt("test")
        v = rm.check_order(make_proposal(), 100_000, inventory_q=0)
        assert v.allowed is False

    def test_reset_from_manual_halt(self):
        rm = make_rm()
        rm.force_halt("test")
        assert rm.reset_from_halt() is True
        assert rm.state == BotState.RUNNING

    def test_status_report(self):
        rm = make_rm()
        rm.record_rtt(50.0)
        rm.register_order("b1", "buy", 100.0)
        status = rm.status()
        assert status["state"] == "RUNNING"
        assert status["mean_rtt_ms"] == 50.0
        assert status["open_bids"] == 1

    def test_ttl_by_vol_regime(self):
        rm = make_rm()
        assert rm.get_ttl_ms("high_vol") == 200.0
        assert rm.get_ttl_ms("normal") == 500.0
        assert rm.get_ttl_ms("low_vol") == 2000.0
