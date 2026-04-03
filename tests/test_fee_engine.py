"""
Comprehensive tests for FeeEngine.

Covers every fee scenario:
  - All volume tiers (T0–T6)
  - Staking discounts (none through diamond)
  - HIP-3 doubling
  - Maker-share rebates (negative fees = you earn)
  - Builder fee injection
  - Round-trip cost
  - Min viable spread
  - Viability check
  - Tier updates
"""

import pytest
from MM_algo.core.fee_engine import (
    FeeEngine, FeeConfig, FeeBreakdown,
    VOLUME_TIERS, STAKING_DISCOUNTS, MM_REBATES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_engine(**kwargs) -> FeeEngine:
    return FeeEngine(FeeConfig(**kwargs))


# ---------------------------------------------------------------------------
# Volume tier rate tests
# ---------------------------------------------------------------------------

class TestVolumeTiers:
    """Verify base rates for each volume tier."""

    @pytest.mark.parametrize("tier,exp_maker,exp_taker", [
        (0, 0.000_15, 0.000_45),
        (1, 0.000_12, 0.000_40),
        (2, 0.000_08, 0.000_35),
        (3, 0.000_04, 0.000_30),
        (4, 0.000_00, 0.000_28),
        (5, 0.000_00, 0.000_26),
        (6, 0.000_00, 0.000_24),
    ])
    def test_base_rates_per_tier(self, tier, exp_maker, exp_taker):
        engine = make_engine(volume_tier=tier)
        assert engine.get_maker_rate() == pytest.approx(exp_maker, abs=1e-9)
        assert engine.get_taker_rate() == pytest.approx(exp_taker, abs=1e-9)

    def test_tier4_and_above_maker_is_zero(self):
        for tier in [4, 5, 6]:
            engine = make_engine(volume_tier=tier)
            assert engine.get_maker_rate() == pytest.approx(0.0, abs=1e-9)

    def test_update_volume_tier_from_volume(self):
        engine = make_engine(volume_tier=0)
        assert engine.get_maker_rate() == pytest.approx(0.000_15, abs=1e-9)

        engine.update_volume_tier(30_000_000)  # T2
        assert engine.get_maker_rate() == pytest.approx(0.000_08, abs=1e-9)

        engine.update_volume_tier(600_000_000)  # T4
        assert engine.get_maker_rate() == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Staking discount tests
# ---------------------------------------------------------------------------

class TestStakingDiscounts:

    @pytest.mark.parametrize("staking,discount", [
        ("none", 0.00),
        ("wood", 0.05),
        ("bronze", 0.10),
        ("silver", 0.15),
        ("gold", 0.20),
        ("platinum", 0.30),
        ("diamond", 0.40),
    ])
    def test_staking_reduces_maker_rate(self, staking, discount):
        engine = make_engine(volume_tier=0, staking_tier=staking)
        base = 0.000_15
        expected = base * (1.0 - discount)
        assert engine.get_maker_rate() == pytest.approx(expected, abs=1e-9)

    def test_diamond_staking_on_t0(self):
        # T0 maker = 1.5bps, diamond = 40% off → 0.9bps
        engine = make_engine(volume_tier=0, staking_tier="diamond")
        assert engine.get_maker_rate() == pytest.approx(0.000_09, abs=1e-9)

    def test_staking_applies_to_taker_too(self):
        engine = make_engine(volume_tier=0, staking_tier="gold")
        # T0 taker = 4.5bps, gold = 20% off → 3.6bps
        assert engine.get_taker_rate() == pytest.approx(0.000_36, abs=1e-9)


# ---------------------------------------------------------------------------
# HIP-3 doubling tests
# ---------------------------------------------------------------------------

class TestHIP3:

    def test_hip3_doubles_protocol_fee(self):
        engine = make_engine(volume_tier=0)
        standard = engine.compute_fee(100_000, "maker", "standard")
        hip3 = engine.compute_fee(100_000, "maker", "hip3")
        # HIP-3 protocol fee should be 2× standard
        assert hip3.protocol_fee == pytest.approx(standard.protocol_fee * 2, abs=0.01)

    def test_hip3_with_staking(self):
        engine = make_engine(volume_tier=0, staking_tier="gold")
        hip3 = engine.compute_fee(100_000, "maker", "hip3")
        # base=1.5bps × 2 (hip3) = 3.0bps, gold = 20% off → 2.4bps on $100k = $24.00
        assert hip3.protocol_fee == pytest.approx(24.00, abs=0.01)

    def test_hip3_taker_doubles(self):
        engine = make_engine(volume_tier=0)
        std_taker = engine.compute_fee(100_000, "taker", "standard")
        hip3_taker = engine.compute_fee(100_000, "taker", "hip3")
        assert hip3_taker.protocol_fee == pytest.approx(std_taker.protocol_fee * 2, abs=0.01)


# ---------------------------------------------------------------------------
# Builder fee tests
# ---------------------------------------------------------------------------

class TestBuilderFee:

    def test_zero_builder_fee(self):
        engine = make_engine(builder_fee_bps=0)
        bd = engine.compute_fee(100_000, "maker")
        assert bd.builder_fee == pytest.approx(0.0, abs=1e-9)

    def test_5bps_builder_fee(self):
        engine = make_engine(builder_fee_bps=5)
        bd = engine.compute_fee(100_000, "maker")
        # 5bps on $100k = $50
        assert bd.builder_fee == pytest.approx(50.0, abs=0.01)

    def test_builder_fee_not_affected_by_staking(self):
        engine = make_engine(builder_fee_bps=5, staking_tier="diamond")
        bd = engine.compute_fee(100_000, "maker")
        # Builder fee is always $50 regardless of staking
        assert bd.builder_fee == pytest.approx(50.0, abs=0.01)

    def test_builder_fee_not_affected_by_hip3(self):
        engine = make_engine(builder_fee_bps=5)
        bd_std = engine.compute_fee(100_000, "maker", "standard")
        bd_hip = engine.compute_fee(100_000, "maker", "hip3")
        assert bd_std.builder_fee == bd_hip.builder_fee


# ---------------------------------------------------------------------------
# Maker-share rebate tests
# ---------------------------------------------------------------------------

class TestMakerRebates:

    def test_no_rebate_below_threshold(self):
        engine = make_engine(maker_volume_share_pct=0.004)  # < 0.5%
        bd = engine.compute_fee(100_000, "maker")
        assert bd.maker_rebate == pytest.approx(0.0, abs=1e-9)

    def test_tier1_rebate(self):
        engine = make_engine(volume_tier=0, maker_volume_share_pct=0.006)
        bd = engine.compute_fee(100_000, "maker")
        # rebate = -0.1bps × $100k = -$1.00
        assert bd.maker_rebate == pytest.approx(-1.00, abs=0.01)

    def test_tier2_rebate(self):
        engine = make_engine(volume_tier=0, maker_volume_share_pct=0.02)
        bd = engine.compute_fee(100_000, "maker")
        # rebate = -0.2bps × $100k = -$2.00
        assert bd.maker_rebate == pytest.approx(-2.00, abs=0.01)

    def test_tier3_rebate(self):
        engine = make_engine(volume_tier=0, maker_volume_share_pct=0.04)
        bd = engine.compute_fee(100_000, "maker")
        # rebate = -0.3bps × $100k = -$3.00
        assert bd.maker_rebate == pytest.approx(-3.00, abs=0.01)

    def test_rebate_can_make_net_fee_negative(self):
        """At T4+ (0 maker fee) + rebate, the MM actually EARNS on every fill."""
        engine = make_engine(
            volume_tier=4,
            maker_volume_share_pct=0.04,
            builder_fee_bps=0,
        )
        bd = engine.compute_fee(100_000, "maker")
        assert bd.net_fee < 0, f"Expected negative net_fee (earning), got {bd.net_fee}"

    def test_rebate_only_applies_to_maker(self):
        engine = make_engine(volume_tier=0, maker_volume_share_pct=0.04)
        taker_bd = engine.compute_fee(100_000, "taker")
        assert taker_bd.maker_rebate == pytest.approx(0.0, abs=1e-9)

    def test_update_maker_share(self):
        engine = make_engine(volume_tier=4, maker_volume_share_pct=0.0)
        assert engine.get_maker_rate() == pytest.approx(0.0, abs=1e-9)
        engine.update_maker_share(0.04)
        # Now has -0.03bps rebate
        assert engine.get_maker_rate() == pytest.approx(-0.000_03, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_fee integration tests
# ---------------------------------------------------------------------------

class TestComputeFee:

    def test_standard_maker_t0_no_extras(self):
        engine = make_engine(volume_tier=0)
        bd = engine.compute_fee(100_000, "maker", "standard")
        # 1.5bps on $100k = $15
        assert bd.protocol_fee == pytest.approx(15.0, abs=0.01)
        assert bd.net_fee == pytest.approx(15.0, abs=0.01)
        assert bd.net_fee_bps == pytest.approx(1.5, abs=0.01)

    def test_net_fee_bps_field(self):
        engine = make_engine(volume_tier=0, builder_fee_bps=5)
        bd = engine.compute_fee(100_000, "maker")
        # 1.5 + 5.0 = 6.5 bps
        assert bd.net_fee_bps == pytest.approx(6.5, abs=0.1)

    def test_zero_notional_no_crash(self):
        engine = make_engine()
        bd = engine.compute_fee(0, "maker")
        assert bd.net_fee == pytest.approx(0.0, abs=1e-9)

    def test_invalid_side_raises(self):
        engine = make_engine()
        with pytest.raises(ValueError, match="side must be"):
            engine.compute_fee(100_000, "invalid_side")

    def test_to_dict(self):
        engine = make_engine()
        bd = engine.compute_fee(100_000, "maker")
        d = bd.to_dict()
        assert "net_fee" in d
        assert "net_fee_bps" in d
        assert d["side"] == "maker"


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

class TestRoundTrip:

    def test_round_trip_standard_t0(self):
        engine = make_engine(volume_tier=0)
        rt = engine.compute_round_trip(100_000, "standard")
        # 2 × $15 = $30
        assert rt == pytest.approx(30.0, abs=0.01)

    def test_round_trip_hip3_t0(self):
        engine = make_engine(volume_tier=0)
        rt = engine.compute_round_trip(100_000, "hip3")
        # 2 × $30 = $60 (HIP-3 doubles)
        assert rt == pytest.approx(60.0, abs=0.01)

    def test_round_trip_t4_with_rebate(self):
        engine = make_engine(volume_tier=4, maker_volume_share_pct=0.04, builder_fee_bps=0)
        rt = engine.compute_round_trip(100_000)
        # T4 maker=0, rebate=-0.03bps → each leg earns $0.30, RT = -$0.60
        assert rt < 0, f"Expected negative RT (earning), got {rt}"


# ---------------------------------------------------------------------------
# Min viable spread tests
# ---------------------------------------------------------------------------

class TestMinViableSpread:

    def test_basic_min_spread(self):
        engine = make_engine(volume_tier=0, builder_fee_bps=0)
        # maker_rate=1.5bps, builder=0, adverse=20%
        # 2 × 1.5 / 0.8 = 3.75 bps
        spread = engine.min_viable_spread_bps(100_000, adverse_sel_pct=0.20)
        assert spread == pytest.approx(3.75, abs=0.1)

    def test_min_spread_with_builder(self):
        engine = make_engine(volume_tier=0, builder_fee_bps=2)
        # 2 × (1.5 + 2.0) / 0.8 = 8.75 bps
        spread = engine.min_viable_spread_bps(100_000, adverse_sel_pct=0.20)
        assert spread == pytest.approx(8.75, abs=0.1)

    def test_min_spread_zero_adverse_selection(self):
        engine = make_engine(volume_tier=0, builder_fee_bps=0)
        spread = engine.min_viable_spread_bps(100_000, adverse_sel_pct=0.0)
        # 2 × 1.5 / 1.0 = 3.0 bps
        assert spread == pytest.approx(3.0, abs=0.1)


# ---------------------------------------------------------------------------
# Viability tests
# ---------------------------------------------------------------------------

class TestViability:

    def test_viable_when_spread_above_min(self):
        engine = make_engine(volume_tier=0)
        assert engine.is_viable(5.0, 100_000) is True

    def test_not_viable_when_spread_below_min(self):
        engine = make_engine(volume_tier=0)
        # min ~3.75 bps (T0, 0 builder, 20% adverse)
        assert engine.is_viable(1.0, 100_000) is False

    def test_viable_at_t4_with_rebate(self):
        engine = make_engine(volume_tier=4, maker_volume_share_pct=0.04, builder_fee_bps=0)
        # negative effective rate → any positive spread is viable
        assert engine.is_viable(0.1, 100_000) is True


# ---------------------------------------------------------------------------
# Fee report
# ---------------------------------------------------------------------------

class TestFeeReport:

    def test_fee_report_returns_string(self):
        engine = make_engine(volume_tier=2, staking_tier="gold")
        report = engine.fee_report()
        assert "T2" in report
        assert "gold" in report
        assert "Effective maker" in report

    def test_log_fee_breakdown(self):
        engine = make_engine()
        bd = engine.compute_fee(100_000, "maker")
        d = engine.log_fee_breakdown(bd)
        assert isinstance(d, dict)
        assert "net_fee" in d


# ---------------------------------------------------------------------------
# Edge cases and combined scenarios
# ---------------------------------------------------------------------------

class TestCombinedScenarios:

    def test_full_stack_hip3_diamond_rebate_builder(self):
        """
        HIP-3 market, diamond staking, tier-3 rebate, 2bps builder.
        Most complex scenario — all components active.
        """
        engine = make_engine(
            volume_tier=0,
            staking_tier="diamond",
            builder_fee_bps=2,
            maker_volume_share_pct=0.04,
        )
        bd = engine.compute_fee(100_000, "maker", "hip3")

        # base maker = 1.5bps
        # hip3 = ×2 = 3.0bps
        # diamond = 40% off → 3.0 × 0.6 = 1.8bps → $18.00
        assert bd.protocol_fee == pytest.approx(18.00, abs=0.02)

        # builder = 2bps → $20.00
        assert bd.builder_fee == pytest.approx(20.00, abs=0.01)

        # rebate = -0.3bps → -$3.00
        assert bd.maker_rebate == pytest.approx(-3.00, abs=0.01)

        # net = 18.00 + 20.00 - 3.00 = $35.00
        assert bd.net_fee == pytest.approx(35.00, abs=0.05)

    def test_tier6_diamond_rebate_zero_builder(self):
        """Best possible scenario: T6 + diamond + tier-3 rebate + 0 builder."""
        engine = make_engine(
            volume_tier=6,
            staking_tier="diamond",
            builder_fee_bps=0,
            maker_volume_share_pct=0.04,
        )
        bd = engine.compute_fee(1_000_000, "maker", "standard")
        # T6 maker = 0, diamond discount = irrelevant on 0
        # rebate = -0.3bps × $1M = -$30.00
        assert bd.net_fee == pytest.approx(-30.00, abs=0.01)
        assert bd.net_fee < 0  # We EARN money
