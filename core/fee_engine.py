"""
FeeEngine — computes the FULL fee stack for any trade on Hyperliquid (or any DEX).

This is the foundation module: no trade should ever be placed without
confirming it is fee-viable through this engine.

Fee stack (Hyperliquid perps):
  protocol_fee  = base_rate × hip3_multiplier × (1 - staking_discount)
  builder_fee   = builder_fee_bps / 10_000 × notional
  maker_rebate  = negative if MM rebate tier reached (separate from volume tier)
  net_fee       = protocol_fee + builder_fee + maker_rebate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fee tables — single source of truth, mirrors Hyperliquid docs
# ---------------------------------------------------------------------------

VOLUME_TIERS: list[dict[str, float]] = [
    {"min": 0,             "taker": 0.000_45, "maker": 0.000_15},   # T0
    {"min": 5_000_000,     "taker": 0.000_40, "maker": 0.000_12},   # T1
    {"min": 25_000_000,    "taker": 0.000_35, "maker": 0.000_08},   # T2
    {"min": 100_000_000,   "taker": 0.000_30, "maker": 0.000_04},   # T3
    {"min": 500_000_000,   "taker": 0.000_28, "maker": 0.000_00},   # T4
    {"min": 2_000_000_000, "taker": 0.000_26, "maker": 0.000_00},   # T5
    {"min": 7_000_000_000, "taker": 0.000_24, "maker": 0.000_00},   # T6
]

STAKING_DISCOUNTS: dict[str, float] = {
    "none":     0.00,
    "wood":     0.05,
    "bronze":   0.10,
    "silver":   0.15,
    "gold":     0.20,
    "platinum": 0.30,
    "diamond":  0.40,
}

# Maker-share rebates — separate from volume tier, requires platform share
MM_REBATES: list[dict[str, float]] = [
    {"min_share": 0.005, "rebate": -0.000_01},  # >0.5%  → -0.1 bps
    {"min_share": 0.015, "rebate": -0.000_02},  # >1.5%  → -0.2 bps
    {"min_share": 0.030, "rebate": -0.000_03},  # >3.0%  → -0.3 bps
]

# Market type multipliers applied to base protocol rate
MARKET_MULTIPLIERS: dict[str, float] = {
    "standard": 1.0,
    "hip3":     2.0,    # HIP-3 markets double the base rate
    "spot_qq":  0.2,    # spot between two quote assets: 80% lower
}

# ---------------------------------------------------------------------------
# Pacifica fee tables
# ---------------------------------------------------------------------------

PACIFICA_TIERS: list[dict[str, float]] = [
    {"min": 0,             "taker": 0.000_40, "maker": 0.000_15},  # Tier 1
    {"min": 5_000_000,     "taker": 0.000_38, "maker": 0.000_12},  # Tier 2
    {"min": 10_000_000,    "taker": 0.000_36, "maker": 0.000_09},  # Tier 3
    {"min": 25_000_000,    "taker": 0.000_34, "maker": 0.000_06},  # Tier 4
    {"min": 50_000_000,    "taker": 0.000_32, "maker": 0.000_03},  # Tier 5
    {"min": 100_000_000,   "taker": 0.000_30, "maker": 0.000_00},  # VIP 1
    {"min": 250_000_000,   "taker": 0.000_29, "maker": 0.000_00},  # VIP 2
    {"min": 500_000_000,   "taker": 0.000_28, "maker": 0.000_00},  # VIP 3
]

# RWA markets get 50% fee discount
PACIFICA_RWA_SYMBOLS: set[str] = {
    "PLATINUM", "URNM", "GOLD", "SILVER", "PAXG", "CL", "COPPER",
    "NATGAS", "EURUSD", "USDJPY", "NVDA", "TSLA", "PLTR",
    "SP500", "GOOGL", "CRCL", "HOOD",
}
PACIFICA_RWA_DISCOUNT: float = 0.5  # 50% off all tiers


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeeConfig:
    """Configuration driving the fee engine — loaded from YAML."""
    volume_tier: int = 0
    staking_tier: str = "none"
    builder_fee_bps: float = 0.0
    maker_volume_share_pct: float = 0.0

    # Exchange selection: "hyperliquid" (default) or "pacifica"
    exchange: str = "hyperliquid"
    # Pacifica-specific: is this an RWA market? (50% discount)
    is_rwa_market: bool = False
    # Pacifica-specific: symbol for RWA auto-detection
    symbol: str = ""

    # Overrides (for exchanges with different schedules)
    custom_maker_rate: float | None = None
    custom_taker_rate: float | None = None

    @classmethod
    def from_yaml(cls, path: str) -> FeeConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        fees = raw.get("fees", {})
        return cls(
            volume_tier=fees.get("volume_tier", 0),
            staking_tier=fees.get("staking_tier", "none"),
            builder_fee_bps=fees.get("builder_fee_bps", 0.0),
            maker_volume_share_pct=fees.get("maker_volume_share_pct", 0.0),
        )


@dataclass
class FeeBreakdown:
    """Itemised breakdown of fees for ONE leg of a trade."""
    notional: float
    side: str                    # "maker" or "taker"
    market_type: str
    protocol_fee: float          # base rate × notional (after staking, before rebate)
    hip3_fee: float              # additional fee from HIP-3 doubling (0 if standard)
    builder_fee: float           # builder code fee
    staking_discount: float      # negative number = savings
    maker_rebate: float          # negative if MM rebate tier hit
    net_fee: float               # sum of all components
    net_fee_bps: float           # net_fee / notional × 10_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "notional": self.notional,
            "side": self.side,
            "market_type": self.market_type,
            "protocol_fee": round(self.protocol_fee, 6),
            "hip3_fee": round(self.hip3_fee, 6),
            "builder_fee": round(self.builder_fee, 6),
            "staking_discount": round(self.staking_discount, 6),
            "maker_rebate": round(self.maker_rebate, 6),
            "net_fee": round(self.net_fee, 6),
            "net_fee_bps": round(self.net_fee_bps, 4),
        }


# ---------------------------------------------------------------------------
# FeeEngine
# ---------------------------------------------------------------------------

class FeeEngine:
    """
    Computes the full fee stack for any trade on Hyperliquid (or any DEX
    with a maker/taker model).

    Usage:
        cfg = FeeConfig(volume_tier=0, staking_tier="gold", builder_fee_bps=0)
        engine = FeeEngine(cfg)
        breakdown = engine.compute_fee(100_000, side="maker", market_type="standard")
        print(breakdown.net_fee)
    """

    def __init__(self, config: FeeConfig) -> None:
        self._cfg = config
        # Resolve rates at init; re-resolve when tiers are updated
        self._resolve_rates()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_maker_rate(self) -> float:
        """Effective maker rate after tier + staking + rebate (as a fraction, not bps)."""
        return self._effective_maker_rate

    def get_taker_rate(self) -> float:
        """Effective taker rate after tier + staking (as a fraction)."""
        return self._effective_taker_rate

    def compute_fee(
        self,
        notional: float,
        side: str = "maker",
        market_type: str = "standard",
    ) -> FeeBreakdown:
        """
        Returns an itemised FeeBreakdown for one leg of a trade.

        Args:
            notional:    trade size in USD
            side:        "maker" or "taker"
            market_type: "standard", "hip3", or "spot_qq"
        """
        if side not in ("maker", "taker"):
            raise ValueError(f"side must be 'maker' or 'taker', got '{side}'")

        multiplier = MARKET_MULTIPLIERS.get(market_type, 1.0)
        base_rate = self._base_maker if side == "maker" else self._base_taker

        # Protocol fee before any adjustment
        raw_protocol = base_rate * notional

        # HIP-3 additional fee: the extra amount from doubling
        # For standard markets this is 0
        hip3_extra = raw_protocol * (multiplier - 1.0)

        # Total protocol fee before staking discount
        gross_protocol = raw_protocol * multiplier

        # Staking discount applies to protocol fee only
        staking_rate = STAKING_DISCOUNTS.get(self._cfg.staking_tier, 0.0)
        staking_savings = -gross_protocol * staking_rate
        protocol_after_staking = gross_protocol + staking_savings

        # Builder fee — always additive, not affected by staking or HIP-3
        builder_fee = (self._cfg.builder_fee_bps / 10_000) * notional

        # Maker-share rebate — only for maker side on Hyperliquid
        rebate = 0.0
        if side == "maker" and self._cfg.exchange == "hyperliquid":
            rebate_rate = self._get_rebate_rate()
            rebate = rebate_rate * notional  # rebate_rate is negative

        net = protocol_after_staking + builder_fee + rebate
        net_bps = (net / notional * 10_000) if notional > 0 else 0.0

        return FeeBreakdown(
            notional=notional,
            side=side,
            market_type=market_type,
            protocol_fee=protocol_after_staking,
            hip3_fee=hip3_extra * (1.0 - staking_rate),  # staking applies to the doubled portion too
            builder_fee=builder_fee,
            staking_discount=staking_savings,
            maker_rebate=rebate,
            net_fee=net,
            net_fee_bps=net_bps,
        )

    def compute_round_trip(
        self,
        notional: float,
        market_type: str = "standard",
    ) -> float:
        """Total cost in USD to open + close a position as maker on both legs."""
        open_leg = self.compute_fee(notional, side="maker", market_type=market_type)
        close_leg = self.compute_fee(notional, side="maker", market_type=market_type)
        return open_leg.net_fee + close_leg.net_fee

    def min_viable_spread_bps(
        self,
        notional: float,
        adverse_sel_pct: float = 0.20,
    ) -> float:
        """
        Minimum half-spread (in bps) needed to be cash-flow positive
        after fees and expected adverse selection.

        Formula:
            min_spread = 2 × (effective_maker_rate + builder_rate) / (1 - adverse_sel_pct)
        Returns the value in bps.
        """
        effective_maker = self._effective_maker_rate  # already includes staking + rebate
        builder_rate = self._cfg.builder_fee_bps / 10_000
        raw = 2.0 * (effective_maker + builder_rate) / (1.0 - adverse_sel_pct)
        return raw * 10_000  # convert fraction to bps

    def is_viable(self, spread_bps: float, notional: float) -> bool:
        """True if the offered spread covers all fees with a safety margin."""
        min_spread = self.min_viable_spread_bps(notional)
        return spread_bps >= min_spread

    def update_volume_tier(self, rolling_14d_volume: float) -> None:
        """Re-derive base rates from rolling 14-day volume. Call daily."""
        tiers = PACIFICA_TIERS if self._cfg.exchange == "pacifica" else VOLUME_TIERS
        new_tier = 0
        for i, tier in enumerate(tiers):
            if rolling_14d_volume >= tier["min"]:
                new_tier = i
        if new_tier != self._cfg.volume_tier:
            logger.info("Volume tier changed: T%d → T%d (vol=$%.0f)",
                        self._cfg.volume_tier, new_tier, rolling_14d_volume)
        self._cfg.volume_tier = new_tier
        self._resolve_rates()

    def update_maker_share(self, maker_vol_pct: float) -> None:
        """Update current maker volume share. Call when platform data is available."""
        self._cfg.maker_volume_share_pct = maker_vol_pct
        self._resolve_rates()

    def fee_report(self) -> str:
        """Human-readable summary of current effective rates."""
        lines = [
            "=" * 55,
            "  FeeEngine — Current Rate Summary",
            "=" * 55,
            f"  Volume tier:      T{self._cfg.volume_tier}",
            f"  Staking tier:     {self._cfg.staking_tier}",
            f"  Maker vol share:  {self._cfg.maker_volume_share_pct * 100:.2f}%",
            f"  Builder fee:      {self._cfg.builder_fee_bps:.1f} bps",
            "-" * 55,
            f"  Base maker rate:  {self._base_maker * 10_000:.2f} bps",
            f"  Base taker rate:  {self._base_taker * 10_000:.2f} bps",
            f"  Staking discount: {STAKING_DISCOUNTS.get(self._cfg.staking_tier, 0) * 100:.0f}%",
            f"  MM rebate rate:   {self._get_rebate_rate() * 10_000:.2f} bps",
            "-" * 55,
            f"  Effective maker:  {self._effective_maker_rate * 10_000:.4f} bps",
            f"  Effective taker:  {self._effective_taker_rate * 10_000:.4f} bps",
            "-" * 55,
            f"  Round-trip (maker/maker) on $100k standard:  "
            f"${self.compute_round_trip(100_000, 'standard'):.2f}",
            f"  Round-trip (maker/maker) on $100k HIP-3:     "
            f"${self.compute_round_trip(100_000, 'hip3'):.2f}",
            f"  Min viable spread (20% adverse sel):         "
            f"{self.min_viable_spread_bps(100_000):.2f} bps",
            "=" * 55,
        ]
        return "\n".join(lines)

    def log_fee_breakdown(self, breakdown: FeeBreakdown) -> dict[str, Any]:
        """Log and return the breakdown as a dict. Useful for P&L attribution."""
        d = breakdown.to_dict()
        logger.info("Fee breakdown: %s", d)
        return d

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _resolve_rates(self) -> None:
        """Derive effective rates from current config. Called on init and tier change."""
        # Select fee table based on exchange
        if self._cfg.exchange == "pacifica":
            tiers = PACIFICA_TIERS
        else:
            tiers = VOLUME_TIERS

        tier_idx = max(0, min(self._cfg.volume_tier, len(tiers) - 1))
        tier = tiers[tier_idx]

        if self._cfg.custom_maker_rate is not None:
            self._base_maker = self._cfg.custom_maker_rate
        else:
            self._base_maker = tier["maker"]

        if self._cfg.custom_taker_rate is not None:
            self._base_taker = self._cfg.custom_taker_rate
        else:
            self._base_taker = tier["taker"]

        # Pacifica RWA discount (50% off all tiers)
        if self._cfg.exchange == "pacifica":
            is_rwa = self._cfg.is_rwa_market or self._cfg.symbol in PACIFICA_RWA_SYMBOLS
            if is_rwa:
                self._base_maker *= PACIFICA_RWA_DISCOUNT
                self._base_taker *= PACIFICA_RWA_DISCOUNT

        # Hyperliquid-specific: staking discount + maker rebate
        if self._cfg.exchange == "hyperliquid":
            staking_disc = STAKING_DISCOUNTS.get(self._cfg.staking_tier, 0.0)
            rebate_rate = self._get_rebate_rate()
            self._effective_maker_rate = self._base_maker * (1.0 - staking_disc) + rebate_rate
            self._effective_taker_rate = self._base_taker * (1.0 - staking_disc)
        else:
            # Pacifica: no staking discounts, no maker-share rebates
            self._effective_maker_rate = self._base_maker
            self._effective_taker_rate = self._base_taker

    def _get_rebate_rate(self) -> float:
        """Look up maker-share rebate from current share percentage. Returns <=0."""
        share = self._cfg.maker_volume_share_pct
        rebate = 0.0
        for tier in MM_REBATES:
            if share >= tier["min_share"]:
                rebate = tier["rebate"]
        return rebate
