"""
QuoteEngine — Avellaneda-Stoikov pricing with OFI adjustment.

Combines the AS reservation price model with the OFI/toxicity signals
from MarketSignals to produce final bid/ask quotes every tick.

This is the Module 4 (Quote Generator) from the architecture spec.
It consumes MarketSignals (Module 3) and FeeEngine (Module 2).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from MM_algo.core.signal_engine import MarketSignals, MarketState
from MM_algo.core.fee_engine import FeeEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class QuoteDecision:
    """Complete output of one quote generation cycle."""
    timestamp: float
    mid_price: float
    reservation_price: float       # r (AS indifference price)
    r_adjusted: float              # r after OFI adjustment
    bid_quote: float               # final bid price
    ask_quote: float               # final ask price
    spread_bps: float              # delta* in basis points
    half_spread: float             # delta*/2 in price units
    ofi_signal: float              # z-scored OFI
    toxicity_score: float
    vol_regime: str
    sigma: float                   # fast volatility (for spread calc)
    inventory_q: float
    skew: float                    # inventory skew applied
    recommended_action: str        # "quote_both" | "quote_ask_only" |
                                   # "quote_bid_only" | "halt"
    bid_size: float                # recommended order size for bid
    ask_size: float                # recommended order size for ask
    fee_viable: bool               # whether spread covers fees
    min_viable_spread_bps: float


@dataclass
class QuoteEngineConfig:
    """All tunable parameters for the quote engine."""
    # AS model
    gamma: float = 0.1             # risk aversion
    lambda_skew: float = 0.5       # inventory skew strength
    min_spread_bps: float = 2.0    # hard floor on spread

    # OFI
    ofi_alpha: float = 0.05        # OFI sensitivity
    ofi_window: int = 300          # OFI buffer length

    # Inventory
    q_max: float = 50_000.0        # max inventory in USD
    min_order_notional: float = 10.0  # exchange minimum

    # Position sizing
    position_sizing_pct: float = 0.05  # fraction of capital per order
    available_capital: float = 100_000.0

    # Signal engine params
    toxicity_halt_threshold: float = 0.85
    consecutive_adverse_limit: int = 5

    # Funding
    funding_skew_threshold: float = 0.0002  # 0.02%/hr


# ---------------------------------------------------------------------------
# QuoteEngine
# ---------------------------------------------------------------------------

class QuoteEngine:
    """
    Produces bid/ask quotes by combining:
      1. Avellaneda-Stoikov reservation price
      2. OFI-adjusted mid
      3. Toxicity-gated spread
      4. Fee-viability check
      5. Inventory-aware sizing

    Usage:
        engine = QuoteEngine(config, fee_engine)
        engine.on_l2_snapshot(bid, bid_sz, ask, ask_sz, ts)
        engine.on_trade(side, price, size, ts)
        decision = engine.generate_quotes(inventory_q=1000.0)
    """

    def __init__(
        self,
        config: QuoteEngineConfig,
        fee_engine: FeeEngine,
    ) -> None:
        self._cfg = config
        self._fee = fee_engine

        # Internal signal engine
        self._signals = MarketSignals(
            ofi_alpha=config.ofi_alpha,
            ofi_window=config.ofi_window,
            toxicity_halt_threshold=config.toxicity_halt_threshold,
            consecutive_adverse_limit=config.consecutive_adverse_limit,
        )

        # Order arrival intensity tracker (fills per second over last 60s)
        self._fill_times: deque[float] = deque(maxlen=500)
        self._kappa: float = 0.01  # minimum to avoid div/0

    # -------------------------------------------------------------------
    # Feed handlers — delegate to MarketSignals
    # -------------------------------------------------------------------

    def on_l2_snapshot(
        self,
        best_bid: float,
        best_bid_size: float,
        best_ask: float,
        best_ask_size: float,
        timestamp: float | None = None,
    ) -> None:
        self._signals.on_l2_snapshot(
            best_bid, best_bid_size, best_ask, best_ask_size, timestamp
        )

    def on_trade(
        self,
        side: str,
        price: float,
        size: float,
        timestamp: float | None = None,
    ) -> None:
        self._signals.on_trade(side, price, size, timestamp)

    def on_own_fill(
        self,
        fill_side: str,
        fill_price: float,
        current_mid: float,
        timestamp: float | None = None,
    ) -> None:
        """Record our own fill for toxicity tracking and kappa estimation."""
        ts = timestamp if timestamp is not None else time.time()
        self._fill_times.append(ts)
        self._signals.on_own_fill(fill_side, fill_price, current_mid)

    def update_funding_rate(self, hourly_rate: float) -> None:
        self._signals.update_funding_rate(hourly_rate)

    # -------------------------------------------------------------------
    # Core quote generation
    # -------------------------------------------------------------------

    def generate_quotes(
        self,
        inventory_q: float,
        timestamp: float | None = None,
    ) -> QuoteDecision:
        """
        Main entry point. Generates a full QuoteDecision from current
        market state and inventory.

        Args:
            inventory_q: current net position in USD (positive=long)
            timestamp:   epoch seconds (None → now)
        """
        ts = timestamp if timestamp is not None else time.time()

        # Step 1: Get signal state
        state: MarketState = self._signals.compute(timestamp=ts)
        mid = state.mid_price

        if mid <= 0:
            return self._halt_decision(ts, mid, inventory_q, state, "no_valid_mid")

        # Step 2: Compute volatility (use fast sigma from signals)
        sigma = state.sigma_fast
        sigma_sq = sigma * sigma

        # Step 3: Order arrival intensity
        kappa = self._estimate_kappa(ts)

        # Step 4: AS reservation price
        gamma = self._cfg.gamma
        q_norm = inventory_q / self._cfg.q_max if self._cfg.q_max > 0 else 0.0
        reservation = mid - q_norm * gamma * sigma_sq * mid

        # Step 5: OFI adjustment
        r_adjusted = reservation + state.ofi_reservation_adj * mid

        # Step 6: Optimal half-spread (AS formula)
        if gamma > 0 and kappa > 0:
            half_spread = (gamma * sigma_sq + (2.0 / gamma) * np.log(1.0 + gamma / kappa)) / 2.0
        else:
            half_spread = sigma  # fallback

        # Convert to price units
        half_spread_price = half_spread * mid

        # Step 7: Apply toxicity multiplier
        half_spread_price *= state.spread_multiplier

        # Step 8: Apply funding skew
        funding_adj = self._funding_skew(state.funding_rate, mid)

        # Step 9: Enforce minimum spread
        min_half_spread = self._cfg.min_spread_bps * mid / 20_000  # min_bps/2 in price
        half_spread_price = max(half_spread_price, min_half_spread)

        # Step 10: Inventory skew
        skew = self._cfg.lambda_skew * q_norm * half_spread_price

        # Step 11: Final quotes
        bid = r_adjusted - half_spread_price - skew + funding_adj
        ask = r_adjusted + half_spread_price + skew + funding_adj

        spread_bps = (ask - bid) / mid * 10_000

        # Step 12: Fee viability check
        min_viable = self._fee.min_viable_spread_bps(
            self._cfg.available_capital * self._cfg.position_sizing_pct
        )
        fee_viable = spread_bps >= min_viable

        # If spread too tight, widen to minimum viable
        if not fee_viable:
            required_half = min_viable * mid / 20_000
            bid = r_adjusted - required_half - skew + funding_adj
            ask = r_adjusted + required_half + skew + funding_adj
            half_spread_price = required_half
            spread_bps = (ask - bid) / mid * 10_000
            fee_viable = True

        # Step 13: Determine action
        action = self._determine_action(state, inventory_q)

        # Step 14: Compute order sizes
        bid_size, ask_size = self._compute_sizes(
            inventory_q, sigma, mid
        )

        return QuoteDecision(
            timestamp=ts,
            mid_price=mid,
            reservation_price=reservation,
            r_adjusted=r_adjusted,
            bid_quote=bid,
            ask_quote=ask,
            spread_bps=spread_bps,
            half_spread=half_spread_price,
            ofi_signal=state.ofi_zscore,
            toxicity_score=state.toxicity_score,
            vol_regime=state.vol_regime,
            sigma=sigma,
            inventory_q=inventory_q,
            skew=skew,
            recommended_action=action,
            bid_size=bid_size,
            ask_size=ask_size,
            fee_viable=fee_viable,
            min_viable_spread_bps=min_viable,
        )

    # -------------------------------------------------------------------
    # Backtest mode
    # -------------------------------------------------------------------

    def backtest(
        self,
        l2_snapshots: list[dict],
        trades: list[dict],
        inventory_series: list[float] | None = None,
    ) -> list[QuoteDecision]:
        """
        Replay historical data and generate a QuoteDecision timeseries.

        Args:
            l2_snapshots: list of {ts, best_bid, best_bid_size, best_ask, best_ask_size}
            trades:       list of {ts, side, price, size}
            inventory_series: optional inventory at each snapshot (default 0)
        """
        results: list[QuoteDecision] = []
        snap_idx = 0
        trade_idx = 0
        n_snaps = len(l2_snapshots)
        n_trades = len(trades)
        inv_idx = 0

        while snap_idx < n_snaps or trade_idx < n_trades:
            snap_ts = l2_snapshots[snap_idx]["ts"] if snap_idx < n_snaps else float("inf")
            trade_ts = trades[trade_idx]["ts"] if trade_idx < n_trades else float("inf")

            if snap_ts <= trade_ts:
                s = l2_snapshots[snap_idx]
                self.on_l2_snapshot(
                    s["best_bid"], s["best_bid_size"],
                    s["best_ask"], s["best_ask_size"],
                    timestamp=s["ts"],
                )
                q = 0.0
                if inventory_series and inv_idx < len(inventory_series):
                    q = inventory_series[inv_idx]
                    inv_idx += 1

                decision = self.generate_quotes(inventory_q=q, timestamp=s["ts"])
                results.append(decision)
                snap_idx += 1
            else:
                t = trades[trade_idx]
                self.on_trade(t["side"], t["price"], t["size"], timestamp=t["ts"])
                trade_idx += 1

        return results

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _estimate_kappa(self, now: float) -> float:
        """Estimate order arrival intensity (fills/sec) over last 60s."""
        if len(self._fill_times) < 2:
            return self._kappa

        cutoff = now - 60.0
        recent = [t for t in self._fill_times if t > cutoff]
        if len(recent) < 2:
            return max(self._kappa, 0.01)

        duration = now - recent[0]
        if duration <= 0:
            return self._kappa

        self._kappa = max(len(recent) / duration, 0.01)
        return self._kappa

    def _funding_skew(self, funding_rate: float, mid: float) -> float:
        """
        Skew quotes based on funding rate.
        Positive funding (longs pay) → skew asks tighter (encourage shorts).
        Negative funding (shorts pay) → skew bids tighter.
        Returns a price offset to apply to both bid and ask.
        """
        threshold = self._cfg.funding_skew_threshold
        if abs(funding_rate) < threshold:
            return 0.0

        # Magnitude of the skew: proportional to funding rate
        magnitude = abs(funding_rate) * mid * 0.5
        if funding_rate > 0:
            return -magnitude  # lower both quotes (tighter ask)
        else:
            return magnitude   # raise both quotes (tighter bid)

    def _determine_action(
        self,
        state: MarketState,
        inventory_q: float,
    ) -> str:
        """Determine recommended quoting action."""
        if state.action == "halt":
            return "halt"

        q_ratio = abs(inventory_q) / self._cfg.q_max if self._cfg.q_max > 0 else 0.0

        # Near inventory limit: only quote the reducing side
        if q_ratio > 0.9:
            if inventory_q > 0:
                return "quote_ask_only"  # only sell to reduce long
            else:
                return "quote_bid_only"  # only buy to reduce short

        return "quote_both"

    def _compute_sizes(
        self,
        inventory_q: float,
        sigma: float,
        mid: float,
    ) -> tuple[float, float]:
        """
        Compute bid and ask order sizes.
        Scales down with inventory proximity and volatility.
        """
        cfg = self._cfg
        base_notional = cfg.available_capital * cfg.position_sizing_pct

        # Inventory dampening: reduce size as inventory approaches limit
        q_ratio = abs(inventory_q) / cfg.q_max if cfg.q_max > 0 else 0.0
        inv_scalar = max(1.0 - q_ratio, 0.0)

        # Volatility dampening: reduce size in high vol
        vol_scalar = 1.0 / (1.0 + 3.0 * sigma) if sigma > 0 else 1.0

        size_notional = base_notional * inv_scalar * vol_scalar

        # Convert to base asset quantity
        size = size_notional / mid if mid > 0 else 0.0

        # Enforce minimum
        if size_notional < cfg.min_order_notional:
            size = 0.0

        return size, size

    def _halt_decision(
        self,
        ts: float,
        mid: float,
        inventory_q: float,
        state: MarketState,
        reason: str,
    ) -> QuoteDecision:
        """Return a halt QuoteDecision."""
        return QuoteDecision(
            timestamp=ts,
            mid_price=mid,
            reservation_price=mid,
            r_adjusted=mid,
            bid_quote=0.0,
            ask_quote=0.0,
            spread_bps=0.0,
            half_spread=0.0,
            ofi_signal=state.ofi_zscore,
            toxicity_score=state.toxicity_score,
            vol_regime=state.vol_regime,
            sigma=state.sigma_fast,
            inventory_q=inventory_q,
            skew=0.0,
            recommended_action="halt",
            bid_size=0.0,
            ask_size=0.0,
            fee_viable=False,
            min_viable_spread_bps=0.0,
        )
