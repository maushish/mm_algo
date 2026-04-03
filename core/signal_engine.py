"""
Signal Engine — OFI, Toxicity, Volatility, and Funding signals.

Maintains streaming state from the order book and trade feed,
outputting actionable MarketState signals every tick.

All hot-path computation uses numpy and deques — no pandas.
Target: < 2ms per update.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class MarketState:
    """Snapshot of all signals at a single point in time."""
    timestamp: float               # epoch seconds
    mid_price: float
    ofi_zscore: float
    ofi_reservation_adj: float     # price adjustment from OFI (in quote units)
    toxicity_score: float
    vol_regime: str                # "normal" | "high_vol" | "low_vol"
    sigma_fast: float              # 10-min EWMA vol (annualised)
    sigma_slow: float              # 1-hr  EWMA vol (annualised)
    funding_rate: float            # hourly funding rate
    spread_multiplier: float       # 1.0 | 1.5 | 2.5 (from toxicity)
    action: str                    # "quote" | "halt"
    halt_reason: str | None = None


# ---------------------------------------------------------------------------
# Toxicity response table
# ---------------------------------------------------------------------------

TOXICITY_THRESHOLDS = [
    (0.85, 2.5, "halt"),    # > 0.85: halt
    (0.65, 2.5, "quote"),   # 0.65–0.85: danger, widen 2.5×
    (0.40, 1.5, "quote"),   # 0.40–0.65: caution, widen 1.5×
    (0.00, 1.0, "quote"),   # < 0.40: normal
]


def _toxicity_response(score: float, halt_threshold: float = 0.85):
    """Returns (spread_multiplier, action) based on toxicity score."""
    if score > halt_threshold:
        return 2.5, "halt"
    for threshold, multiplier, action in TOXICITY_THRESHOLDS:
        if score >= threshold:
            return multiplier, action
    return 1.0, "quote"


# ---------------------------------------------------------------------------
# MarketSignals — the core signal engine
# ---------------------------------------------------------------------------

class MarketSignals:
    """
    Streaming signal engine. Feed it L2 snapshots and trades,
    get back a MarketState with actionable signals.

    Usage:
        signals = MarketSignals(config)
        signals.on_l2_snapshot(best_bid, best_bid_size, best_ask, best_ask_size)
        signals.on_trade(side="buy", price=100.5, size=10.0)
        state = signals.compute()
    """

    def __init__(
        self,
        *,
        ofi_alpha: float = 0.05,
        ofi_window: int = 300,         # ~150s at 500ms intervals
        trade_window: int = 100,
        vol_fast_span_s: float = 600,  # 10-min EWMA
        vol_slow_span_s: float = 3600, # 1-hr EWMA
        toxicity_halt_threshold: float = 0.85,
        consecutive_adverse_limit: int = 5,
        funding_rate: float = 0.0,
    ) -> None:
        # Config
        self._ofi_alpha = ofi_alpha
        self._toxicity_halt = toxicity_halt_threshold
        self._consec_adverse_limit = consecutive_adverse_limit

        # OFI state
        self._ofi_buffer: deque[float] = deque(maxlen=ofi_window)
        self._prev_best_bid_size: float | None = None
        self._prev_best_ask_size: float | None = None

        # Price / mid
        self._mid_price: float = 0.0
        self._best_bid: float = 0.0
        self._best_ask: float = 0.0

        # Trade / toxicity state
        self._trade_buffer: deque[dict] = deque(maxlen=trade_window)
        self._consecutive_adverse: int = 0

        # Volatility — EWMA of squared returns
        self._vol_fast_span = vol_fast_span_s
        self._vol_slow_span = vol_slow_span_s
        self._last_mid: float | None = None
        self._last_mid_time: float | None = None
        self._ewma_var_fast: float = 0.0
        self._ewma_var_slow: float = 0.0
        self._vol_initialised: bool = False

        # Funding
        self._funding_rate: float = funding_rate

        # Latest OFI z-score (cached between compute calls)
        self._latest_ofi_z: float = 0.0

    # -------------------------------------------------------------------
    # Public feed handlers
    # -------------------------------------------------------------------

    def on_l2_snapshot(
        self,
        best_bid: float,
        best_bid_size: float,
        best_ask: float,
        best_ask_size: float,
        timestamp: float | None = None,
    ) -> None:
        """
        Feed an L2 top-of-book snapshot. Call every ~500ms or on every
        orderbook websocket message.
        """
        ts = timestamp if timestamp is not None else time.time()
        self._best_bid = best_bid
        self._best_ask = best_ask
        self._mid_price = (best_bid + best_ask) / 2.0

        # OFI delta
        if self._prev_best_bid_size is not None:
            delta_bid = best_bid_size - self._prev_best_bid_size
            delta_ask = best_ask_size - self._prev_best_ask_size
            ofi_raw = delta_bid - delta_ask
            self._ofi_buffer.append(ofi_raw)

        self._prev_best_bid_size = best_bid_size
        self._prev_best_ask_size = best_ask_size

        # Volatility update
        self._update_volatility(self._mid_price, ts)

    def on_trade(
        self,
        side: str,
        price: float,
        size: float,
        timestamp: float | None = None,
    ) -> None:
        """Feed a market trade event (from the trade websocket)."""
        self._trade_buffer.append({
            "side": side,
            "price": price,
            "size": size,
            "ts": timestamp if timestamp is not None else time.time(),
        })

    def on_own_fill(
        self,
        fill_side: str,
        fill_price: float,
        current_mid: float,
    ) -> None:
        """
        Feed our own fill to track adverse selection streaks.

        A fill is 'adverse' if we bought and mid dropped, or sold and mid rose.
        In practice you'd check mid 5s after fill; here we accept the caller's
        assessment of current_mid at check time.
        """
        if fill_side == "buy" and current_mid < fill_price:
            self._consecutive_adverse += 1
        elif fill_side == "sell" and current_mid > fill_price:
            self._consecutive_adverse += 1
        else:
            self._consecutive_adverse = 0

    def update_funding_rate(self, hourly_rate: float) -> None:
        """Update funding rate. Call every ~5 minutes."""
        self._funding_rate = hourly_rate

    # -------------------------------------------------------------------
    # Core compute — call every tick
    # -------------------------------------------------------------------

    def compute(self, timestamp: float | None = None) -> MarketState:
        """
        Compute the full MarketState from accumulated data.
        Should be called every ~500ms.
        """
        ts = timestamp if timestamp is not None else time.time()

        # OFI z-score
        ofi_z = self._compute_ofi_zscore()
        self._latest_ofi_z = ofi_z

        # Volatility
        sigma_fast = np.sqrt(max(self._ewma_var_fast, 0.0))
        sigma_slow = np.sqrt(max(self._ewma_var_slow, 0.0))

        # Vol regime
        if sigma_slow > 0 and sigma_fast > 2.0 * sigma_slow:
            vol_regime = "high_vol"
        elif sigma_slow > 0 and sigma_fast < 0.5 * sigma_slow:
            vol_regime = "low_vol"
        else:
            vol_regime = "normal"

        # OFI reservation adjustment
        ofi_adj = self._ofi_alpha * ofi_z * sigma_fast

        # Toxicity
        toxicity = self._compute_toxicity()

        # Toxicity response
        spread_mult, action = _toxicity_response(toxicity, self._toxicity_halt)

        # Consecutive adverse fill override
        halt_reason = None
        if action == "halt":
            halt_reason = f"toxicity={toxicity:.3f} > {self._toxicity_halt}"
        if self._consecutive_adverse >= self._consec_adverse_limit:
            action = "halt"
            halt_reason = (
                f"consecutive_adverse_fills={self._consecutive_adverse} "
                f">= {self._consec_adverse_limit}"
            )

        return MarketState(
            timestamp=ts,
            mid_price=self._mid_price,
            ofi_zscore=ofi_z,
            ofi_reservation_adj=ofi_adj,
            toxicity_score=toxicity,
            vol_regime=vol_regime,
            sigma_fast=sigma_fast,
            sigma_slow=sigma_slow,
            funding_rate=self._funding_rate,
            spread_multiplier=spread_mult,
            action=action,
            halt_reason=halt_reason,
        )

    # -------------------------------------------------------------------
    # Backtest mode
    # -------------------------------------------------------------------

    def backtest(
        self,
        l2_snapshots: list[dict],
        trades: list[dict],
    ) -> list[MarketState]:
        """
        Replay historical L2 snapshots and trades, returning a MarketState
        timeseries for strategy validation.

        Args:
            l2_snapshots: list of dicts with keys:
                ts, best_bid, best_bid_size, best_ask, best_ask_size
            trades: list of dicts with keys:
                ts, side, price, size

        Both lists must be sorted by ts. They are merged by timestamp.
        """
        results: list[MarketState] = []

        # Merge the two streams by timestamp
        snap_idx = 0
        trade_idx = 0
        n_snaps = len(l2_snapshots)
        n_trades = len(trades)

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
                state = self.compute(timestamp=s["ts"])
                results.append(state)
                snap_idx += 1
            else:
                t = trades[trade_idx]
                self.on_trade(t["side"], t["price"], t["size"], timestamp=t["ts"])
                trade_idx += 1

        return results

    # -------------------------------------------------------------------
    # Internal computation
    # -------------------------------------------------------------------

    def _compute_ofi_zscore(self) -> float:
        """OFI z-score over the buffer."""
        if len(self._ofi_buffer) < 2:
            return 0.0
        arr = np.array(self._ofi_buffer)
        mean = arr.mean()
        std = arr.std()
        if std < 1e-12:
            return 0.0
        latest = arr[-1]
        return float((latest - mean) / (std + 1e-9))

    def _compute_toxicity(self) -> float:
        """
        Toxicity = |buys - sells| / total trades.
        Over the last N trades in the buffer.
        """
        if len(self._trade_buffer) == 0:
            return 0.0
        buys = sum(1 for t in self._trade_buffer if t["side"] == "buy")
        total = len(self._trade_buffer)
        sells = total - buys
        return abs(buys - sells) / total

    def _update_volatility(self, mid: float, ts: float) -> None:
        """
        Update EWMA variance estimates from the latest mid price.

        Uses the standard EWMA recursion:
            var_new = (1 - alpha) * var_old + alpha * return²
        where alpha = dt / span (capped at 1.0 for robustness).
        """
        if self._last_mid is None or self._last_mid_time is None:
            self._last_mid = mid
            self._last_mid_time = ts
            return

        dt = ts - self._last_mid_time
        if dt <= 0 or self._last_mid <= 0:
            self._last_mid = mid
            self._last_mid_time = ts
            return

        # Log return squared
        ret = np.log(mid / self._last_mid)
        ret_sq = ret * ret

        # EWMA alpha — capped at 1.0 for first few observations
        alpha_fast = min(dt / self._vol_fast_span, 1.0)
        alpha_slow = min(dt / self._vol_slow_span, 1.0)

        if not self._vol_initialised:
            self._ewma_var_fast = ret_sq
            self._ewma_var_slow = ret_sq
            self._vol_initialised = True
        else:
            self._ewma_var_fast = (1 - alpha_fast) * self._ewma_var_fast + alpha_fast * ret_sq
            self._ewma_var_slow = (1 - alpha_slow) * self._ewma_var_slow + alpha_slow * ret_sq

        self._last_mid = mid
        self._last_mid_time = ts
