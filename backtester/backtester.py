"""
MMBacktester — realistic market making simulation engine.

Replays historical L2 data with queue-position-aware fill simulation,
full fee attribution, and adverse selection tracking.

Key realism features:
  - Queue position modelling (no front-of-queue assumption)
  - Partial fills proportional to traded volume at level
  - Full FeeEngine integration on every simulated fill
  - Adverse selection tagging (mid moves against within N seconds)
  - Deterministic (same seed → same result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from MM_algo.core.fee_engine import FeeEngine, FeeConfig
from MM_algo.core.quote_engine import QuoteEngine, QuoteEngineConfig
from MM_algo.core.signal_engine import MarketSignals

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FillRecord:
    """One simulated fill with full P&L attribution."""
    timestamp: float
    side: str
    price: float
    size: float
    notional: float
    fill_type: str              # "normal" or "adverse"
    spread_captured: float      # half-spread revenue
    fee_paid: float             # total fees (could be negative for rebate)
    adverse_selection_loss: float
    net_pnl: float
    inventory_after: float
    mid_at_fill: float
    mid_after_5s: float | None = None


@dataclass
class BacktestSummary:
    """Aggregate statistics from a backtest run."""
    total_pnl: float
    pnl_from_spread: float
    pnl_from_rebates: float
    total_fees_paid: float
    adverse_selection_loss: float
    fill_count: int
    adverse_fill_count: int
    adverse_selection_rate: float
    fill_rate_per_hour: float
    maker_volume_total: float
    max_drawdown_pct: float
    sharpe_ratio: float              # hourly
    time_at_q_max_pct: float         # fraction of time near inventory limit
    fee_pct_of_gross: float          # fees as % of gross P&L
    breakeven_adverse_rate: float
    duration_hours: float
    params: dict = field(default_factory=dict)


@dataclass
class SimulatedOrder:
    """An order in the simulated book."""
    side: str
    price: float
    size: float
    placed_at: float
    ttl_s: float
    queue_ahead: float        # volume ahead of us in the queue


# ---------------------------------------------------------------------------
# MMBacktester
# ---------------------------------------------------------------------------

class MMBacktester:
    """
    Replay historical data and simulate MM strategy P&L.

    Usage:
        bt = MMBacktester(quote_config, fee_config)
        fills, summary = bt.run(l2_snapshots, trades, funding)
        bt.plot_results(fills, summary)
    """

    def __init__(
        self,
        quote_config: QuoteEngineConfig | None = None,
        fee_config: FeeConfig | None = None,
        queue_priority_factor: float = 0.3,
        adverse_sel_window_s: float = 5.0,
        seed: int = 42,
    ) -> None:
        self._qcfg = quote_config or QuoteEngineConfig()
        self._fcfg = fee_config or FeeConfig()
        self._queue_pf = queue_priority_factor
        self._adverse_window = adverse_sel_window_s
        self._rng = np.random.RandomState(seed)

    # -------------------------------------------------------------------
    # Main backtest
    # -------------------------------------------------------------------

    def run(
        self,
        l2_snapshots: list[dict],
        trades: list[dict],
        funding: list[dict] | None = None,
    ) -> tuple[list[FillRecord], BacktestSummary]:
        """
        Run a full backtest.

        Args:
            l2_snapshots: [{ts, best_bid, best_bid_size, best_ask, best_ask_size, ...}]
            trades:       [{ts, side, price, size}]
            funding:      [{ts, rate}] (optional)

        Returns:
            (list of FillRecord, BacktestSummary)
        """
        fee_engine = FeeEngine(FeeConfig(
            volume_tier=self._fcfg.volume_tier,
            staking_tier=self._fcfg.staking_tier,
            builder_fee_bps=self._fcfg.builder_fee_bps,
            maker_volume_share_pct=self._fcfg.maker_volume_share_pct,
        ))
        quote_engine = QuoteEngine(self._qcfg, fee_engine)

        fills: list[FillRecord] = []
        inventory_q = 0.0
        equity = self._qcfg.available_capital
        peak_equity = equity
        max_drawdown = 0.0
        time_near_qmax = 0.0

        # Active simulated orders
        active_bid: SimulatedOrder | None = None
        active_ask: SimulatedOrder | None = None

        # Merge streams
        snap_idx = 0
        trade_idx = 0
        fund_idx = 0
        n_snaps = len(l2_snapshots)
        n_trades = len(trades)
        n_fund = len(funding) if funding else 0

        # Hourly P&L tracking for Sharpe
        hourly_pnl: list[float] = []
        current_hour_pnl = 0.0
        current_hour_start = l2_snapshots[0]["ts"] if l2_snapshots else 0

        last_mid = 0.0

        while snap_idx < n_snaps or trade_idx < n_trades:
            snap_ts = l2_snapshots[snap_idx]["ts"] if snap_idx < n_snaps else float("inf")
            trade_ts = trades[trade_idx]["ts"] if trade_idx < n_trades else float("inf")

            if snap_ts <= trade_ts and snap_idx < n_snaps:
                # Process L2 snapshot — generate new quotes
                s = l2_snapshots[snap_idx]
                ts = s["ts"]

                # Update funding
                while fund_idx < n_fund and funding[fund_idx]["ts"] <= ts:
                    quote_engine.update_funding_rate(funding[fund_idx]["rate"])
                    fund_idx += 1

                quote_engine.on_l2_snapshot(
                    s["best_bid"], s["best_bid_size"],
                    s["best_ask"], s["best_ask_size"],
                    timestamp=ts,
                )
                last_mid = (s["best_bid"] + s["best_ask"]) / 2.0

                # Cancel stale orders
                if active_bid and ts - active_bid.placed_at > active_bid.ttl_s:
                    active_bid = None
                if active_ask and ts - active_ask.placed_at > active_ask.ttl_s:
                    active_ask = None

                # Generate new quotes
                decision = quote_engine.generate_quotes(inventory_q, timestamp=ts)

                if decision.recommended_action != "halt":
                    bid_size_total = s.get("best_bid_size", 100.0)
                    ask_size_total = s.get("best_ask_size", 100.0)
                    ttl = 0.5 if decision.vol_regime == "high_vol" else 2.0

                    if decision.recommended_action in ("quote_both", "quote_bid_only"):
                        if decision.bid_size > 0:
                            active_bid = SimulatedOrder(
                                side="buy",
                                price=decision.bid_quote,
                                size=decision.bid_size,
                                placed_at=ts,
                                ttl_s=ttl,
                                queue_ahead=bid_size_total * (1 - self._queue_pf),
                            )

                    if decision.recommended_action in ("quote_both", "quote_ask_only"):
                        if decision.ask_size > 0:
                            active_ask = SimulatedOrder(
                                side="sell",
                                price=decision.ask_quote,
                                size=decision.ask_size,
                                placed_at=ts,
                                ttl_s=ttl,
                                queue_ahead=ask_size_total * (1 - self._queue_pf),
                            )
                else:
                    active_bid = None
                    active_ask = None

                # Track time near q_max
                if self._qcfg.q_max > 0 and abs(inventory_q) > self._qcfg.q_max * 0.8:
                    time_near_qmax += 0.5  # approximate 0.5s per snapshot

                snap_idx += 1

            elif trade_idx < n_trades:
                # Process market trade — check if it fills our orders
                t = trades[trade_idx]
                ts = t["ts"]
                trade_side = t["side"]
                trade_price = float(t["price"])
                trade_size = float(t["size"])

                quote_engine.on_trade(trade_side, trade_price, trade_size, timestamp=ts)

                # Check bid fill: market sell trades that hit our bid
                if active_bid and trade_side == "sell" and trade_price <= active_bid.price:
                    active_bid.queue_ahead -= trade_size
                    if active_bid.queue_ahead <= 0:
                        fill_size = min(active_bid.size, trade_size + active_bid.queue_ahead)
                        if fill_size > 0:
                            fill_notional = fill_size * active_bid.price
                            spread_captured = (last_mid - active_bid.price) * fill_size
                            bd = fee_engine.compute_fee(fill_notional, "maker")

                            # Adverse selection check
                            mid_after = self._find_mid_after(
                                l2_snapshots, snap_idx, ts, self._adverse_window
                            )
                            adverse_loss = 0.0
                            fill_type = "normal"
                            if mid_after is not None and mid_after < active_bid.price:
                                adverse_loss = (active_bid.price - mid_after) * fill_size
                                fill_type = "adverse"

                            net = spread_captured - bd.net_fee - adverse_loss
                            inventory_q += fill_notional

                            fill = FillRecord(
                                timestamp=ts,
                                side="buy",
                                price=active_bid.price,
                                size=fill_size,
                                notional=fill_notional,
                                fill_type=fill_type,
                                spread_captured=spread_captured,
                                fee_paid=bd.net_fee,
                                adverse_selection_loss=adverse_loss,
                                net_pnl=net,
                                inventory_after=inventory_q,
                                mid_at_fill=last_mid,
                                mid_after_5s=mid_after,
                            )
                            fills.append(fill)
                            equity += net
                            current_hour_pnl += net

                            peak_equity = max(peak_equity, equity)
                            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
                            max_drawdown = max(max_drawdown, dd)

                        active_bid = None

                # Check ask fill: market buy trades that hit our ask
                if active_ask and trade_side == "buy" and trade_price >= active_ask.price:
                    active_ask.queue_ahead -= trade_size
                    if active_ask.queue_ahead <= 0:
                        fill_size = min(active_ask.size, trade_size + active_ask.queue_ahead)
                        if fill_size > 0:
                            fill_notional = fill_size * active_ask.price
                            spread_captured = (active_ask.price - last_mid) * fill_size
                            bd = fee_engine.compute_fee(fill_notional, "maker")

                            mid_after = self._find_mid_after(
                                l2_snapshots, snap_idx, ts, self._adverse_window
                            )
                            adverse_loss = 0.0
                            fill_type = "normal"
                            if mid_after is not None and mid_after > active_ask.price:
                                adverse_loss = (mid_after - active_ask.price) * fill_size
                                fill_type = "adverse"

                            net = spread_captured - bd.net_fee - adverse_loss
                            inventory_q -= fill_notional

                            fill = FillRecord(
                                timestamp=ts,
                                side="sell",
                                price=active_ask.price,
                                size=fill_size,
                                notional=fill_notional,
                                fill_type=fill_type,
                                spread_captured=spread_captured,
                                fee_paid=bd.net_fee,
                                adverse_selection_loss=adverse_loss,
                                net_pnl=net,
                                inventory_after=inventory_q,
                                mid_at_fill=last_mid,
                                mid_after_5s=mid_after,
                            )
                            fills.append(fill)
                            equity += net
                            current_hour_pnl += net

                            peak_equity = max(peak_equity, equity)
                            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
                            max_drawdown = max(max_drawdown, dd)

                        active_ask = None

                # Hourly P&L bucketing
                if ts - current_hour_start >= 3600:
                    hourly_pnl.append(current_hour_pnl)
                    current_hour_pnl = 0.0
                    current_hour_start = ts

                trade_idx += 1

        # Final hour bucket
        if current_hour_pnl != 0:
            hourly_pnl.append(current_hour_pnl)

        # Compute summary
        summary = self._compute_summary(fills, hourly_pnl, l2_snapshots,
                                         max_drawdown, time_near_qmax)
        return fills, summary

    # -------------------------------------------------------------------
    # Parameter sweep
    # -------------------------------------------------------------------

    def run_sweep(
        self,
        l2_snapshots: list[dict],
        trades: list[dict],
        param_grid: dict | None = None,
        funding: list[dict] | None = None,
    ) -> list[BacktestSummary]:
        """
        Run backtests across a parameter grid.

        Args:
            param_grid: dict mapping param names to lists of values.
                Default grid:
                    gamma: [0.05, 0.1, 0.2, 0.5]
                    lambda_skew: [0.2, 0.5, 1.0]
                    q_max: [10000, 25000, 50000, 100000]
                    ofi_alpha: [0.01, 0.05, 0.10]

        Returns:
            List of BacktestSummary sorted by total_pnl descending.
        """
        if param_grid is None:
            param_grid = {
                "gamma": [0.05, 0.1, 0.2, 0.5],
                "lambda_skew": [0.2, 0.5, 1.0],
                "q_max": [10_000, 25_000, 50_000, 100_000],
                "ofi_alpha": [0.01, 0.05, 0.10],
            }

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        results: list[BacktestSummary] = []

        for combo in product(*values):
            params = dict(zip(keys, combo))
            logger.info("Sweep: %s", params)

            qcfg = QuoteEngineConfig(
                gamma=params.get("gamma", self._qcfg.gamma),
                lambda_skew=params.get("lambda_skew", self._qcfg.lambda_skew),
                q_max=params.get("q_max", self._qcfg.q_max),
                ofi_alpha=params.get("ofi_alpha", self._qcfg.ofi_alpha),
                min_spread_bps=self._qcfg.min_spread_bps,
                position_sizing_pct=self._qcfg.position_sizing_pct,
                available_capital=self._qcfg.available_capital,
            )

            bt = MMBacktester(
                quote_config=qcfg,
                fee_config=self._fcfg,
                queue_priority_factor=self._queue_pf,
                adverse_sel_window_s=self._adverse_window,
                seed=42,  # deterministic
            )
            _, summary = bt.run(l2_snapshots, trades, funding)
            summary.params = params
            results.append(summary)

        results.sort(key=lambda s: s.total_pnl, reverse=True)
        return results

    # -------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------

    def plot_results(
        self,
        fills: list[FillRecord],
        summary: BacktestSummary,
        save_path: str | None = None,
    ) -> None:
        """
        Plot backtest results:
          subplot 1: cumulative P&L by source
          subplot 2: inventory over time
          subplot 3: spread captured vs adverse selection loss
          subplot 4: fee breakdown (pie chart)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed — skipping plot")
            return

        if not fills:
            logger.warning("No fills to plot")
            return

        timestamps = [f.timestamp for f in fills]
        cum_spread = np.cumsum([f.spread_captured for f in fills])
        cum_fees = np.cumsum([-f.fee_paid for f in fills])
        cum_adverse = np.cumsum([-f.adverse_selection_loss for f in fills])
        cum_net = np.cumsum([f.net_pnl for f in fills])
        inventory = [f.inventory_after for f in fills]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Cumulative P&L by source
        ax1 = axes[0, 0]
        ax1.plot(timestamps, cum_spread, label="Spread capture", alpha=0.8)
        ax1.plot(timestamps, cum_fees, label="Fees (neg=paid)", alpha=0.8)
        ax1.plot(timestamps, cum_adverse, label="Adverse sel. loss", alpha=0.8)
        ax1.plot(timestamps, cum_net, label="Net P&L", linewidth=2, color="black")
        ax1.set_title("Cumulative P&L by Source")
        ax1.legend(fontsize=8)
        ax1.set_ylabel("USD")
        ax1.grid(True, alpha=0.3)

        # 2. Inventory over time
        ax2 = axes[0, 1]
        ax2.fill_between(timestamps, inventory, alpha=0.3)
        ax2.plot(timestamps, inventory, linewidth=0.5)
        ax2.set_title("Inventory (USD)")
        ax2.set_ylabel("Net Delta (USD)")
        ax2.grid(True, alpha=0.3)

        # 3. Spread captured vs adverse selection per fill
        ax3 = axes[1, 0]
        spreads = [f.spread_captured for f in fills]
        adverses = [f.adverse_selection_loss for f in fills]
        ax3.scatter(spreads, adverses, alpha=0.3, s=5)
        ax3.set_xlabel("Spread Captured ($)")
        ax3.set_ylabel("Adverse Selection Loss ($)")
        ax3.set_title("Spread vs Adverse Selection per Fill")
        ax3.grid(True, alpha=0.3)

        # 4. Fee breakdown pie chart
        ax4 = axes[1, 1]
        total_spread = sum(f.spread_captured for f in fills)
        total_fees = sum(f.fee_paid for f in fills)
        total_adverse = sum(f.adverse_selection_loss for f in fills)
        total_net = sum(f.net_pnl for f in fills)
        labels = ["Spread Capture", "Fees Paid", "Adverse Loss", "Net P&L"]
        sizes = [abs(total_spread), abs(total_fees), abs(total_adverse), abs(total_net)]
        ax4.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax4.set_title("P&L Attribution")

        plt.suptitle(
            f"Backtest: {summary.fill_count} fills, "
            f"PnL=${summary.total_pnl:.2f}, "
            f"Sharpe={summary.sharpe_ratio:.2f}, "
            f"MaxDD={summary.max_drawdown_pct:.2f}%",
            fontsize=12,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info("Plot saved to %s", save_path)
        else:
            plt.show()

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _find_mid_after(
        self,
        snapshots: list[dict],
        current_idx: int,
        fill_ts: float,
        window_s: float,
    ) -> float | None:
        """Find the mid price window_s seconds after fill_ts."""
        target_ts = fill_ts + window_s
        for i in range(current_idx, min(current_idx + 100, len(snapshots))):
            s = snapshots[i]
            if s["ts"] >= target_ts:
                return (s["best_bid"] + s["best_ask"]) / 2.0
        return None

    def _compute_summary(
        self,
        fills: list[FillRecord],
        hourly_pnl: list[float],
        snapshots: list[dict],
        max_drawdown: float,
        time_near_qmax: float,
    ) -> BacktestSummary:
        if not fills:
            return BacktestSummary(
                total_pnl=0, pnl_from_spread=0, pnl_from_rebates=0,
                total_fees_paid=0, adverse_selection_loss=0,
                fill_count=0, adverse_fill_count=0, adverse_selection_rate=0,
                fill_rate_per_hour=0, maker_volume_total=0,
                max_drawdown_pct=0, sharpe_ratio=0, time_at_q_max_pct=0,
                fee_pct_of_gross=0, breakeven_adverse_rate=0, duration_hours=0,
            )

        total_pnl = sum(f.net_pnl for f in fills)
        pnl_spread = sum(f.spread_captured for f in fills)
        total_fees = sum(f.fee_paid for f in fills)
        rebates = sum(min(f.fee_paid, 0) for f in fills)  # negative fees = rebates
        adverse_loss = sum(f.adverse_selection_loss for f in fills)
        adverse_count = sum(1 for f in fills if f.fill_type == "adverse")
        maker_vol = sum(f.notional for f in fills)

        duration_s = fills[-1].timestamp - fills[0].timestamp if len(fills) > 1 else 1
        duration_hrs = max(duration_s / 3600, 0.001)

        total_time_s = (
            snapshots[-1]["ts"] - snapshots[0]["ts"]
            if len(snapshots) > 1 else 1
        )

        # Sharpe ratio (hourly)
        if len(hourly_pnl) > 1:
            hp = np.array(hourly_pnl)
            sharpe = float(hp.mean() / (hp.std() + 1e-9))
        else:
            sharpe = 0.0

        gross_pnl = pnl_spread
        fee_pct = (total_fees / gross_pnl * 100) if gross_pnl > 0 else 0.0

        # Breakeven adverse selection rate
        if pnl_spread > 0:
            breakeven_adverse = 1.0 - (total_fees / pnl_spread)
        else:
            breakeven_adverse = 0.0

        return BacktestSummary(
            total_pnl=total_pnl,
            pnl_from_spread=pnl_spread,
            pnl_from_rebates=rebates,
            total_fees_paid=total_fees,
            adverse_selection_loss=adverse_loss,
            fill_count=len(fills),
            adverse_fill_count=adverse_count,
            adverse_selection_rate=adverse_count / len(fills) if fills else 0,
            fill_rate_per_hour=len(fills) / duration_hrs,
            maker_volume_total=maker_vol,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe,
            time_at_q_max_pct=time_near_qmax / total_time_s if total_time_s > 0 else 0,
            fee_pct_of_gross=fee_pct,
            breakeven_adverse_rate=breakeven_adverse,
            duration_hours=duration_hrs,
        )
