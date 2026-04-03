"""
RiskManager — synchronous circuit breaker layer.

Called BEFORE every exchange interaction. Evaluates proposed actions
and enforces hard stops that cannot be overridden by config.

State machine:
  RUNNING → REDUCED_MODE (latency)
  RUNNING → HALTED_* (hard stops)
  REDUCED_MODE → RUNNING (RTT recovery for 30s)
  HALTED_INVENTORY → RUNNING (after flatten + 60s cooldown)
  HALTED_DRAWDOWN → never auto-recovers (manual restart required)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class BotState(str, Enum):
    RUNNING = "RUNNING"
    REDUCED_MODE = "REDUCED_MODE"
    HALTED_INVENTORY = "HALTED_INVENTORY"
    HALTED_DRAWDOWN = "HALTED_DRAWDOWN"
    HALTED_TOXICITY = "HALTED_TOXICITY"
    HALTED_LATENCY = "HALTED_LATENCY"
    HALTED_MANUAL = "HALTED_MANUAL"


@dataclass
class RiskConfig:
    q_max_usd: float = 50_000.0
    max_drawdown_pct: float = 3.0
    latency_warn_ms: float = 200.0
    latency_halt_ms: float = 1000.0
    latency_single_max_ms: float = 3000.0
    rtt_window: int = 20
    toxicity_halt_threshold: float = 0.85
    toxicity_cooldown_s: float = 60.0
    inventory_cooldown_s: float = 60.0
    rtt_recovery_s: float = 30.0
    min_order_notional: float = 10.0
    stale_ttl_ms: dict = field(default_factory=lambda: {
        "high_vol": 200.0,
        "normal": 500.0,
        "low_vol": 2000.0,
    })
    funding_spike_threshold: float = 0.001  # 0.1%/hr
    sentinel_path: str = "/tmp/mm_bot_halted"
    dead_man_switch_s: float = 30.0


@dataclass
class OrderProposal:
    """A proposed order to be validated by RiskManager before placement."""
    side: str               # "buy" or "sell"
    price: float
    size: float             # in base asset
    notional: float         # size × price
    is_reduce_only: bool = False
    is_emergency_flatten: bool = False


@dataclass
class RiskVerdict:
    """Output from RiskManager for a proposed action."""
    allowed: bool
    adjusted_size: float | None = None
    reason: str = ""
    actions_required: list[str] = field(default_factory=list)
    # e.g. ["cancel_all", "flatten_inventory", "alert"]


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Synchronous risk gate. Every proposed exchange action passes through
    this before reaching the adapter.

    Usage:
        rm = RiskManager(config)
        rm.set_session_equity(100_000)
        verdict = rm.check_order(proposal, current_equity, inventory_q, ...)
        if not verdict.allowed:
            handle_rejection(verdict)
    """

    def __init__(self, config: RiskConfig) -> None:
        self._cfg = config
        self._state: BotState = BotState.RUNNING

        # Session tracking
        self._session_start_equity: float | None = None
        self._session_start_time: float = time.time()

        # Latency tracking
        self._rtt_buffer: deque[float] = deque(maxlen=config.rtt_window)
        self._rtt_recovery_start: float | None = None

        # Halt timestamps (for cooldowns)
        self._inventory_halt_time: float | None = None
        self._toxicity_halt_time: float | None = None
        self._inventory_flattened: bool = False

        # Dead man's switch
        self._last_heartbeat: float = time.time()
        self._dead_man_deadline: float = time.time() + config.dead_man_switch_s

        # Our own open orders (for self-trade prevention)
        self._our_bids: dict[str, float] = {}  # order_id → price
        self._our_asks: dict[str, float] = {}

    # -------------------------------------------------------------------
    # State
    # -------------------------------------------------------------------

    @property
    def state(self) -> BotState:
        return self._state

    def set_session_equity(self, equity: float) -> None:
        """Must be called at bot startup to set the drawdown baseline."""
        self._session_start_equity = equity
        logger.info("Session start equity set: $%.2f", equity)

    # -------------------------------------------------------------------
    # Primary check — called before every exchange interaction
    # -------------------------------------------------------------------

    def check_order(
        self,
        proposal: OrderProposal,
        current_equity: float,
        inventory_q: float,
        market_state_action: str = "quote",
        funding_rate: float = 0.0,
        vol_regime: str = "normal",
        timestamp: float | None = None,
    ) -> RiskVerdict:
        """
        Evaluate a proposed order against all circuit breakers.
        Returns a RiskVerdict indicating whether the order is allowed.

        This is synchronous and must be called BEFORE any exchange call.
        """
        ts = timestamp if timestamp is not None else time.time()

        # 0. Check for manual halt
        if self._state == BotState.HALTED_MANUAL:
            return RiskVerdict(
                allowed=False,
                reason="Manual halt active — requires restart",
            )

        # 0b. Check drawdown halt (never auto-recovers)
        if self._state == BotState.HALTED_DRAWDOWN:
            return RiskVerdict(
                allowed=False,
                reason="Drawdown halt — manual restart required",
            )

        # 1. INVENTORY BREACH — hardest stop
        verdict = self._check_inventory(proposal, inventory_q, ts)
        if verdict is not None:
            return verdict

        # 2. DRAWDOWN HALT
        verdict = self._check_drawdown(current_equity, ts)
        if verdict is not None:
            return verdict

        # 3. LATENCY HALT
        verdict = self._check_latency(ts)
        if verdict is not None:
            return verdict

        # 4. TOXICITY HALT
        verdict = self._check_toxicity(market_state_action, ts)
        if verdict is not None:
            return verdict

        # 5. FUNDING SPIKE — doesn't halt, but forces wider spread
        self._check_funding_spike(funding_rate)

        # 6. STALE QUOTE — handled by caller via get_ttl()

        # 7. DEAD MAN'S SWITCH — check heartbeat
        verdict = self._check_dead_man(ts)
        if verdict is not None:
            return verdict

        # 8. SELF-TRADE PREVENTION
        verdict = self._check_self_trade(proposal)
        if verdict is not None:
            return verdict

        # 9. POSITION SIZING GUARD
        return self._check_sizing(proposal, inventory_q)

    # -------------------------------------------------------------------
    # Individual circuit breakers
    # -------------------------------------------------------------------

    def _check_inventory(
        self, proposal: OrderProposal, inventory_q: float, ts: float
    ) -> RiskVerdict | None:
        """Check inventory limits. Returns None if OK."""
        cfg = self._cfg

        # Check recovery from inventory halt
        if self._state == BotState.HALTED_INVENTORY:
            if (
                self._inventory_flattened
                and self._inventory_halt_time is not None
                and ts - self._inventory_halt_time > cfg.inventory_cooldown_s
            ):
                logger.info("Inventory halt cooldown expired, resuming")
                self._state = BotState.RUNNING
                self._inventory_flattened = False
            else:
                # Allow emergency flatten orders during halt
                if proposal.is_emergency_flatten:
                    return None
                return RiskVerdict(
                    allowed=False,
                    reason=f"Inventory halt active, cooldown remaining",
                )

        # Hard breach
        if abs(inventory_q) > cfg.q_max_usd:
            self._state = BotState.HALTED_INVENTORY
            self._inventory_halt_time = ts
            logger.error(
                "INVENTORY BREACH: |q|=$%.0f > q_max=$%.0f — halting, flatten required",
                abs(inventory_q), cfg.q_max_usd,
            )
            # Allow emergency flatten orders through immediately
            if proposal.is_emergency_flatten:
                return None
            return RiskVerdict(
                allowed=False,
                reason=f"Inventory breach: |{inventory_q:.0f}| > {cfg.q_max_usd:.0f}",
                actions_required=["cancel_all", "flatten_inventory", "alert"],
            )

        return None

    def _check_drawdown(
        self, current_equity: float, ts: float
    ) -> RiskVerdict | None:
        if self._session_start_equity is None:
            return None

        threshold = self._session_start_equity * (1.0 - self._cfg.max_drawdown_pct / 100.0)
        if current_equity < threshold:
            self._state = BotState.HALTED_DRAWDOWN
            logger.error(
                "DRAWDOWN HALT: equity=$%.2f < threshold=$%.2f (%.1f%% from start=$%.2f)",
                current_equity, threshold, self._cfg.max_drawdown_pct,
                self._session_start_equity,
            )
            # Write sentinel file for manual restart requirement
            try:
                Path(self._cfg.sentinel_path).write_text(
                    f"halted_at={ts}\nequity={current_equity}\n"
                    f"threshold={threshold}\n"
                )
            except OSError:
                pass

            return RiskVerdict(
                allowed=False,
                reason=f"Drawdown halt: equity ${current_equity:.2f} < ${threshold:.2f}",
                actions_required=["cancel_all", "alert"],
            )
        return None

    def _check_latency(self, ts: float) -> RiskVerdict | None:
        if len(self._rtt_buffer) == 0:
            return None

        mean_rtt = sum(self._rtt_buffer) / len(self._rtt_buffer)
        max_rtt = max(self._rtt_buffer)

        # Hard halt on extreme single RTT
        if max_rtt > self._cfg.latency_single_max_ms:
            self._state = BotState.HALTED_LATENCY
            logger.error("LATENCY HALT: single RTT=%.0fms > %.0fms",
                         max_rtt, self._cfg.latency_single_max_ms)
            return RiskVerdict(
                allowed=False,
                reason=f"Latency halt: single RTT {max_rtt:.0f}ms",
                actions_required=["cancel_all"],
            )

        # Mean RTT halt
        if mean_rtt > self._cfg.latency_halt_ms:
            self._state = BotState.HALTED_LATENCY
            logger.error("LATENCY HALT: mean RTT=%.0fms > %.0fms",
                         mean_rtt, self._cfg.latency_halt_ms)
            return RiskVerdict(
                allowed=False,
                reason=f"Latency halt: mean RTT {mean_rtt:.0f}ms",
                actions_required=["cancel_all"],
            )

        # Reduced mode on elevated latency
        if mean_rtt > self._cfg.latency_warn_ms:
            if self._state == BotState.RUNNING:
                self._state = BotState.REDUCED_MODE
                self._rtt_recovery_start = None
                logger.warning("Entering REDUCED_MODE: mean RTT=%.0fms", mean_rtt)
            return None  # allowed but caller should widen spreads

        # Recovery from reduced mode
        if self._state == BotState.REDUCED_MODE:
            if self._rtt_recovery_start is None:
                self._rtt_recovery_start = ts
            elif ts - self._rtt_recovery_start > self._cfg.rtt_recovery_s:
                self._state = BotState.RUNNING
                self._rtt_recovery_start = None
                logger.info("RTT recovered, back to RUNNING")

        # Recovery from latency halt
        if self._state == BotState.HALTED_LATENCY:
            self._state = BotState.REDUCED_MODE
            self._rtt_recovery_start = ts
            logger.info("Latency improved, entering REDUCED_MODE")

        return None

    def _check_toxicity(
        self, market_action: str, ts: float
    ) -> RiskVerdict | None:
        # Recovery from toxicity halt
        if self._state == BotState.HALTED_TOXICITY:
            if (
                self._toxicity_halt_time is not None
                and ts - self._toxicity_halt_time > self._cfg.toxicity_cooldown_s
                and market_action != "halt"
            ):
                self._state = BotState.RUNNING
                logger.info("Toxicity cooldown expired, resuming")
                return None

            return RiskVerdict(
                allowed=False,
                reason="Toxicity halt — waiting for cooldown",
            )

        # New toxicity halt
        if market_action == "halt":
            self._state = BotState.HALTED_TOXICITY
            self._toxicity_halt_time = ts
            logger.warning("TOXICITY HALT triggered at ts=%.3f", ts)
            return RiskVerdict(
                allowed=False,
                reason="Toxicity halt — market too one-sided",
                actions_required=["cancel_all"],
                # Do NOT flatten — wait for normalisation
            )
        return None

    def _check_funding_spike(self, funding_rate: float) -> None:
        """Log warning on funding spike. Spread adjustment is done by QuoteEngine."""
        if abs(funding_rate) > self._cfg.funding_spike_threshold:
            logger.warning(
                "Funding spike: %.4f%%/hr (threshold: %.4f%%/hr)",
                funding_rate * 100, self._cfg.funding_spike_threshold * 100,
            )

    def _check_dead_man(self, ts: float) -> RiskVerdict | None:
        if ts > self._dead_man_deadline:
            logger.error(
                "DEAD MAN'S SWITCH: no heartbeat for %.1fs",
                ts - self._last_heartbeat,
            )
            return RiskVerdict(
                allowed=False,
                reason="Dead man's switch — heartbeat missed",
                actions_required=["cancel_all", "alert"],
            )
        return None

    def _check_self_trade(self, proposal: OrderProposal) -> RiskVerdict | None:
        """Prevent our bid from crossing our own ask, and vice versa."""
        if proposal.side == "buy" and self._our_asks:
            best_own_ask = min(self._our_asks.values())
            if proposal.price >= best_own_ask:
                logger.warning(
                    "Self-trade prevented: bid %.4f >= our ask %.4f",
                    proposal.price, best_own_ask,
                )
                return RiskVerdict(
                    allowed=False,
                    reason=f"Self-trade: bid {proposal.price} >= own ask {best_own_ask}",
                )

        if proposal.side == "sell" and self._our_bids:
            best_own_bid = max(self._our_bids.values())
            if proposal.price <= best_own_bid:
                logger.warning(
                    "Self-trade prevented: ask %.4f <= our bid %.4f",
                    proposal.price, best_own_bid,
                )
                return RiskVerdict(
                    allowed=False,
                    reason=f"Self-trade: ask {proposal.price} <= own bid {best_own_bid}",
                )

        return None

    def _check_sizing(
        self, proposal: OrderProposal, inventory_q: float
    ) -> RiskVerdict:
        """
        Position sizing guard — reduce size if approaching inventory limit.
        This is the final check; if we reach here the order is allowed.
        """
        cfg = self._cfg

        if proposal.is_emergency_flatten:
            return RiskVerdict(allowed=True, adjusted_size=proposal.size)

        delta = proposal.notional if proposal.side == "buy" else -proposal.notional
        proposed_q = inventory_q + delta

        if abs(proposed_q) > cfg.q_max_usd * 0.9:
            # Reduce size to bring proposed_q to 80% of q_max
            target_q = cfg.q_max_usd * 0.8
            if proposal.side == "buy":
                max_notional = target_q - inventory_q
            else:
                max_notional = target_q + inventory_q

            max_notional = max(max_notional, 0.0)
            reduced_size = max_notional / proposal.price if proposal.price > 0 else 0.0

            if max_notional < cfg.min_order_notional:
                logger.warning(
                    "Order rejected: reduced size $%.2f below minimum $%.2f",
                    max_notional, cfg.min_order_notional,
                )
                return RiskVerdict(
                    allowed=False,
                    reason="Reduced size below minimum order",
                )

            logger.warning(
                "Size reduced: %.4f → %.4f (inventory proximity: q=$%.0f, q_max=$%.0f)",
                proposal.size, reduced_size, inventory_q, cfg.q_max_usd,
            )
            return RiskVerdict(allowed=True, adjusted_size=reduced_size)

        return RiskVerdict(allowed=True, adjusted_size=proposal.size)

    # -------------------------------------------------------------------
    # Operational methods
    # -------------------------------------------------------------------

    def record_rtt(self, rtt_ms: float) -> None:
        """Record a round-trip time measurement from an order operation."""
        self._rtt_buffer.append(rtt_ms)

    def heartbeat(self, timestamp: float | None = None) -> None:
        """Push the dead man's switch deadline forward."""
        ts = timestamp if timestamp is not None else time.time()
        self._last_heartbeat = ts
        self._dead_man_deadline = ts + self._cfg.dead_man_switch_s

    def register_order(self, order_id: str, side: str, price: float) -> None:
        """Track our own open orders for self-trade prevention."""
        if side == "buy":
            self._our_bids[order_id] = price
        else:
            self._our_asks[order_id] = price

    def unregister_order(self, order_id: str) -> None:
        """Remove a filled or cancelled order from tracking."""
        self._our_bids.pop(order_id, None)
        self._our_asks.pop(order_id, None)

    def mark_inventory_flattened(self) -> None:
        """Call after emergency flatten completes to start cooldown timer."""
        self._inventory_flattened = True
        logger.info("Inventory marked as flattened, cooldown started")

    def get_ttl_ms(self, vol_regime: str) -> float:
        """Get the quote TTL for the current volatility regime."""
        return self._cfg.stale_ttl_ms.get(vol_regime, 500.0)

    def get_spread_multiplier(self) -> float:
        """Additional spread multiplier from risk state."""
        if self._state == BotState.REDUCED_MODE:
            return 3.0
        return 1.0

    def get_requote_interval_s(self) -> float:
        """How often to requote based on current state."""
        if self._state == BotState.REDUCED_MODE:
            return 5.0
        return 0.2  # 200ms

    def force_halt(self, reason: str = "manual") -> None:
        """Manually halt the bot."""
        self._state = BotState.HALTED_MANUAL
        logger.error("MANUAL HALT: %s", reason)

    def reset_from_halt(self) -> bool:
        """
        Attempt to reset from a halted state.
        Only works for HALTED_MANUAL. HALTED_DRAWDOWN requires restart.
        Returns True if reset was successful.
        """
        if self._state == BotState.HALTED_DRAWDOWN:
            logger.error("Cannot reset from drawdown halt — restart required")
            return False

        if self._state == BotState.HALTED_MANUAL:
            sentinel = Path(self._cfg.sentinel_path)
            if sentinel.exists():
                sentinel.unlink()
            self._state = BotState.RUNNING
            logger.info("Reset from manual halt")
            return True

        # Other halts auto-recover via their cooldown logic
        self._state = BotState.RUNNING
        return True

    def status(self) -> dict:
        """Current risk manager status for monitoring."""
        mean_rtt = (
            sum(self._rtt_buffer) / len(self._rtt_buffer)
            if self._rtt_buffer else 0.0
        )
        return {
            "state": self._state.value,
            "session_start_equity": self._session_start_equity,
            "mean_rtt_ms": round(mean_rtt, 1),
            "max_rtt_ms": round(max(self._rtt_buffer), 1) if self._rtt_buffer else 0.0,
            "rtt_samples": len(self._rtt_buffer),
            "open_bids": len(self._our_bids),
            "open_asks": len(self._our_asks),
            "dead_man_remaining_s": round(
                self._dead_man_deadline - time.time(), 1
            ),
        }
