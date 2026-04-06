"""
MM Bot — entry point.

Orchestrates the full market making loop:
  1. Load config
  2. Connect to exchange
  3. Start signal + quote engines
  4. Run the quote → risk check → place loop
  5. Monitor and log
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import yaml

from MM_algo.adapters.hyperliquid import HyperliquidAdapter
from MM_algo.core.fee_engine import FeeEngine, FeeConfig
from MM_algo.core.quote_engine import QuoteEngine, QuoteEngineConfig
from MM_algo.core.risk_manager import RiskManager, RiskConfig, OrderProposal, BotState

logger = logging.getLogger("mm_bot")


def load_config(path: str = "MM_algo/config/config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Resolve env vars
    for section in cfg.values():
        if isinstance(section, dict):
            for k, v in section.items():
                if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                    env_key = v[2:-1]
                    section[k] = os.environ.get(env_key, "")
    return cfg


async def run_bot(config: dict) -> None:
    """Main async event loop for the market maker."""
    # Build components
    fees_cfg = config.get("fees", {})
    fee_engine = FeeEngine(FeeConfig(
        volume_tier=fees_cfg.get("volume_tier", 0),
        staking_tier=fees_cfg.get("staking_tier", "none"),
        builder_fee_bps=fees_cfg.get("builder_fee_bps", 0),
        maker_volume_share_pct=fees_cfg.get("maker_volume_share_pct", 0),
    ))

    pricing_cfg = config.get("pricing", {})
    quote_engine = QuoteEngine(
        config=QuoteEngineConfig(
            gamma=pricing_cfg.get("gamma", 0.1),
            lambda_skew=pricing_cfg.get("lambda_skew", 0.5),
            ofi_alpha=pricing_cfg.get("ofi_alpha", 0.05),
            min_spread_bps=pricing_cfg.get("min_spread_bps", 2.0),
            q_max=config.get("risk", {}).get("q_max_usd", 50_000),
            available_capital=100_000,  # overwritten after equity fetch below
        ),
        fee_engine=fee_engine,
    )

    risk_cfg = config.get("risk", {})
    risk_mgr = RiskManager(RiskConfig(
        q_max_usd=risk_cfg.get("q_max_usd", 50_000),
        max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 3.0),
        latency_warn_ms=risk_cfg.get("latency_warn_ms", 200),
        latency_halt_ms=risk_cfg.get("latency_halt_ms", 1000),
    ))

    exchange_cfg = config.get("exchange", {})
    adapter = HyperliquidAdapter(
        private_key=exchange_cfg.get("private_key", ""),
        builder_code=exchange_cfg.get("builder_code", ""),
        network=exchange_cfg.get("network", "testnet"),
        simulate=exchange_cfg.get("simulate", True),
    )

    await adapter.connect()
    equity = await adapter.get_equity()
    risk_mgr.set_session_equity(equity)
    quote_engine._cfg.available_capital = equity

    # Enable self-trade prevention
    await adapter.set_self_trade_prevention("expire_maker")

    # Schedule dead man's switch
    await adapter.schedule_cancel_all("SOL", delay_s=30)

    markets = config.get("markets", [])
    symbol = markets[0]["symbol"] if markets else "SOL"

    logger.info("Bot starting — symbol=%s, network=%s, simulate=True", symbol, exchange_cfg.get("network"))
    logger.info("\n%s", fee_engine.fee_report())

    # Main loop
    tick = 0
    while risk_mgr.state not in (BotState.HALTED_DRAWDOWN, BotState.HALTED_MANUAL):
        try:
            risk_mgr.heartbeat()
            await adapter.schedule_cancel_all(symbol, delay_s=30)

            # Get current state
            inventory_q = await adapter.get_position(symbol)
            equity = await adapter.get_equity()
            funding = await adapter.get_funding_rate(symbol)
            quote_engine.update_funding_rate(funding)

            # Generate quotes
            decision = quote_engine.generate_quotes(inventory_q)

            if decision.recommended_action == "halt":
                logger.warning("Quote engine recommends HALT — cancelling all")
                await adapter.cancel_all_orders(symbol)
                await asyncio.sleep(5)
                continue

            # Risk check for bid
            if decision.bid_size > 0 and decision.recommended_action in ("quote_both", "quote_bid_only"):
                bid_proposal = OrderProposal(
                    side="buy", price=decision.bid_quote,
                    size=decision.bid_size,
                    notional=decision.bid_quote * decision.bid_size,
                )
                bid_verdict = risk_mgr.check_order(
                    bid_proposal, equity, inventory_q,
                    market_state_action=decision.recommended_action,
                    funding_rate=funding,
                )
            else:
                bid_verdict = None

            # Risk check for ask
            if decision.ask_size > 0 and decision.recommended_action in ("quote_both", "quote_ask_only"):
                ask_proposal = OrderProposal(
                    side="sell", price=decision.ask_quote,
                    size=decision.ask_size,
                    notional=decision.ask_quote * decision.ask_size,
                )
                ask_verdict = risk_mgr.check_order(
                    ask_proposal, equity, inventory_q,
                    market_state_action=decision.recommended_action,
                )
            else:
                ask_verdict = None

            # Execute: cancel then place
            bid_px = decision.bid_quote if (bid_verdict and bid_verdict.allowed) else 0
            bid_sz = (bid_verdict.adjusted_size or decision.bid_size) if (bid_verdict and bid_verdict.allowed) else 0
            ask_px = decision.ask_quote if (ask_verdict and ask_verdict.allowed) else 0
            ask_sz = (ask_verdict.adjusted_size or decision.ask_size) if (ask_verdict and ask_verdict.allowed) else 0

            if bid_sz > 0 or ask_sz > 0:
                cancel_res, place_res = await adapter.requote(
                    symbol, bid_px, bid_sz, ask_px, ask_sz,
                )
                for r in place_res:
                    risk_mgr.record_rtt(r.rtt_ms)

            tick += 1
            if tick % 50 == 0:
                logger.info(
                    "tick=%d mid=%.2f spread=%.1fbps inv=$%.0f action=%s",
                    tick, decision.mid_price, decision.spread_bps,
                    inventory_q, decision.recommended_action,
                )

            await asyncio.sleep(risk_mgr.get_requote_interval_s())

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down")
            break
        except Exception as e:
            logger.error("Main loop error: %s", e, exc_info=True)
            await adapter.cancel_all_orders(symbol)
            await asyncio.sleep(5)

    # Cleanup
    await adapter.cancel_all_orders(symbol)
    await adapter.disconnect()
    logger.info("Bot shutdown complete")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    config = load_config()
    asyncio.run(run_bot(config))


if __name__ == "__main__":
    main()
