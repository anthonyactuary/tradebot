"""Run the BTC 15m Kalshi trading process end-to-end.

This is the single entrypoint you run. All tunables live in the CONFIG block.

What it does per poll:
- Poll Kalshi for active BTC 15m markets (expiring soon)
- Fetch recent Coinbase 1m candles for features
- Run ML inference (p_up)
- Compute fee-adjusted edge + decision
- Size via Kelly (optional bankroll)
- Enforce inventory / exposure limits
- Place (or dry-run) an order

Run:
    # From the tradebot/ folder:
    .\\.venv\\Scripts\\python.exe -m tradebot.tools.run_btc15m_trader

    # Or from the repo root:
    .\\tradebot\\.venv\\Scripts\\python.exe -m tradebot.tools.run_btc15m_trader
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import sys
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

# httpx is very chatty at INFO (it logs every request). Keep our app logs at INFO,
# but silence request spam unless there's a warning/error.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Allow running this file directly ("python path/to/run_btc15m_trader.py") in a
# src-layout repo by ensuring the src/ root is on sys.path.
_THIS_FILE = Path(__file__).resolve()
_SRC_ROOT = _THIS_FILE.parents[2]  # .../src
if (_SRC_ROOT / "tradebot").is_dir():
    src_root_str = str(_SRC_ROOT)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient
from tradebot.tools.btc15m_live_inference import fetch_recent_1m_closes, load_model
from tradebot.tools.btc15m_trade_signal import TradeSignal, _to_pretty_line, signal_for_snapshot
from tradebot.tools.inventory_check import fetch_inventory_summary
from tradebot.tools.csv_logging import default_live_csv_log_path, install_csv_log_handler
from tradebot.tools.order_execution import EntryMode, RiskLimits, place_order
from tradebot.tools.kalshi_market_poll import MarketSnapshot, poll_once


log = logging.getLogger(__name__)

FeeMode = Literal["maker", "taker"]


@dataclass(frozen=True)
class TraderConfig:
    # Model
    model_dir: str = "src/tradebot/data/btc15m_model_coinbase_83d_purged30"

    # Polling
    live: bool = True
    poll_seconds: float = 5.0
    duration_seconds: float = 0.0  # 0 = run forever
    asset: str = "BTC"
    horizon_minutes: int = 60
    limit_markets: int = 2
    min_seconds_to_expiry: int = 30

    # Signal
    fee_mode: FeeMode = "taker"
    threshold: float = 0.0  # fee-adjusted EV threshold

    # Position sizing
    use_kalshi_balance_as_bankroll: bool = True
    bankroll_usd: float | None = None
    kelly_mult: float = 0.1
    kelly_max_fraction: float = 0.1

    # Execution
    enable_execution: bool = True
    dry_run: bool = False
    max_seconds_to_expiry_to_trade: int | None = 840
    time_in_force: Literal["fill_or_kill", "good_till_canceled", "immediate_or_cancel"] | None = "immediate_or_cancel"
    max_orders_per_poll: int = 1

    # Entry execution mode
    # - taker_ioc: buy at best ask using IoC (default)
    # - maker_only: place GTC maker-style bids with spread/price guards and rate limiting
    entry_mode: EntryMode = "taker_ioc"
    max_entry_spread_cents: int | None = 5
    maker_improve_cents: int = 0
    min_seconds_between_entry_orders: int = 20

    # Logging
    # When enabled, writes all Python logs to a CSV under runs/ while still logging to terminal.
    csv_log_enabled: bool = True
    csv_log_path: str | None = None

    # Execution gates (anti-churn)
    min_hold_seconds: int = 300
    min_entry_edge: float = 0.025  # EV threshold when TTE >= entry_edge_tte_threshold
    min_entry_edge_late: float = 0.05  # EV threshold when TTE < entry_edge_tte_threshold
    entry_edge_tte_threshold: int = 300  # 5 minutes in seconds
    dead_zone: float = 0.05
    exit_delta: float = 0.10
    catastrophic_exit_delta: float = 0.18

    # Data sanity
    # Block entry orders if Kalshi strike is too far from external spot (helps when
    # strike is briefly wrong early in a market).
    spot_strike_sanity_enabled: bool = True
    max_spot_strike_deviation_fraction: float = 0.02

    # Re-entry policy
    # Default behavior is: after flatten, do not re-enter in the same ticker.
    # If enabled, a flip-flatten (decision change) may immediately re-enter the opposite side,
    # and that re-entry position will be held until expiry (no further flawttening).
    allow_reentry_after_flatten: bool = True

    # Risk limits (set to None to disable a specific limit)
    max_total_abs_contracts: int | None = 7.0
    max_total_exposure_usd: float | None = 7.0
    max_ticker_abs_contracts: int | None = 7.0
    max_ticker_exposure_usd: float | None = 7.0

CONFIG = TraderConfig()


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _trade_ev_after_fees(signal: TradeSignal) -> float | None:
    if signal.edge is None or signal.decision is None or signal.decision.side is None:
        return None
    if signal.decision.side == "YES":
        return float(signal.edge.ev_yes_after_fees)
    return float(signal.edge.ev_no_after_fees)


async def _run_once(
    *,
    client: KalshiClient,
    model: object,
    feature_names: list[str],
    cfg: TraderConfig,
    run_id: str,
    poll_id: int,
) -> None:
    try:
        snaps: list[MarketSnapshot] = await poll_once(
            client,
            asset=str(cfg.asset),
            horizon_minutes=int(cfg.horizon_minutes),
            limit_markets=int(cfg.limit_markets),
            min_seconds_to_expiry=int(cfg.min_seconds_to_expiry),
        )
    except Exception as e:
        # Kalshi occasionally returns transient 5xx. Don't let one poll kill the bot.
        log.warning("POLL_ONCE_ERROR error=%s", e)
        return

    if not snaps:
        log.info("No active markets found")
        return

    # One Coinbase call per poll (not per market)
    closes = await fetch_recent_1m_closes(limit=6)

    bankroll_usd: float | None = None
    if cfg.bankroll_usd is not None:
        bankroll_usd = float(cfg.bankroll_usd)
    elif cfg.use_kalshi_balance_as_bankroll:
        try:
            bal = await client.get_balance()
            bal_cents = float(bal.get("balance") or 0)
            pv_cents = float(bal.get("portfolio_value") or 0)
            bankroll_usd = bal_cents / 100.0
            log.info("BALANCE available=$%.2f portfolio_value=$%.2f", bal_cents / 100.0, pv_cents / 100.0)
        except Exception as e:
            log.warning("BALANCE error=%s", e)

    # Fetch inventory once per poll for the tickers we care about.
    tickers = [s.ticker for s in snaps]
    inventory = await fetch_inventory_summary(client=client, tickers=tickers)

    risk_limits = RiskLimits(
        max_total_abs_contracts=cfg.max_total_abs_contracts,
        max_total_exposure_usd=cfg.max_total_exposure_usd,
        max_ticker_abs_contracts=cfg.max_ticker_abs_contracts,
        max_ticker_exposure_usd=cfg.max_ticker_exposure_usd,
    )

    # Compute signals sequentially so logs are readable.
    signals: list[tuple[MarketSnapshot, TradeSignal]] = []
    for s in snaps:
        sig = await signal_for_snapshot(
            model,
            feature_names,
            snap=s,
            recent_closes_1m=closes,
            contracts=1,  # per-contract edge/fee basis
            fee_mode=str(cfg.fee_mode),
            threshold=float(cfg.threshold),
            bankroll_usd=bankroll_usd,
            kelly_multiplier=float(cfg.kelly_mult),
            kelly_max_fraction=float(cfg.kelly_max_fraction),
        )
        signals.append((s, sig))
        # Use logging so it also gets captured by CSV logs.
        log.info(_to_pretty_line(sig))

        # Structured row for later calibration / dataset generation.
        # Keep this separate from the human-readable line above.
        decision_side = sig.decision.side if sig.decision is not None else None
        contracts = sig.position.contracts if (sig.position is not None and sig.position.contracts is not None) else None
        kelly_fraction = float(sig.position.kelly_fraction) if sig.position is not None else None
        log.info(
            "PRED_SNAPSHOT %s",
            s.ticker,
            extra={
                "csv_fields": {
                    "run_id": str(run_id),
                    "poll_id": int(poll_id),
                    "ticker": str(s.ticker),
                    "decision": decision_side,
                    "qty": contracts,
                    "kelly_fraction": kelly_fraction,
                    "spot_usd": sig.proxy_spot_usd,
                    "strike_usd": sig.price_to_beat,
                    "strike_src": (s.price_to_beat_source or sig.spot_source or "?"),
                    "tte_s": sig.seconds_to_expiry,
                    "p_yes": sig.p_yes,
                    "p_no": sig.p_no,
                    "market_p_yes": sig.market_p_yes,
                    "market_p_no": sig.market_p_no,
                    "fee_assumption": str(cfg.fee_mode),
                    "error": sig.error,
                    "error_type": ("signal_error" if sig.error else ""),
                    # Leave additional fields for JSON fallback.
                    "spot_source": sig.spot_source,
                    "coinbase_mid_usd": getattr(s, "coinbase_mid_usd", None),
                    "coinbase_best_bid": getattr(s, "coinbase_best_bid", None),
                    "coinbase_best_ask": getattr(s, "coinbase_best_ask", None),
                    "coinbase_vwap_60s": getattr(s, "coinbase_vwap_60s", None),
                    "coinbase_vwap_count": getattr(s, "coinbase_vwap_count", None),
                    "coinbase_vwap_age_ms": getattr(s, "coinbase_vwap_age_ms", None),
                }
            },
        )

    if not cfg.enable_execution:
        return

    # Pick up to N trades per poll by best EV (after fees) to avoid spamming orders.
    actionable: list[tuple[float, MarketSnapshot, TradeSignal]] = []
    for snap, sig in signals:
        ev = _trade_ev_after_fees(sig)
        if ev is None:
            continue
        if sig.position is None or sig.position.contracts is None or int(sig.position.contracts) <= 0:
            continue
        actionable.append((float(ev), snap, sig))

    actionable.sort(key=lambda x: x[0], reverse=True)

    placed = 0
    for ev, snap, sig in actionable:
        if placed >= max(0, int(cfg.max_orders_per_poll)):
            break

        # place_order() will re-check decision.side and position sizing.
        try:
            out = await place_order(
                client=client,
                snap=snap,
                decision=sig.decision,  # type: ignore[arg-type]
                edge=sig.edge,  # type: ignore[arg-type]
                position=sig.position,
                run_id=str(run_id),
                poll_id=int(poll_id),
                max_seconds_to_expiry=(
                    int(cfg.max_seconds_to_expiry_to_trade)
                    if cfg.max_seconds_to_expiry_to_trade is not None
                    else None
                ),
                min_hold_seconds=int(cfg.min_hold_seconds),
                entry_mode=str(cfg.entry_mode),
                max_entry_spread_cents=(
                    int(cfg.max_entry_spread_cents) if cfg.max_entry_spread_cents is not None else None
                ),
                maker_improve_cents=int(cfg.maker_improve_cents),
                min_seconds_between_entry_orders=int(cfg.min_seconds_between_entry_orders),
                min_entry_edge=float(cfg.min_entry_edge),
                min_entry_edge_late=float(cfg.min_entry_edge_late),
                entry_edge_tte_threshold=int(cfg.entry_edge_tte_threshold),
                dead_zone=float(cfg.dead_zone),
                exit_delta=float(cfg.exit_delta),
                catastrophic_exit_delta=float(cfg.catastrophic_exit_delta),
                allow_reentry_after_flatten=bool(cfg.allow_reentry_after_flatten),
                fee_mode=str(cfg.fee_mode),
                dry_run=bool(cfg.dry_run),
                spot_strike_sanity_enabled=bool(cfg.spot_strike_sanity_enabled),
                max_spot_strike_deviation_fraction=float(cfg.max_spot_strike_deviation_fraction),
                time_in_force=cfg.time_in_force,
                risk_limits=risk_limits,
                risk_tickers=tickers,
                inventory=inventory,
            )
            if out is not None:
                placed += 1
        except Exception as e:
            log.exception("PLACE_ORDER_ERROR %s side=%s error=%s", snap.ticker, str(sig.decision.side), e)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Ensure relative paths (src/, runs/) resolve correctly even when invoked from repo root.
    project_root = Path(__file__).resolve().parents[3]
    os.chdir(project_root)

    # httpx is very chatty at INFO (it logs every request). Keep our app logs at INFO,
    # but silence request spam unless there's a warning/error.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    csv_handler = None
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    if bool(CONFIG.csv_log_enabled):
        path = str(CONFIG.csv_log_path) if CONFIG.csv_log_path else default_live_csv_log_path(prefix="btc15m_live_v2")
        # Ensure runs/ exists if user provided a relative path under it.
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        csv_handler = install_csv_log_handler(path=path, level=logging.INFO)
        log.info("CSV_LOG enabled path=%s run_id=%s", path, str(run_id))

    model, feature_names = load_model(str(CONFIG.model_dir))

    async def runner() -> None:
        settings = Settings.load()
        client = KalshiClient.from_settings(settings)
        try:
            if not CONFIG.live:
                await _run_once(client=client, model=model, feature_names=feature_names, cfg=CONFIG, run_id=str(run_id), poll_id=0)
                return

            start = _utcnow().timestamp()
            poll_id = 0
            while True:
                if CONFIG.duration_seconds and CONFIG.duration_seconds > 0:
                    if (_utcnow().timestamp() - start) >= float(CONFIG.duration_seconds):
                        break

                try:
                    await _run_once(
                        client=client,
                        model=model,
                        feature_names=feature_names,
                        cfg=CONFIG,
                        run_id=str(run_id),
                        poll_id=int(poll_id),
                    )
                except Exception:
                    log.exception("RUN_ONCE_ERROR")
                poll_id += 1
                await asyncio.sleep(max(0.1, float(CONFIG.poll_seconds)))
        finally:
            await client.aclose()

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        log.info("Stopped")
    finally:
        if csv_handler is not None:
            try:
                logging.getLogger().removeHandler(csv_handler)
            except Exception:
                pass
            try:
                csv_handler.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
