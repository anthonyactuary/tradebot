"""End-to-end BTC 15m trade signal (poll -> model -> edge -> decision).

This module ties together:
- Kalshi market polling (`kalshi_market_poll.poll_once`)
- Coinbase recent 1m candles (via `btc15m_live_inference.fetch_recent_1m_closes`)
- ML inference (`btc15m_live_inference.build_feature_dict` + `predict_probability`)
- Fees (from current YES/NO asks)
- Edge + decision (`trade_edge_calculation`)

It intentionally does NOT place orders.

Usage (one-shot):
    python -m tradebot.tools.btc15m_trade_signal \
    --model-dir src/tradebot/data/btc15m_model_coinbase_80d_purged30

Usage (live loop):
    python -m tradebot.tools.btc15m_trade_signal \
    --model-dir src/tradebot/data/btc15m_model_coinbase_80d_purged30 --live
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any, Literal

from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient
from tradebot.tools.kalshi_market_poll import MarketSnapshot, poll_once
from tradebot.tools.trade_edge_calculation import EdgeResult, TradeDecision, trade_decision, trade_edge_calculation

from tradebot.tools.kalshi_fees import fees_from_snapshot
from tradebot.tools.position_size import PositionSize, calc_position_size

# Reuse the exact feature engineering + model loader used in live inference.
from tradebot.tools.btc15m_live_inference import (
    load_model,
    build_feature_dict,
    fetch_recent_1m_closes,
    predict_probability,
    choose_proxy_spot_usd,
)


FeeMode = Literal["maker", "taker"]


@dataclass(frozen=True)
class TradeSignal:
    poll_utc_iso: str
    ticker: str

    price_to_beat: float | None
    btc_spot_usd: float | None
    proxy_spot_usd: float | None
    spot_source: str | None
    seconds_to_expiry: int | None

    market_p_yes: float | None
    market_p_no: float | None

    contracts: int
    fee_mode: FeeMode
    fee_yes_usd_per_contract: float | None
    fee_no_usd_per_contract: float | None

    features: dict[str, float] | None
    p_yes: float | None
    p_no: float | None

    edge: EdgeResult | None
    decision: TradeDecision | None

    position: PositionSize | None

    error: str | None = None


def _best_ev_after_fees(edge: EdgeResult) -> tuple[Literal["YES", "NO"], float]:
    """Return (side, ev_after_fees_usd) for the better of YES/NO."""
    if edge.ev_yes_after_fees >= edge.ev_no_after_fees:
        return ("YES", float(edge.ev_yes_after_fees))
    return ("NO", float(edge.ev_no_after_fees))


def _to_clean_log_row(s: TradeSignal) -> dict[str, Any]:
    """Minimal per-market output for human scanning."""
    out: dict[str, Any] = {
        "ticker": s.ticker,
        "seconds_to_expiry": s.seconds_to_expiry,
        "spot_source": s.spot_source,
        "proxy_spot_usd": s.proxy_spot_usd,
        "fee_mode": s.fee_mode,
        "fee_yes_usd_per_contract": s.fee_yes_usd_per_contract,
        "fee_no_usd_per_contract": s.fee_no_usd_per_contract,
        "p_up": s.p_yes,
        "market_p_up": s.market_p_yes,
        "ev_after_fees_pct": None,
        "decision": None,
        "kelly_fraction": None,
        "contracts": None,
        "error": s.error,
    }

    if s.decision is not None:
        out["decision"] = s.decision.side

    if s.edge is not None:
        _side, best_ev = _best_ev_after_fees(s.edge)
        out["ev_after_fees_pct"] = float(best_ev) * 100.0

    if s.position is not None:
        out["kelly_fraction"] = float(s.position.kelly_fraction)
        out["contracts"] = s.position.contracts

    return out


def _fmt_pct(v: float | None, *, ndp: int = 2) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v) * 100.0:.{int(ndp)}f}%"
    except Exception:
        return "-"


def _fmt_ev_pct(v: float | None, *, ndp: int = 2) -> str:
    """EV as a percent of $1 payout (so 0.027 -> 2.70%)."""
    if v is None:
        return "-"
    try:
        return f"{float(v) * 100.0:.{int(ndp)}f}%"
    except Exception:
        return "-"


def _fmt_usd(v: float | None, *, ndp: int = 2) -> str:
    if v is None:
        return "-"
    try:
        return f"${float(v):.{int(ndp)}f}"
    except Exception:
        return "-"


def _to_pretty_line(s: TradeSignal) -> str:
    p_up = s.p_yes
    mkt_up = s.market_p_yes

    ev_best_after: float | None = None
    ev_side: str | None = None
    if s.edge is not None:
        ev_side, ev_best_after = _best_ev_after_fees(s.edge)

    decision = s.decision.side if s.decision is not None else None
    decision_str = decision or "-"
    ev_side_str = ev_side or "-"

    fee_yes_pc = s.fee_yes_usd_per_contract
    fee_no_pc = s.fee_no_usd_per_contract
    fee_pc: float | None = None
    if decision == "YES":
        fee_pc = fee_yes_pc
    elif decision == "NO":
        fee_pc = fee_no_pc

    # Example:
    # 02:15:13Z  KXBTC15M-...  tte=14m58s  p_up=96.74%  mkt=85.00%  EV_yes=2.74%  EV_no=-3.10%  decision=YES
    ts = "-"
    try:
        ts_dt = dt.datetime.fromisoformat(s.poll_utc_iso)
        ts = ts_dt.strftime("%H:%M:%SZ")
    except Exception:
        pass

    parts = [
        ts,
        s.ticker,
        f"tte={_fmt_tte(s.seconds_to_expiry)}",
        f"spot={_fmt_usd(s.proxy_spot_usd, ndp=2)}",
        f"strike={_fmt_usd(s.price_to_beat, ndp=2)}",
        f"spot_src={s.spot_source or '-'}",
        f"fee_mode={s.fee_mode}",
        f"fee/c={_fmt_usd(fee_pc, ndp=4) if fee_pc is not None else '-'}",
        f"p_up={_fmt_pct(p_up)}",
        f"mkt={_fmt_pct(mkt_up)}",
        f"EV_yes={_fmt_ev_pct(s.edge.ev_yes_after_fees) if s.edge is not None else '-'}",
        f"EV_no={_fmt_ev_pct(s.edge.ev_no_after_fees) if s.edge is not None else '-'}",
        f"decision={decision_str}",
        f"kelly={_fmt_pct(s.position.kelly_fraction) if s.position is not None else '-'}",
        f"qty={s.position.contracts if (s.position is not None and s.position.contracts is not None) else '-'}",
    ]
    if s.error:
        parts.append(f"error={s.error}")
    return "  ".join(parts)


def _fmt_tte(seconds_to_expiry: int | None) -> str:
    if seconds_to_expiry is None:
        return "-"
    try:
        total = int(seconds_to_expiry)
    except Exception:
        return "-"
    if total < 0:
        total = 0
    m, s = divmod(total, 60)
    return f"{m}m{s:02d}s"


def _pick_fee_per_contract(
    fees: dict[str, float | None],
    *,
    side: Literal["YES", "NO"],
    fee_mode: FeeMode,
    contracts: int,
) -> float | None:
    if fee_mode == "maker":
        key = "maker_fee_yes_usd" if side == "YES" else "maker_fee_no_usd"
    else:
        key = "taker_fee_yes_usd" if side == "YES" else "taker_fee_no_usd"
    v = fees.get(key)
    if v is None:
        return None
    # fees_from_snapshot() returns TOTAL fee for `contracts`.
    # We want a per-contract value for EV and sizing math.
    c = int(contracts)
    denom = max(1, c)
    return float(v) / float(denom)


async def signal_for_snapshot(
    model: Any,
    feature_names: list[str],
    *,
    snap: MarketSnapshot,
    recent_closes_1m: list[float],
    contracts: int,
    fee_mode: FeeMode,
    threshold: float,
    bankroll_usd: float | None,
    kelly_multiplier: float,
    kelly_max_fraction: float,
    use_kraken_composite: bool = True,
) -> TradeSignal:
    try:
        if snap.price_to_beat is None:
            raise ValueError("missing price_to_beat")
        if snap.seconds_to_expiry is None:
            raise ValueError("missing seconds_to_expiry")
        if snap.market_p_yes is None or snap.market_p_no is None:
            raise ValueError("missing market_p_yes/market_p_no")

        proxy_spot_usd, spot_source = choose_proxy_spot_usd(snap, use_kraken_composite=bool(use_kraken_composite))
        if proxy_spot_usd is None:
            return TradeSignal(
                poll_utc_iso=snap.poll_utc_iso,
                ticker=snap.ticker,
                price_to_beat=snap.price_to_beat,
                btc_spot_usd=snap.btc_spot_usd,
                proxy_spot_usd=None,
                spot_source=str(spot_source),
                seconds_to_expiry=snap.seconds_to_expiry,
                market_p_yes=snap.market_p_yes,
                market_p_no=snap.market_p_no,
                contracts=int(contracts),
                fee_mode=fee_mode,
                fee_yes_usd_per_contract=None,
                fee_no_usd_per_contract=None,
                features=None,
                p_yes=None,
                p_no=None,
                edge=None,
                decision=None,
                position=None,
                error=f"skip:missing_spot spot_source={spot_source}",
            )

        feats = build_feature_dict(
            price_to_beat=float(snap.price_to_beat),
            btc_spot_usd=float(proxy_spot_usd),
            seconds_to_expiry=int(snap.seconds_to_expiry),
            recent_closes_1m=recent_closes_1m,
        )

        p_yes = float(predict_probability(model, feature_names, feats))
        p_no = 1.0 - p_yes

        fees = fees_from_snapshot(snap=snap, contracts=int(contracts))
        fee_yes = _pick_fee_per_contract(fees, side="YES", fee_mode=fee_mode, contracts=int(contracts))
        fee_no = _pick_fee_per_contract(fees, side="NO", fee_mode=fee_mode, contracts=int(contracts))

        edge = trade_edge_calculation(
            p_yes=p_yes,
            p_no=p_no,
            market_p_yes=float(snap.market_p_yes),
            market_p_no=float(snap.market_p_no),
            fee_yes_usd_per_contract=float(fee_yes or 0.0),
            fee_no_usd_per_contract=float(fee_no or 0.0),
        )
        decision = trade_decision(edge=edge, threshold=float(threshold))

        position: PositionSize | None = None
        if decision.side is not None:
            if decision.side == "YES":
                price = float(snap.market_p_yes)
                p_win = float(p_yes)
                fee = float(fee_yes or 0.0)
            else:
                price = float(snap.market_p_no)
                p_win = float(p_no)
                fee = float(fee_no or 0.0)

            position = calc_position_size(
                side=str(decision.side),
                p_win=float(p_win),
                price=float(price),
                bankroll_usd=bankroll_usd,
                kelly_multiplier=float(kelly_multiplier),
                fee_per_contract_usd=float(fee),
                max_fraction=float(kelly_max_fraction),
                min_contracts_if_positive_edge=1,
            )

        return TradeSignal(
            poll_utc_iso=snap.poll_utc_iso,
            ticker=snap.ticker,
            price_to_beat=snap.price_to_beat,
            btc_spot_usd=snap.btc_spot_usd,
            proxy_spot_usd=float(proxy_spot_usd),
            spot_source=str(spot_source),
            seconds_to_expiry=snap.seconds_to_expiry,
            market_p_yes=snap.market_p_yes,
            market_p_no=snap.market_p_no,
            contracts=int(contracts),
            fee_mode=fee_mode,
            fee_yes_usd_per_contract=fee_yes,
            fee_no_usd_per_contract=fee_no,
            features=feats,
            p_yes=p_yes,
            p_no=p_no,
            edge=edge,
            decision=decision,
            position=position,
            error=None,
        )
    except Exception as e:
        return TradeSignal(
            poll_utc_iso=getattr(snap, "poll_utc_iso", dt.datetime.now(dt.timezone.utc).isoformat()),
            ticker=getattr(snap, "ticker", ""),
            price_to_beat=getattr(snap, "price_to_beat", None),
            btc_spot_usd=getattr(snap, "btc_spot_usd", None),
            proxy_spot_usd=None,
            spot_source=None,
            seconds_to_expiry=getattr(snap, "seconds_to_expiry", None),
            market_p_yes=getattr(snap, "market_p_yes", None),
            market_p_no=getattr(snap, "market_p_no", None),
            contracts=int(contracts),
            fee_mode=fee_mode,
            fee_yes_usd_per_contract=None,
            fee_no_usd_per_contract=None,
            features=None,
            p_yes=None,
            p_no=None,
            edge=None,
            decision=None,
            position=None,
            error=str(e),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC 15m trade signal (Kalshi + Coinbase + ML + edge)")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing model.json and meta.json from training",
    )
    parser.add_argument("--live", action="store_true", help="Continuously poll and print signals")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--asset", type=str, default="BTC")
    parser.add_argument("--horizon-minutes", type=int, default=60)
    parser.add_argument("--limit-markets", type=int, default=2)
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument(
        "--bankroll-usd",
        type=float,
        default=None,
        help="Bankroll used to convert Kelly fraction into a contract quantity (optional)",
    )
    parser.add_argument(
        "--kelly-mult",
        type=float,
        default=0.25,
        help="Fractional Kelly multiplier (1.0 = full Kelly)",
    )
    parser.add_argument(
        "--kelly-max-fraction",
        type=float,
        default=0.25,
        help="Cap Kelly fraction to this maximum",
    )
    parser.add_argument("--fee-mode", type=str, choices=["maker", "taker"], default="taker")
    parser.add_argument(
        "--output",
        type=str,
        choices=["pretty", "clean", "json"],
        default="pretty",
        help="Output format: 'pretty' prints one line per market; 'clean' prints minimal JSON; 'json' prints full payload",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Require fee-adjusted EV above this threshold to signal a trade (0.0 = positive EV after fees)",
    )
    args = parser.parse_args()

    model, feature_names = load_model(str(args.model_dir))

    async def runner() -> None:
        settings = Settings.load()
        client = KalshiClient.from_settings(settings)
        try:
            async def one_poll() -> list[TradeSignal]:
                snaps = await poll_once(
                    client,
                    asset=str(args.asset),
                    horizon_minutes=int(args.horizon_minutes),
                    limit_markets=int(args.limit_markets),
                )

                # One Coinbase call per poll (not per market)
                closes = await fetch_recent_1m_closes(limit=6)

                return [
                    await signal_for_snapshot(
                        model,
                        feature_names,
                        snap=s,
                        recent_closes_1m=closes,
                        contracts=int(args.contracts),
                        fee_mode=str(args.fee_mode),
                        threshold=float(args.threshold),
                        bankroll_usd=(float(args.bankroll_usd) if args.bankroll_usd is not None else None),
                        kelly_multiplier=float(args.kelly_mult),
                        kelly_max_fraction=float(args.kelly_max_fraction),
                    )
                    for s in snaps
                ]

            if not args.live:
                signals = await one_poll()
                out = str(args.output)
                if out == "json":
                    print(json.dumps([asdict(s) for s in signals], indent=2, sort_keys=True))
                elif out == "clean":
                    print(json.dumps([_to_clean_log_row(s) for s in signals], indent=2, sort_keys=True))
                else:
                    for s in signals:
                        print(_to_pretty_line(s))
                return

            end = dt.datetime.now().timestamp() + float(args.duration_seconds)
            while dt.datetime.now().timestamp() < end:
                signals = await one_poll()
                out = str(args.output)
                if out == "json":
                    print(json.dumps([asdict(s) for s in signals], sort_keys=True))
                elif out == "clean":
                    print(json.dumps([_to_clean_log_row(s) for s in signals], sort_keys=True))
                else:
                    for s in signals:
                        print(_to_pretty_line(s))
                await asyncio.sleep(max(0.1, float(args.poll_seconds)))
        finally:
            await client.aclose()

    asyncio.run(runner())


if __name__ == "__main__":
    main()
