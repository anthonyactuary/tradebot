"""Kalshi market polling helpers for ML inference.

This module extracts the Kalshi-specific data we need per poll to later
construct ML model inputs:
- market ticker
- seconds to expiry (market close time)
- strike / "price to beat" (best-effort from market payload)
- best bid/ask (YES + NO) from the Kalshi orderbook

It intentionally does not implement trading logic.

Usage (prints JSON snapshots each poll):
  C:/Users/slump/Tradebot/.venv/Scripts/python.exe -m tradebot.tools.kalshi_market_poll --live
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient
from tradebot.kalshi.orderbook import BestPrices, compute_best_prices


COINBASE_TICKER_URL = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _to_dt(v: Any) -> dt.datetime | None:
    if not v:
        return None
    if isinstance(v, dt.datetime):
        return v
    if isinstance(v, str):
        try:
            return dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception:
            return None
    if isinstance(v, (int, float)):
        try:
            ts = float(v)
            # Heuristic: treat very large values as milliseconds.
            if ts > 1e12:
                ts = ts / 1000.0
            return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
        except Exception:
            return None
    return None


def _market_cutoff_time(market: dict[str, Any]) -> tuple[dt.datetime | None, str]:
    """Return the relevant cutoff time for risk/inference.

    Kalshi market payloads may include multiple timestamps.
    For live inference we care about when trading closes.
    """
    candidates: list[tuple[str, dt.datetime]] = []
    for field in ("close_time", "expiration_time", "expected_expiration_time"):
        parsed = _to_dt(market.get(field))
        if parsed is not None:
            candidates.append((field, parsed))
    if not candidates:
        return (None, "")

    # Use the earliest known cutoff time.
    field, when = min(candidates, key=lambda x: x[1])
    return (when, field)


def _market_price_to_beat(market: dict[str, Any]) -> tuple[float | None, str]:
    """Best-effort extraction of the market's strike / "price to beat" threshold."""

    def _parse_num(v: Any) -> float | None:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            s = s.replace("$", "").replace(",", "")
            try:
                return float(s)
            except Exception:
                return None
        return None

    # Kalshi crypto 15m markets often encode the strike as `floor_strike`.
    for field in (
        "floor_strike",
        "price_to_beat",
        "strike_price",
        "strike",
        "target_price",
        "threshold",
    ):
        parsed = _parse_num(market.get(field))
        if parsed is not None and parsed > 0:
            return (parsed, field)

    def _extract_best_candidate(text: str) -> float | None:
        matches = re.findall(
            r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,})(?:\.[0-9]+)?",
            text,
        )
        if not matches:
            return None
        candidates: list[float] = []
        for m in matches:
            try:
                candidates.append(float(m.replace(",", "")))
            except Exception:
                continue
        candidates = [c for c in candidates if 1_000 <= c <= 10_000_000]
        if not candidates:
            return None
        return max(candidates)

    for field in (
        "yes_sub_title",
        "no_sub_title",
        "subtitle",
        "title",
        "short_title",
        "event_title",
        "rules_primary",
        "rules_secondary",
    ):
        text = market.get(field)
        if not isinstance(text, str) or not text:
            continue
        parsed = _extract_best_candidate(text)
        if parsed is not None:
            return (parsed, field)

    return (None, "")


@dataclass(frozen=True)
class MarketSnapshot:
    poll_utc_iso: str
    ticker: str

    # External spot reference (Coinbase)
    btc_spot_usd: float | None

    # Timing
    seconds_to_expiry: int | None
    cutoff_time_utc_iso: str | None
    cutoff_time_source: str

    # Strike
    price_to_beat: float | None
    price_to_beat_source: str

    # Orderbook best prices (YES-space)
    best_yes_bid: int | None
    best_yes_ask: int | None
    best_no_bid: int | None
    best_no_ask: int | None
    yes_mid: float | None

    # Market-implied probabilities (from asks)
    market_p_yes: float | None
    market_p_no: float | None


def _mid(bid: int | None, ask: int | None) -> float | None:
    if bid is None or ask is None:
        return None
    return (float(bid) + float(ask)) / 2.0


def _prob_from_best_prices(*, bid: int | None, ask: int | None) -> float | None:
    """Market-implied probability in dollars from best prices.

    Prefer ask (what we'd pay to enter), otherwise fall back to bid (what we'd get to exit).
    """
    v = ask if ask is not None else bid
    if v is None:
        return None
    try:
        cents = int(v)
    except Exception:
        return None
    if cents < 0:
        return None
    return float(cents) / 100.0


async def fetch_btc_spot_price_coinbase() -> float | None:
    """Fetch a spot BTC price from Coinbase ticker."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"Accept": "application/json"}
            resp = await client.get(COINBASE_TICKER_URL, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            price = data.get("price")
            if price is None:
                return None
            return float(price)
    except Exception:
        return None


async def pick_active_markets(
    client: KalshiClient,
    *,
    asset: str = "BTC",
    horizon_minutes: int = 60,
    limit_markets: int = 2,
    min_seconds_to_expiry: int = 0,
) -> list[dict[str, Any]]:
    """Pick currently-open BTC 15-minute markets expiring soon."""
    now = _utcnow()
    horizon = dt.timedelta(minutes=int(horizon_minutes))

    series_ticker = f"KX{asset.upper()}15M"
    page = await client.get_markets_page(
        limit=100,
        status="open",
        series_ticker=series_ticker,
        mve_filter="exclude",
    )

    markets = list(page.get("markets", []) or [])

    valid: list[tuple[dt.datetime, dict[str, Any]]] = []
    for m in markets:
        cutoff, _field = _market_cutoff_time(m)
        if cutoff is None:
            continue
        if cutoff < now:
            continue
        if cutoff - now > horizon:
            continue
        if int((cutoff - now).total_seconds()) < int(min_seconds_to_expiry):
            continue
        valid.append((cutoff, m))

    valid.sort(key=lambda x: x[0])
    return [m for _, m in valid[: max(0, int(limit_markets))]]


async def fetch_snapshot_for_market(
    client: KalshiClient,
    market: dict[str, Any],
    *,
    btc_spot_usd: float | None,
) -> MarketSnapshot:
    now = _utcnow()
    ticker = str(market.get("ticker") or "").strip()
    if not ticker:
        raise ValueError("Market missing ticker")

    cutoff, cutoff_field = _market_cutoff_time(market)
    seconds_to_expiry: int | None
    cutoff_iso: str | None
    if cutoff is None:
        seconds_to_expiry = None
        cutoff_iso = None
    else:
        seconds_to_expiry = int((cutoff - now).total_seconds())
        cutoff_iso = cutoff.isoformat()

    ptb, ptb_field = _market_price_to_beat(market)

    ob = await client.get_orderbook(ticker)
    best: BestPrices = compute_best_prices(ob)

    market_p_yes = _prob_from_best_prices(bid=best.best_yes_bid, ask=best.best_yes_ask)
    market_p_no = _prob_from_best_prices(bid=best.best_no_bid, ask=best.best_no_ask)

    return MarketSnapshot(
        poll_utc_iso=now.isoformat(),
        ticker=ticker,
        btc_spot_usd=btc_spot_usd,
        seconds_to_expiry=seconds_to_expiry,
        cutoff_time_utc_iso=cutoff_iso,
        cutoff_time_source=cutoff_field,
        price_to_beat=ptb,
        price_to_beat_source=ptb_field,
        best_yes_bid=best.best_yes_bid,
        best_yes_ask=best.best_yes_ask,
        best_no_bid=best.best_no_bid,
        best_no_ask=best.best_no_ask,
        yes_mid=_mid(best.best_yes_bid, best.best_yes_ask),

        market_p_yes=market_p_yes,
        market_p_no=market_p_no,
    )


async def poll_once(
    client: KalshiClient,
    *,
    asset: str = "BTC",
    horizon_minutes: int = 60,
    limit_markets: int = 2,
    min_seconds_to_expiry: int = 0,
) -> list[MarketSnapshot]:
    btc_spot_usd = await fetch_btc_spot_price_coinbase()
    markets = await pick_active_markets(
        client,
        asset=asset,
        horizon_minutes=horizon_minutes,
        limit_markets=limit_markets,
        min_seconds_to_expiry=int(min_seconds_to_expiry),
    )

    if not markets:
        return []

    snaps = await asyncio.gather(
        *(fetch_snapshot_for_market(client, m, btc_spot_usd=btc_spot_usd) for m in markets)
    )
    return list(snaps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll Kalshi BTC 15m markets for ML inputs")
    parser.add_argument("--live", action="store_true", help="Continuously poll and print JSON snapshots")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--asset", type=str, default="BTC")
    parser.add_argument("--horizon-minutes", type=int, default=60)
    parser.add_argument("--limit-markets", type=int, default=2)
    args = parser.parse_args()

    async def runner() -> None:
        settings = Settings.load()
        client = KalshiClient.from_settings(settings)
        try:
            if not args.live:
                snaps = await poll_once(
                    client,
                    asset=str(args.asset),
                    horizon_minutes=int(args.horizon_minutes),
                    limit_markets=int(args.limit_markets),
                )
                print(json.dumps([s.__dict__ for s in snaps], indent=2, sort_keys=True))
                return

            end = dt.datetime.now().timestamp() + float(args.duration_seconds)
            while dt.datetime.now().timestamp() < end:
                snaps = await poll_once(
                    client,
                    asset=str(args.asset),
                    horizon_minutes=int(args.horizon_minutes),
                    limit_markets=int(args.limit_markets),
                )
                print(json.dumps([s.__dict__ for s in snaps], sort_keys=True))
                await asyncio.sleep(max(0.1, float(args.poll_seconds)))
        finally:
            await client.aclose()

    asyncio.run(runner())


if __name__ == "__main__":
    main()
