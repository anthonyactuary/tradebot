"""Kalshi market polling helpers for ML inference.

This module extracts the Kalshi-specific data we need per poll to later
construct ML model inputs:
- market ticker
- seconds to expiry (market close time)
- strike / "price to beat" (best-effort from market payload)
- best bid/ask (YES + NO) from the Kalshi orderbook

It intentionally does not implement trading logic.

Usage (prints JSON snapshots each poll):
    python -m tradebot.tools.kalshi_market_poll --live
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import re
from collections import deque
from dataclasses import dataclass
from typing import Any

import httpx

from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient
from tradebot.kalshi.orderbook import BestPrices, compute_best_prices
from tradebot.tools.coinbase_vwap import RollingVWAP


COINBASE_TICKER_URL = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
COINBASE_TRADES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/trades"
COINBASE_BOOK_URL = "https://api.exchange.coinbase.com/products/BTC-USD/book"

KRAKEN_TICKER_URL = "https://api.kraken.com/0/public/Ticker"


log = logging.getLogger(__name__)


# Rolling VWAP state (module-level so it persists across polls)
_ROLLING_VWAP = RollingVWAP()

# Dedup trade ids with a bounded FIFO (acts like a simple LRU)
_SEEN_TRADE_IDS: set[str] = set()
_SEEN_TRADE_IDS_Q: deque[str] = deque()
_MAX_SEEN_TRADE_IDS = 5000


def _seen_trade_id_add(trade_id: str) -> bool:
    """Return True if trade_id is new (and is added), else False."""

    tid = str(trade_id)
    if tid in _SEEN_TRADE_IDS:
        return False
    _SEEN_TRADE_IDS.add(tid)
    _SEEN_TRADE_IDS_Q.append(tid)
    while len(_SEEN_TRADE_IDS_Q) > int(_MAX_SEEN_TRADE_IDS):
        old = _SEEN_TRADE_IDS_Q.popleft()
        _SEEN_TRADE_IDS.discard(old)
    return True


def _iso_to_ts_ms(v: Any) -> int | None:
    if not v:
        return None
    if isinstance(v, (int, float)):
        # Heuristic: milliseconds vs seconds
        ts = float(v)
        if ts > 1e12:
            return int(ts)
        return int(ts * 1000.0)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            when = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            return int(when.timestamp() * 1000.0)
        except Exception:
            return None
    return None


async def fetch_recent_btc_trades_coinbase(*, limit: int = 100) -> list[dict[str, Any]]:
    """Fetch the most recent BTC-USD trades from Coinbase REST."""

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"Accept": "application/json"}
            resp = await client.get(COINBASE_TRADES_URL, headers=headers, params={"limit": int(limit)})
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
            return []
    except Exception as e:
        log.warning("COINBASE_TRADES_ERROR error=%s", e)
        return []


async def fetch_coinbase_best_bid_ask() -> tuple[float | None, float | None, float | None]:
    """Fetch top-of-book bid/ask for BTC-USD from Coinbase and return (bid, ask, mid)."""

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"Accept": "application/json"}
            resp = await client.get(COINBASE_BOOK_URL, headers=headers, params={"level": 1})
            resp.raise_for_status()
            data = resp.json() or {}

        bids = data.get("bids") or []
        asks = data.get("asks") or []
        best_bid: float | None = None
        best_ask: float | None = None

        if isinstance(bids, list) and bids:
            try:
                best_bid = float(bids[0][0])
            except Exception:
                best_bid = None

        if isinstance(asks, list) and asks:
            try:
                best_ask = float(asks[0][0])
            except Exception:
                best_ask = None

        mid: float | None = None
        if best_bid is not None and best_ask is not None:
            mid = (float(best_bid) + float(best_ask)) / 2.0

        return (best_bid, best_ask, mid)
    except Exception as e:
        log.warning("COINBASE_BOOK_ERROR error=%s", e)
        return (None, None, None)


async def fetch_kraken_best_bid_ask() -> tuple[float | None, float | None, float | None]:
    """Fetch top-of-book bid/ask for BTC-USD from Kraken and return (bid, ask, mid)."""

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(KRAKEN_TICKER_URL, params={"pair": "XBTUSD"})
            resp.raise_for_status()
            data = resp.json()

        if data.get("error"):
            log.warning("KRAKEN_API_ERROR errors=%s", data["error"])
            return (None, None, None)

        result = data.get("result", {}).get("XXBTZUSD", {})

        # Kraken format: "a" = [ask_price, whole_lot_volume, lot_volume]
        #                "b" = [bid_price, whole_lot_volume, lot_volume]
        ask_data = result.get("a", [])
        bid_data = result.get("b", [])

        best_ask = float(ask_data[0]) if ask_data else None
        best_bid = float(bid_data[0]) if bid_data else None

        mid: float | None = None
        if best_bid is not None and best_ask is not None:
            mid = (float(best_bid) + float(best_ask)) / 2.0

        return (best_bid, best_ask, mid)
    except Exception as e:
        log.warning("KRAKEN_BOOK_ERROR error=%s", e)
        return (None, None, None)


def compute_composite_mid(coinbase_mid: float | None, kraken_mid: float | None) -> float | None:
    """Compute composite mid from available exchange mids.
    
    Simple average when both are available, fall back to single source otherwise.
    """
    if coinbase_mid is not None and kraken_mid is not None:
        return (coinbase_mid + kraken_mid) / 2.0
    if coinbase_mid is not None:
        return coinbase_mid
    if kraken_mid is not None:
        return kraken_mid
    return None


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
        # Match prices like $94,902.05 or 94902 - require at least 5 digits to avoid year extraction (2026)
        matches = re.findall(
            r"\$\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{5,})(?:\.[0-9]+)?",
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
        # BTC prices are typically 10,000 - 1,000,000
        candidates = [c for c in candidates if 10_000 <= c <= 1_000_000]
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

    # Coinbase L1 orderbook
    coinbase_best_bid: float | None
    coinbase_best_ask: float | None
    coinbase_mid_usd: float | None

    # Coinbase rolling VWAP (derived from recent trades)
    coinbase_vwap_60s: float | None
    coinbase_vwap_count: int
    coinbase_vwap_age_ms: int | None

    # Kraken L1 orderbook
    kraken_best_bid: float | None
    kraken_best_ask: float | None
    kraken_mid_usd: float | None

    # Composite mid (Coinbase + Kraken average)
    composite_mid_usd: float | None

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
    """Pick currently-tradeable BTC 15-minute markets expiring soon.

    Note: KXBTC15M markets have status="active" when tradeable, but the API
    status filter only accepts "open", "closed", "settled". We query without
    status filter and filter client-side for tradeable statuses.
    """
    now = _utcnow()
    horizon = dt.timedelta(minutes=int(horizon_minutes))

    series_ticker = f"KX{asset.upper()}15M"
    
    # Kalshi uses US Eastern Time for ticker dates, not UTC
    try:
        import zoneinfo
        et_tz = zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        # Fallback: approximate ET as UTC-5
        et_tz = dt.timezone(dt.timedelta(hours=-5))
    
    now_et = now.astimezone(et_tz)
    today_prefix = f"{series_ticker}-{now_et.strftime('%y%b%d').upper()}"

    # Query all markets for this series - we'll filter by time client-side
    page = await client.get_markets_page(
        limit=200,  # Get more to find current ones
        series_ticker=series_ticker,
        mve_filter="exclude",
    )

    all_markets = list(page.get("markets", []) or [])
    
    # Filter to only today's markets first (alphabetical sort from API puts wrong dates first)
    todays_markets = [m for m in all_markets if str(m.get("ticker", "")).upper().startswith(today_prefix)]
    
    # Accept any status - we'll filter by time
    markets = todays_markets if todays_markets else all_markets

    valid: list[tuple[dt.datetime, dict[str, Any]]] = []
    for m in markets:
        cutoff, _field = _market_cutoff_time(m)
        if cutoff is None:
            continue
        if cutoff < now:
            continue
        if cutoff - now > horizon:
            continue
        tte_seconds = int((cutoff - now).total_seconds())
        if tte_seconds < int(min_seconds_to_expiry):
            continue
        valid.append((cutoff, m))

    valid.sort(key=lambda x: x[0])
    return [m for _, m in valid[: max(0, int(limit_markets))]]


async def fetch_snapshot_for_market(
    client: KalshiClient,
    market: dict[str, Any],
    *,
    btc_spot_usd: float | None,
    coinbase_best_bid: float | None,
    coinbase_best_ask: float | None,
    coinbase_mid_usd: float | None,
    coinbase_vwap_60s: float | None,
    coinbase_vwap_count: int,
    coinbase_vwap_age_ms: int | None,
    kraken_best_bid: float | None,
    kraken_best_ask: float | None,
    kraken_mid_usd: float | None,
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
        coinbase_best_bid=coinbase_best_bid,
        coinbase_best_ask=coinbase_best_ask,
        coinbase_mid_usd=coinbase_mid_usd,
        coinbase_vwap_60s=coinbase_vwap_60s,
        coinbase_vwap_count=int(coinbase_vwap_count),
        coinbase_vwap_age_ms=coinbase_vwap_age_ms,
        kraken_best_bid=kraken_best_bid,
        kraken_best_ask=kraken_best_ask,
        kraken_mid_usd=kraken_mid_usd,
        composite_mid_usd=compute_composite_mid(coinbase_mid_usd, kraken_mid_usd),
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


async def _safe_fetch_snapshot_for_market(
    client: KalshiClient,
    market: dict[str, Any],
    *,
    btc_spot_usd: float | None,
    coinbase_best_bid: float | None,
    coinbase_best_ask: float | None,
    coinbase_mid_usd: float | None,
    coinbase_vwap_60s: float | None,
    coinbase_vwap_count: int,
    coinbase_vwap_age_ms: int | None,
    kraken_best_bid: float | None,
    kraken_best_ask: float | None,
    kraken_mid_usd: float | None,
) -> MarketSnapshot | None:
    ticker = str(market.get("ticker") or "").strip()
    try:
        return await fetch_snapshot_for_market(
            client,
            market,
            btc_spot_usd=btc_spot_usd,
            coinbase_best_bid=coinbase_best_bid,
            coinbase_best_ask=coinbase_best_ask,
            coinbase_mid_usd=coinbase_mid_usd,
            coinbase_vwap_60s=coinbase_vwap_60s,
            coinbase_vwap_count=coinbase_vwap_count,
            coinbase_vwap_age_ms=coinbase_vwap_age_ms,
            kraken_best_bid=kraken_best_bid,
            kraken_best_ask=kraken_best_ask,
            kraken_mid_usd=kraken_mid_usd,
        )
    except Exception as e:
        # Treat per-market failures as transient and skip that market for this poll.
        # This avoids a single bad gateway / timeout from killing the whole bot.
        log.warning("SNAPSHOT_ERROR ticker=%s error=%s", ticker or "<missing>", e)
        return None


async def poll_once(
    client: KalshiClient,
    *,
    asset: str = "BTC",
    horizon_minutes: int = 60,
    limit_markets: int = 2,
    min_seconds_to_expiry: int = 0,
) -> list[MarketSnapshot]:
    # Feed the rolling VWAP with deduped Coinbase trades.
    trades = await fetch_recent_btc_trades_coinbase(limit=100)
    parsed: list[tuple[int, str, float, float]] = []
    for t in trades:
        tid = t.get("trade_id")
        if tid is None:
            tid = t.get("id")
        if tid is None:
            continue

        ts_ms = _iso_to_ts_ms(t.get("time"))
        if ts_ms is None:
            continue

        try:
            price = float(t.get("price"))
            size = float(t.get("size"))
        except Exception:
            continue

        parsed.append((int(ts_ms), str(tid), float(price), float(size)))

    # Coinbase often returns newest-first; ingest oldest->newest so our deque stays ordered.
    parsed.sort(key=lambda x: x[0])

    ingested = 0
    for ts_ms, tid, price, size in parsed:
        if not _seen_trade_id_add(tid):
            continue
        _ROLLING_VWAP.add_trade(ts_ms, price, size)
        ingested += 1

    # Use a "safe" now for VWAP cutoff + freshness that cannot be behind the newest
    # Coinbase timestamp we have in-buffer (avoids negative ages due to clock skew).
    latest_ts_ms = _ROLLING_VWAP.latest_ts_ms()
    now_ms_local = int(_utcnow().timestamp() * 1000.0)
    safe_now_ms = int(now_ms_local) if latest_ts_ms is None else max(int(now_ms_local), int(latest_ts_ms))

    vwap_60s = _ROLLING_VWAP.vwap(safe_now_ms, window_seconds=60)
    vwap_count = int(_ROLLING_VWAP.count())
    vwap_age_ms: int | None = None
    if latest_ts_ms is not None:
        raw_age = int(safe_now_ms) - int(latest_ts_ms)
        if raw_age < -5000:
            log.warning(
                "COINBASE_VWAP_NEG_AGE now_ms_local=%d safe_now_ms=%d latest_ts_ms=%d raw_age_ms=%d",
                int(now_ms_local),
                int(safe_now_ms),
                int(latest_ts_ms),
                int(raw_age),
            )
        vwap_age_ms = max(0, int(raw_age))

    cb_bid, cb_ask, cb_mid = await fetch_coinbase_best_bid_ask()
    kr_bid, kr_ask, kr_mid = await fetch_kraken_best_bid_ask()
    composite_mid = compute_composite_mid(cb_mid, kr_mid)
    log.info(
        "EXCHANGE_PRICES cb_mid=%s kr_mid=%s composite=%s diff=%s",
        "-" if cb_mid is None else f"{float(cb_mid):.2f}",
        "-" if kr_mid is None else f"{float(kr_mid):.2f}",
        "-" if composite_mid is None else f"{float(composite_mid):.2f}",
        "-" if cb_mid is None or kr_mid is None else f"{float(cb_mid - kr_mid):+.2f}",
    )
    log.info(
        "COINBASE_VWAP now_ms=%d now_ms_local=%d trades_ingested=%d vwap_count=%d vwap_60s=%s vwap_age_ms=%s cb_bid=%s cb_ask=%s cb_mid=%s",
        int(safe_now_ms),
        int(now_ms_local),
        int(ingested),
        int(vwap_count),
        "-" if vwap_60s is None else f"{float(vwap_60s):.2f}",
        "-" if vwap_age_ms is None else str(int(vwap_age_ms)),
        "-" if cb_bid is None else f"{float(cb_bid):.2f}",
        "-" if cb_ask is None else f"{float(cb_ask):.2f}",
        "-" if cb_mid is None else f"{float(cb_mid):.2f}",
    )

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
        *(
            _safe_fetch_snapshot_for_market(
                client,
                m,
                btc_spot_usd=btc_spot_usd,
                coinbase_best_bid=cb_bid,
                coinbase_best_ask=cb_ask,
                coinbase_mid_usd=cb_mid,
                coinbase_vwap_60s=vwap_60s,
                coinbase_vwap_count=vwap_count,
                coinbase_vwap_age_ms=vwap_age_ms,
                kraken_best_bid=kr_bid,
                kraken_best_ask=kr_ask,
                kraken_mid_usd=kr_mid,
            )
            for m in markets
        )
    )
    return [s for s in snaps if s is not None]


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
