from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Iterable

from tradebot.kalshi.client import KalshiClient
from tradebot.kalshi.orderbook import compute_best_prices, iter_levels


@dataclass(frozen=True)
class Crypto15mMarket:
    asset: str
    series_ticker: str
    ticker: str
    event_ticker: str
    title: str
    yes_sub_title: str | None
    status: str | None
    expected_expiration_time: datetime.datetime | None
    yes_bid: int | None
    yes_ask: int | None
    no_bid: int | None
    no_ask: int | None
    last_price: int | None
    volume: int | None
    liquidity: int | None
    volume_24h: int | None


def _to_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _to_dt(v: Any) -> datetime.datetime | None:
    if not v:
        return None
    if isinstance(v, datetime.datetime):
        return v
    if not isinstance(v, str):
        return None
    try:
        return datetime.datetime.fromisoformat(v.replace("Z", "+00:00"))
    except Exception:
        return None


def _is_valid_price(p: int | None) -> bool:
    return p is not None and 1 <= p <= 99


def _calc_two_sided_arb(
    *,
    yes_bid: int | None,
    yes_ask: int | None,
    no_bid: int | None,
    no_ask: int | None,
) -> tuple[int | None, int | None, int | None, int | None, int | None]:
    """Compute theoretical two-sided arb for a single binary market.

    BUY both sides (YES at ask + NO at ask): profit if total cost < 100.
    SELL both sides (YES at bid + NO at bid): profit if total credit > 100.
    """
    buy_cost = None
    buy_edge = None
    if _is_valid_price(yes_ask) and _is_valid_price(no_ask):
        buy_cost = int(yes_ask) + int(no_ask)
        buy_edge = 100 - buy_cost

    sell_credit = None
    sell_edge = None
    if _is_valid_price(yes_bid) and _is_valid_price(no_bid):
        sell_credit = int(yes_bid) + int(no_bid)
        sell_edge = sell_credit - 100

    best_edge = None
    if buy_edge is not None or sell_edge is not None:
        best_edge = max(buy_edge if buy_edge is not None else -10**9, sell_edge if sell_edge is not None else -10**9)

    return buy_cost, buy_edge, sell_credit, sell_edge, best_edge


def _format_levels(levels: list[tuple[int, int]]) -> str:
    if not levels:
        return "(empty)"
    return " | ".join(f"{p}:{q}" for p, q in levels)


def _effective_volume(item: Crypto15mMarket) -> int:
    return max(int(item.volume or 0), int(item.volume_24h or 0))


def _effective_liquidity(item: Crypto15mMarket) -> int:
    return int(item.liquidity or 0)


def _discover_15m_series(series: list[dict[str, Any]], asset: str) -> str | None:
    asset_up = asset.upper()
    expected = f"KX{asset_up}15M"

    # Prefer exact ticker match.
    for s in series:
        if (s.get("ticker") or "").upper() == expected:
            return expected

    # Otherwise fall back to title match.
    pat = re.compile(rf"\b{re.escape(asset_up)}\b.*15\s*m", re.I)
    for s in series:
        title = (s.get("title") or "")
        ticker = (s.get("ticker") or "")
        if pat.search(title) and ticker:
            return ticker

    return None


def _fmt_dt(dt: datetime.datetime | None) -> str:
    if dt is None:
        return "?"
    return dt.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


async def scan_crypto_15m_markets(
    *,
    client: KalshiClient,
    assets: Iterable[str],
    horizon_minutes: int,
    per_asset: int,
    min_liquidity: int,
    min_volume_24h: int,
    require_quotes: bool = False,
    orderbook_levels: int = 0,
) -> None:
    log = logging.getLogger("tradebot.crypto")

    assets_norm = [a.strip().upper() for a in assets if a and a.strip()]
    if not assets_norm:
        log.error("No assets provided")
        return

    series = await client.get_series_list(category="Crypto", include_product_metadata=True)
    if not series:
        log.error("No series returned for category=Crypto")
        return

    now = datetime.datetime.now(datetime.timezone.utc)
    horizon = datetime.timedelta(minutes=max(0, horizon_minutes))

    log.info(
        "Scanning Crypto 15m markets assets=%s horizon=%dmin per_asset=%d min_liq=%d min_vol24h=%d",
        ",".join(assets_norm),
        horizon_minutes,
        per_asset,
        min_liquidity,
        min_volume_24h,
    )

    results: list[Crypto15mMarket] = []
    missing: list[str] = []

    for asset in assets_norm:
        st = _discover_15m_series(series, asset)
        if not st:
            missing.append(asset)
            continue

        page = await client.get_markets_page(
            limit=1000,
            status="open",
            series_ticker=st,
            mve_filter="exclude",
        )
        markets = list(page.get("markets") or [])

        for m in markets:
            exp = _to_dt(m.get("expected_expiration_time"))
            if exp is None:
                continue
            if exp < now:
                continue
            if exp - now > horizon:
                continue

            item = Crypto15mMarket(
                asset=asset,
                series_ticker=st,
                ticker=str(m.get("ticker")),
                event_ticker=str(m.get("event_ticker")),
                title=str(m.get("title")),
                yes_sub_title=(m.get("yes_sub_title") or None),
                status=m.get("status"),
                expected_expiration_time=exp,
                yes_bid=_to_int(m.get("yes_bid")),
                yes_ask=_to_int(m.get("yes_ask")),
                no_bid=_to_int(m.get("no_bid")),
                no_ask=_to_int(m.get("no_ask")),
                last_price=_to_int(m.get("last_price")),
                volume=_to_int(m.get("volume")),
                liquidity=_to_int(m.get("liquidity")),
                volume_24h=_to_int(m.get("volume_24h")),
            )

            # If requested, treat "real quotes" as coming from the orderbook.
            # The markets list sometimes returns placeholder 0/100 values.
            if require_quotes:
                ob = await client.get_orderbook(item.ticker)
                prices = compute_best_prices(ob)
                item = replace(
                    item,
                    yes_bid=prices.best_yes_bid,
                    yes_ask=prices.best_yes_ask,
                    no_bid=prices.best_no_bid,
                    no_ask=prices.best_no_ask,
                )

            if min_liquidity > 0:
                if _effective_liquidity(item) < min_liquidity:
                    continue
            if min_volume_24h > 0:
                if _effective_volume(item) < min_volume_24h:
                    continue
            if require_quotes:
                if not _is_valid_price(item.yes_bid) or not _is_valid_price(item.yes_ask):
                    continue

            results.append(item)

    if missing:
        log.warning("No 15m series found for: %s", ", ".join(missing))

    if not results:
        if "demo" in (client.base_url or "").casefold():
            log.warning(
                "No quoteable crypto 15m markets found. You are using demo API (%s); demo crypto often has no liquidity. "
                "If you expect real volume/quotes, set KALSHI_ENV=prod and/or KALSHI_BASE_URL to the production API host in your .env.",
                client.base_url,
            )
        log.info("No matching 15m markets found in horizon")
        return

    results.sort(key=lambda r: (r.asset, r.expected_expiration_time or now))

    log.info("Found markets=%d", len(results))
    print("\n=== Crypto 15m markets (next horizon) ===\n")

    by_asset: dict[str, list[Crypto15mMarket]] = {}
    for r in results:
        by_asset.setdefault(r.asset, []).append(r)

    for asset in assets_norm:
        rows = by_asset.get(asset) or []
        if not rows:
            continue

        print(f"asset={asset} series={rows[0].series_ticker} count={min(len(rows), per_asset)}")
        for r in rows[: max(0, per_asset)]:
            mins = int(round(((r.expected_expiration_time or now) - now).total_seconds() / 60.0))
            yes_ab = (
                f"{r.yes_ask}/{r.yes_bid}"
                if (r.yes_ask is not None and r.yes_bid is not None)
                else "?/?"
            )
            no_ab = (
                f"{r.no_ask}/{r.no_bid}"
                if (r.no_ask is not None and r.no_bid is not None)
                else "?/?"
            )

            print(f"  event={r.event_ticker}")
            print(f"  ticker={r.ticker}")
            print(f"  exp={_fmt_dt(r.expected_expiration_time)} (in ~{mins}m) status={r.status}")
            if r.yes_sub_title:
                print(f"  threshold={r.yes_sub_title}")

            buy_cost, buy_edge, sell_credit, sell_edge, best_edge = _calc_two_sided_arb(
                yes_bid=r.yes_bid,
                yes_ask=r.yes_ask,
                no_bid=r.no_bid,
                no_ask=r.no_ask,
            )

            print(
                f"  YES a/b={yes_ab} | NO a/b={no_ab} | last={r.last_price} | liq={r.liquidity} vol={r.volume} vol24h={r.volume_24h}"
            )
            if best_edge is not None:
                parts: list[str] = []
                if buy_cost is not None and buy_edge is not None:
                    parts.append(f"BUY both: cost={buy_cost} edge={buy_edge}c")
                if sell_credit is not None and sell_edge is not None:
                    parts.append(f"SELL both: credit={sell_credit} edge={sell_edge}c")
                parts.append(f"BEST={best_edge}c")
                print("  ARB: " + " | ".join(parts))

            if orderbook_levels > 0:
                # Show the same ladder you see on the website.
                # Kalshi orderbook returns bids on YES and NO; asks are implied via 100 - opposite bid.
                ob = await client.get_orderbook(r.ticker)
                yes_bids = list(iter_levels(ob, side="yes"))
                no_bids = list(iter_levels(ob, side="no"))

                n = max(1, int(orderbook_levels))
                top_yes_bids = yes_bids[-n:]
                top_no_bids = no_bids[-n:]

                # Implied asks: NO bids imply YES asks at (100 - price), and vice-versa.
                top_yes_asks = [(100 - p, q) for p, q in top_no_bids]
                top_no_asks = [(100 - p, q) for p, q in top_yes_bids]

                print(f"  OB YES bids: {_format_levels(top_yes_bids)}")
                print(f"  OB YES asks: {_format_levels(top_yes_asks)}")
                print(f"  OB NO  bids: {_format_levels(top_no_bids)}")
                print(f"  OB NO  asks: {_format_levels(top_no_asks)}")

            print("-")
        print("")
