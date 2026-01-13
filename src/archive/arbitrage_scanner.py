from __future__ import annotations

import asyncio
import datetime
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from tradebot.kalshi.client import KalshiClient


@dataclass(frozen=True)
class MarketTop:
    ticker: str
    event_ticker: str
    title: str
    yes_sub_title: str | None
    no_sub_title: str | None
    open_time: datetime.datetime | None
    expected_expiration_time: datetime.datetime | None
    yes_bid: int | None
    yes_ask: int | None
    no_bid: int | None
    no_ask: int | None
    liquidity: int | None
    volume_24h: int | None
    status: str | None


@dataclass(frozen=True)
class Candidate:
    event_ticker: str
    prefix: str
    legs: tuple[MarketTop, MarketTop]
    buy_cost: int
    buy_edge: int
    sell_credit: int | None
    sell_edge: int | None
    best_edge: int


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
        # Kalshi returns RFC3339 like: 2026-01-10T09:30:00Z
        return datetime.datetime.fromisoformat(v.replace("Z", "+00:00"))
    except Exception:
        return None


def _market_prefix(ticker: str) -> str:
    # Heuristic for sports game-winner markets often encoded as PREFIX-TEAM.
    # Group by everything before the last dash.
    if "-" not in ticker:
        return ticker
    return ticker.rsplit("-", 1)[0]


def _market_suffix(prefix: str, ticker: str) -> str:
    if ticker.startswith(prefix + "-"):
        return ticker[len(prefix) + 1 :]
    return ticker


_TWO_LEG_OUTCOME_RE = re.compile(r"^[A-Z]{1,10}$")


def _is_valid_price(p: int | None) -> bool:
    return p is not None and 1 <= p <= 99


def _gte(v: int | None, threshold: int) -> bool:
    return (v is not None and v >= threshold)


def _looks_like_two_outcome_partition(prefix: str, a: MarketTop, b: MarketTop) -> bool:
    """Heuristic guardrail for 2-leg partitions.

    The math `100 - (askA + askB)` only makes sense if exactly one leg can win.
    Many sports families (spreads/totals ladders) will sometimes produce 2
    surviving markets after filters, but they are not complements.
    """

    # Avoid common ladder families; we handle special 4-leg cases separately.
    pf = prefix.casefold()
    if any(tok in pf for tok in ("spread", "total", "teamtotal")):
        return False

    sa = _market_suffix(prefix, a.ticker)
    sb = _market_suffix(prefix, b.ticker)
    if sa == sb:
        return False
    if not _TWO_LEG_OUTCOME_RE.fullmatch(sa):
        return False
    if not _TWO_LEG_OUTCOME_RE.fullmatch(sb):
        return False

    return True


async def scan_sports_arbitrage(
    *,
    client: KalshiClient,
    category: str,
    min_edge_cents: int,
    min_liquidity: int,
    min_volume_24h: int,
    min_yes_bid: int,
    min_yes_ask: int,
    max_yes_ask: int,
    require_two_sided: bool,
    live_only: bool,
    live_window_hours: int,
    print_candidates: bool = True,
) -> list[Candidate]:
    log = logging.getLogger("tradebot.scan")

    # 1) Discover sports series
    series = await client.get_series_list(category=category, include_product_metadata=True)
    if not series:
        tags = await client.get_tags_by_categories()
        categories = sorted((tags.get("tags_by_categories") or {}).keys())
        resolved = next((c for c in categories if c.casefold() == category.casefold()), None)
        if resolved and resolved != category:
            log.info("Resolved category '%s' -> '%s'", category, resolved)
            series = await client.get_series_list(category=resolved, include_product_metadata=True)
            category = resolved

    if not series:
        tags = await client.get_tags_by_categories()
        log.error(
            "No series for category=%s. Available categories (from tags_by_categories keys) may help: %s",
            category,
            ", ".join(sorted((tags.get("tags_by_categories") or {}).keys())),
        )
        return []

    if live_only:
        # Kalshi sports series expose a `product_metadata.scope` field.
        # We only want single-game/match series (exclude futures/awards/etc).
        series = [
            s
            for s in series
            if (s.get("product_metadata") or {}).get("scope", "").casefold() in ("game", "games")
        ]

    series_tickers = [s["ticker"] for s in series if s.get("ticker")]
    log.info(
        "Scanning category=%s series=%d filters: min_edge=%dc min_liq=%d min_vol24h=%d min_bid=%d ask_range=[%d,%d] require_two_sided=%s live_only=%s live_window_hours=%d",
        category,
        len(series_tickers),
        min_edge_cents,
        min_liquidity,
        min_volume_24h,
        min_yes_bid,
        min_yes_ask,
        max_yes_ask,
        require_two_sided,
        live_only,
        live_window_hours,
    )

    # 2) Pull all open markets across sports series (paginated per-series)
    markets: list[MarketTop] = []

    # Concurrency limit to be polite.
    # Production endpoints rate-limit more aggressively than demo.
    concurrency = 2 if "demo" not in (client.base_url or "").casefold() else 8
    sem = asyncio.Semaphore(concurrency)

    async def fetch_series_markets(series_ticker: str) -> None:
        cursor: str | None = None
        while True:
            async with sem:
                page = await client.get_markets_page(
                    limit=1000,
                    cursor=cursor,
                    status="open",
                    series_ticker=series_ticker,
                    mve_filter="exclude",
                )

            # Small delay to reduce burstiness across many series.
            await asyncio.sleep(0.05)

            for m in page.get("markets") or []:
                markets.append(
                    MarketTop(
                        ticker=str(m.get("ticker")),
                        event_ticker=str(m.get("event_ticker")),
                        title=str(m.get("title")),
                        yes_sub_title=(m.get("yes_sub_title") or None),
                        no_sub_title=(m.get("no_sub_title") or None),
                        open_time=_to_dt(m.get("open_time")),
                        expected_expiration_time=_to_dt(m.get("expected_expiration_time")),
                        yes_bid=_to_int(m.get("yes_bid")),
                        yes_ask=_to_int(m.get("yes_ask")),
                        no_bid=_to_int(m.get("no_bid")),
                        no_ask=_to_int(m.get("no_ask")),
                        liquidity=_to_int(m.get("liquidity")),
                        volume_24h=_to_int(m.get("volume_24h")),
                        status=m.get("status"),
                    )
                )

            cursor = page.get("cursor")
            if not cursor:
                return

    await asyncio.gather(*(fetch_series_markets(st) for st in series_tickers))
    log.info("Loaded open markets=%d", len(markets))

    # 2b) Apply tradability filters (reduce noise)
    filtered: list[MarketTop] = []
    now = datetime.datetime.now(datetime.timezone.utc)
    live_window = datetime.timedelta(hours=max(live_window_hours, 0))
    for m in markets:
        if not m.event_ticker or not m.ticker:
            continue

        # Live-only filter: only include events expiring soon.
        # We use expected_expiration_time because `close_time` can be far in the future.
        if live_only:
            if m.expected_expiration_time is None:
                continue
            if m.expected_expiration_time < now:
                continue
            if m.expected_expiration_time - now > live_window:
                continue

        # Liquidity/volume gates
        if not _gte(m.liquidity, min_liquidity):
            continue
        if not _gte(m.volume_24h, min_volume_24h):
            continue

        # Price sanity
        if not _is_valid_price(m.yes_bid) or not _is_valid_price(m.yes_ask):
            continue
        if int(m.yes_bid) < min_yes_bid:
            continue
        if int(m.yes_ask) < min_yes_ask or int(m.yes_ask) > max_yes_ask:
            continue

        # Require a real 2-sided market (both YES and NO have bids) unless disabled
        if require_two_sided:
            if not _is_valid_price(m.no_bid):
                continue
            if int(m.no_bid) < 1:
                continue

        filtered.append(m)

    log.info("After filters, markets=%d", len(filtered))

    # 3) Group likely complementary markets
    # We require same event_ticker + same prefix, and exactly 2 markets.
    grouped: dict[tuple[str, str], list[MarketTop]] = defaultdict(list)
    for m in filtered:
        grouped[(m.event_ticker, _market_prefix(m.ticker))].append(m)

    # 4) Compute candidates
    candidates: list[Candidate] = []
    for (event_ticker, prefix), group in grouped.items():
        group_sorted = sorted(group, key=lambda m: m.ticker)

        # Only consider classic 2-outcome partitions (A vs B)
        if len(group_sorted) != 2:
            continue

        if not _looks_like_two_outcome_partition(prefix, group_sorted[0], group_sorted[1]):
            continue

        a, b = group_sorted[0], group_sorted[1]
        buy_cost = int(a.yes_ask) + int(b.yes_ask)
        buy_edge = 100 - buy_cost

        sell_edge = None
        sell_credit = 0
        if _is_valid_price(a.yes_bid) and _is_valid_price(b.yes_bid):
            sell_credit = int(a.yes_bid) + int(b.yes_bid)
            sell_edge = sell_credit - 100

        best_edge = max(buy_edge, sell_edge if sell_edge is not None else -10**9)
        if best_edge < min_edge_cents:
            continue

        candidates.append(
            Candidate(
                event_ticker=event_ticker,
                prefix=prefix,
                legs=(a, b),
                buy_cost=buy_cost,
                buy_edge=buy_edge,
                sell_credit=(sell_credit if sell_edge is not None else None),
                sell_edge=sell_edge,
                best_edge=best_edge,
            )
        )

    candidates.sort(key=lambda c: c.best_edge, reverse=True)

    if not candidates:
        log.info("No 2-leg candidates found at min_edge=%dc", min_edge_cents)
        return []

    log.info("Found %d 2-leg candidates (min_edge=%dc)", len(candidates), min_edge_cents)

    if print_candidates:
        print(f"\n=== Arbitrage candidates (2-leg only) | min_edge={min_edge_cents}c ===\n")
        for c in candidates[:200]:
            a, b = c.legs
            sa = _market_suffix(c.prefix, a.ticker)
            sb = _market_suffix(c.prefix, b.ticker)
            print(f"event={c.event_ticker}")
            print(f"prefix={c.prefix}")
            print(
                f"  leg1={sa}  YES a/b={a.yes_ask}/{a.yes_bid}  liq={a.liquidity} vol24h={a.volume_24h}"
            )
            print(
                f"  leg2={sb}  YES a/b={b.yes_ask}/{b.yes_bid}  liq={b.liquidity} vol24h={b.volume_24h}"
            )
            if c.sell_edge is not None and c.sell_credit is not None:
                print(
                    f"  BUY: cost={c.buy_cost} edge={c.buy_edge}c | "
                    f"SELL: credit={c.sell_credit} edge={c.sell_edge}c | BEST={c.best_edge}c"
                )
            else:
                print(f"  BUY: cost={c.buy_cost} edge={c.buy_edge}c | BEST={c.best_edge}c")
            print("-")

    return candidates
