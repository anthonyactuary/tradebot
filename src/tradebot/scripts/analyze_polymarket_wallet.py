"""Analyze Polymarket trade history for a wallet.

This script is analysis-only: it fetches historical trades/activity via Polymarket's
public Data API and reconstructs FIFO "round trips" (buy -> later sell) per
market+outcome.

Requirements (per request)
- Uses requests
- Paginates with limit/offset until the API returns empty
- Filters to BTC 15-minute direction markets by matching market title/slug text
- Outputs a CSV of reconstructed round trips

Docs
- Developer quickstart endpoints: https://docs.polymarket.com/quickstart/reference/endpoints
- Data API base URL: https://data-api.polymarket.com

Run
  python src/tradebot/scripts/analyze_polymarket_wallet.py \
    --wallet 0xe00740bce98a594e26861838885ab310ec3b548c \
    --out-csv runs/polymarket_roundtrips.csv

Notes
- The Data API response schema may evolve. This script uses best-effort field
  extraction and keeps a small set of canonical fields.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Tuple


DATA_API_BASE_URL_DEFAULT = "https://data-api.polymarket.com"
DEFAULT_WALLET = "0xe00740bce98a594e26861838885ab310ec3b548c"


@dataclass(frozen=True)
class Trade:
    ts: float  # unix seconds
    market_key: str
    market_title: str
    market_slug: str
    outcome: str
    size: float
    price: float
    direction: str  # buy/sell/unknown
    raw: dict[str, Any]

    # Optional metadata
    end_ts: float | None = None  # unix seconds


@dataclass(frozen=True)
class RoundTrip:
    market_key: str
    market_title: str
    market_slug: str
    outcome: str

    entry_ts: float
    exit_ts: float
    size: float
    entry_price: float
    exit_price: float

    hold_time_seconds: float
    pnl_per_contract: float
    pnl_total: float
    captured_cents: float

    entry_time_to_expiry_seconds: float | None


def _require_requests() -> Any:
    try:
        import requests  # type: ignore

        return requests
    except Exception as e:
        raise RuntimeError("Missing dependency 'requests'. Install with: pip install requests") from e


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return float(x)


def _parse_timestamp(value: Any) -> float | None:
    """Parse a timestamp from Data API payloads.

    Accepts:
    - unix seconds
    - unix milliseconds
    - ISO 8601 strings
    """

    if value is None:
        return None

    # numbers
    if isinstance(value, (int, float)):
        v = float(value)
        if not math.isfinite(v):
            return None
        # heuristic ms vs s
        if v > 1e12:
            v /= 1000.0
        if v > 0:
            return float(v)
        return None

    # strings
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # numeric string
        try:
            v = float(s)
            if math.isfinite(v):
                if v > 1e12:
                    v /= 1000.0
                if v > 0:
                    return float(v)
        except Exception:
            pass

        # ISO
        try:
            # Handle trailing Z
            if s.endswith("Z"):
                s2 = s[:-1] + "+00:00"
            else:
                s2 = s
            dtv = dt.datetime.fromisoformat(s2)
            if dtv.tzinfo is None:
                dtv = dtv.replace(tzinfo=dt.timezone.utc)
            return float(dtv.timestamp())
        except Exception:
            return None

    return None


def _normalize_wallet(addr: str) -> str:
    a = (addr or "").strip()
    if not a:
        raise ValueError("wallet is empty")
    if not a.startswith("0x"):
        raise ValueError("wallet must be 0x-prefixed")
    return a.lower()


def _coerce_direction(raw: dict[str, Any]) -> str:
    for k in ("direction", "side", "action", "tradeSide", "orderSide"):
        v = raw.get(k)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("buy", "b", "bid"):
                return "buy"
            if s in ("sell", "s", "ask"):
                return "sell"
    # Some schemas use booleans
    v = raw.get("isBuy")
    if isinstance(v, bool):
        return "buy" if v else "sell"
    return "unknown"


def _pick_first_str(raw: dict[str, Any], keys: Iterable[str]) -> str:
    for k in keys:
        v = raw.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _trade_from_raw(raw: dict[str, Any]) -> Trade | None:
    # Timestamp
    ts = (
        _parse_timestamp(raw.get("timestamp"))
        or _parse_timestamp(raw.get("createdAt"))
        or _parse_timestamp(raw.get("created_at"))
        or _parse_timestamp(raw.get("time"))
    )
    if ts is None:
        return None

    # Market identifiers
    market_key = _pick_first_str(raw, ["conditionId", "market", "marketId", "condition_id", "id"])
    market_slug = _pick_first_str(raw, ["slug", "marketSlug", "market_slug", "eventSlug", "event_slug"])
    market_title = _pick_first_str(raw, ["title", "marketTitle", "market_title", "event", "eventTitle"])

    # Outcome/side name
    outcome = _pick_first_str(raw, ["outcome", "outcomeName", "outcome_name", "token", "asset", "side"])

    # Size/price
    size = (
        _safe_float(raw.get("size"))
        or _safe_float(raw.get("quantity"))
        or _safe_float(raw.get("amount"))
        or _safe_float(raw.get("shares"))
    )
    price = _safe_float(raw.get("price")) or _safe_float(raw.get("avgPrice")) or _safe_float(raw.get("fillPrice"))

    if size is None or price is None:
        return None

    end_ts = _parse_timestamp(raw.get("endDate")) or _parse_timestamp(raw.get("end_date"))

    direction = _coerce_direction(raw)

    mk = market_key or market_slug or market_title
    if not mk:
        return None

    return Trade(
        ts=float(ts),
        market_key=str(mk),
        market_title=str(market_title or ""),
        market_slug=str(market_slug or ""),
        outcome=str(outcome or ""),
        size=float(size),
        price=float(price),
        direction=str(direction),
        raw=dict(raw),
        end_ts=float(end_ts) if end_ts is not None else None,
    )


def _raw_item_timestamp(it: dict[str, Any]) -> float | None:
    return (
        _parse_timestamp(it.get("timestamp"))
        or _parse_timestamp(it.get("createdAt"))
        or _parse_timestamp(it.get("created_at"))
        or _parse_timestamp(it.get("time"))
    )


def _iter_paginated(
    *,
    session: Any,
    base_url: str,
    path: str,
    params: dict[str, Any],
    limit: int,
    max_pages: int | None,
    sleep_seconds: float,
    since_ts: float | None,
    assume_descending_by_time: bool,
) -> Iterator[dict[str, Any]]:
    """Yield raw JSON objects from a limit/offset paginated endpoint."""

    offset = 0
    pages = 0
    total_items_emitted = 0
    # Data API docs typically cap offset at 10,000. Past that, some backends
    # may clamp and return repeating pages, which would otherwise infinite-loop.
    max_offset = 10_000

    # Detect repeated pages (e.g. backend clamping offset). Store a small set of
    # fingerprints to break safely.
    seen_fingerprints: set[str] = set()

    while True:
        if max_pages is not None and pages >= int(max_pages):
            break

        if int(offset) > int(max_offset):
            break

        q = dict(params)
        q["limit"] = int(limit)
        q["offset"] = int(offset)

        url = base_url.rstrip("/") + "/" + path.lstrip("/")
        resp = session.get(url, params=q, timeout=30)
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code} for {url}: {resp.text[:400]}")

        payload = resp.json()

        # The Data API endpoints documented for /positions return a list.
        # Be defensive: allow {items:[...]} as well.
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            v = payload.get("items") or payload.get("data") or payload.get("results")
            items = v if isinstance(v, list) else []
        else:
            items = []

        if not items:
            break

        # Fingerprint this page to detect repeats.
        fp: str | None = None
        try:
            first = items[0] if items else None
            last = items[-1] if items else None
            if isinstance(first, dict) and isinstance(last, dict):
                fp = json.dumps(
                    {
                        "offset": int(offset),
                        "first": {
                            "id": first.get("id"),
                            "tx": first.get("transactionHash") or first.get("txHash"),
                            "ts": _raw_item_timestamp(first),
                        },
                        "last": {
                            "id": last.get("id"),
                            "tx": last.get("transactionHash") or last.get("txHash"),
                            "ts": _raw_item_timestamp(last),
                        },
                        "n": len(items),
                    },
                    sort_keys=True,
                    default=str,
                )
        except Exception:
            fp = None

        if fp is not None:
            if fp in seen_fingerprints:
                break
            seen_fingerprints.add(fp)

        # If caller asked for a time window, filter rows and stop early once we
        # are confident subsequent pages will be even older.
        oldest_ts: float | None = None

        for it in items:
            if isinstance(it, dict):
                ts = _raw_item_timestamp(it)
                if ts is not None:
                    if oldest_ts is None or float(ts) < float(oldest_ts):
                        oldest_ts = float(ts)

                if since_ts is not None:
                    # Only emit rows inside the requested window.
                    if ts is None or float(ts) < float(since_ts):
                        continue

                yield it
                total_items_emitted += 1

        # If this page already contains older-than-since rows and the API is
        # sorted newest->oldest, then all future pages will be out-of-window.
        if since_ts is not None and bool(assume_descending_by_time):
            if oldest_ts is not None and float(oldest_ts) < float(since_ts):
                break

        pages += 1
        offset += int(limit)

        # Lightweight progress to help diagnose long runs.
        if pages % 10 == 0:
            sys.stderr.write(
                f"fetch_progress path={path} pages={pages} offset={offset} emitted={total_items_emitted}\n"
            )
            sys.stderr.flush()

        if sleep_seconds > 0:
            time.sleep(float(sleep_seconds))


def fetch_wallet_trades(
    *,
    wallet: str,
    base_url: str,
    limit: int = 200,
    sleep_seconds: float = 0.0,
    max_pages: int | None = None,
    include_activity: bool = True,
    include_trades: bool = True,
    since_ts: float | None = None,
    end_ts: float | None = None,
) -> list[Trade]:
    """Fetch all trade-like rows for a wallet from the Data API."""

    requests = _require_requests()
    wallet = _normalize_wallet(wallet)

    # Endpoints (per docs quick reference): /trades and /activity.
    # Some wallets may have trade-like fills in activity; we optionally fetch both
    # and de-dupe.
    endpoints: list[str] = []
    if include_trades:
        endpoints.append("trades")
    if include_activity:
        endpoints.append("activity")

    raw_rows: list[dict[str, Any]] = []
    with requests.Session() as s:
        for ep in endpoints:
            params: dict[str, Any] = {"user": wallet}
            assume_desc = True

            # Docs: /trades has takerOnly default true; being explicit is fine.
            # Docs: /activity supports type/start/end + sortBy/sortDirection.
            if ep == "trades":
                params["takerOnly"] = "true"
                assume_desc = True
            elif ep == "activity":
                params["type"] = "TRADE"
                params["sortBy"] = "TIMESTAMP"
                params["sortDirection"] = "DESC"
                assume_desc = True
                if since_ts is not None:
                    params["start"] = int(float(since_ts))
                if end_ts is not None:
                    params["end"] = int(float(end_ts))

            for row in _iter_paginated(
                session=s,
                base_url=base_url,
                path=f"/{ep}",
                params=params,
                limit=int(limit),
                max_pages=max_pages,
                sleep_seconds=float(sleep_seconds),
                since_ts=float(since_ts) if since_ts is not None else None,
                assume_descending_by_time=bool(assume_desc),
            ):
                raw_rows.append(row)

    # De-dupe best-effort
    seen: set[str] = set()
    out: list[Trade] = []

    for r in raw_rows:
        # Prefer an explicit id if present
        rid = r.get("id") or r.get("tradeId") or r.get("txHash") or r.get("hash")
        if isinstance(rid, str) and rid:
            key = f"id:{rid}"
        else:
            key = json.dumps(
                {
                    "t": r.get("timestamp") or r.get("createdAt") or r.get("time"),
                    "m": r.get("conditionId") or r.get("market") or r.get("slug") or r.get("title"),
                    "o": r.get("outcome") or r.get("asset") or r.get("token"),
                    "p": r.get("price") or r.get("avgPrice") or r.get("fillPrice"),
                    "s": r.get("size") or r.get("quantity") or r.get("amount") or r.get("shares"),
                    "d": r.get("direction") or r.get("side") or r.get("action") or r.get("isBuy"),
                },
                sort_keys=True,
                default=str,
            )

        if key in seen:
            continue
        seen.add(key)

        t = _trade_from_raw(r)
        if t is not None:
            out.append(t)

    out.sort(key=lambda x: x.ts)
    return out


def _is_btc_15m_market_text(text: str) -> bool:
    t = (text or "").lower()
    if "btc" not in t:
        return False

    # Keep this deliberately loose; user asked for contains "BTC" and "15" or "15m".
    if "15m" in t or "15 m" in t:
        return True
    if "15" in t and ("minute" in t or "min" in t):
        return True
    if "15" in t:
        return True
    return False


def filter_btc_15m_trades(trades: list[Trade]) -> list[Trade]:
    out: list[Trade] = []
    for t in trades:
        hay = " ".join(
            [
                t.market_title,
                t.market_slug,
                t.market_key,
                str(t.raw.get("eventSlug") or ""),
                str(t.raw.get("event") or ""),
            ]
        )
        if _is_btc_15m_market_text(hay):
            out.append(t)
    return out


def reconstruct_round_trips(trades: list[Trade]) -> tuple[list[RoundTrip], dict[str, int]]:
    """Pair buys with later sells for same (market_key, outcome) using FIFO."""

    # FIFO lots per market+outcome
    lots: dict[tuple[str, str], Deque[tuple[float, float, float, float | None]]] = defaultdict(deque)
    out: list[RoundTrip] = []

    stats = {
        "ignored_unknown_direction": 0,
        "ignored_sell_without_buy": 0,
        "ignored_nonpositive_size": 0,
    }

    for t in trades:
        if t.size <= 0:
            stats["ignored_nonpositive_size"] += 1
            continue

        key = (t.market_key, t.outcome)
        direction = (t.direction or "unknown").lower()

        if direction == "buy":
            lots[key].append((t.ts, t.price, t.size, t.end_ts))
            continue

        if direction != "sell":
            stats["ignored_unknown_direction"] += 1
            continue

        remaining = float(t.size)
        q = lots.get(key)
        if not q:
            stats["ignored_sell_without_buy"] += 1
            continue

        while remaining > 1e-12 and q:
            buy_ts, buy_price, buy_size, end_ts = q[0]
            take = min(float(remaining), float(buy_size))

            # Update / pop lot
            new_buy_size = float(buy_size) - float(take)
            if new_buy_size <= 1e-12:
                q.popleft()
            else:
                q[0] = (buy_ts, buy_price, new_buy_size, end_ts)

            remaining -= float(take)

            hold = float(t.ts - buy_ts)
            pnl_per = float(t.price - buy_price)
            pnl_total = float(pnl_per) * float(take)
            captured_cents = float(pnl_per) * 100.0

            entry_tte = None
            if end_ts is not None:
                entry_tte = float(end_ts - buy_ts)

            out.append(
                RoundTrip(
                    market_key=t.market_key,
                    market_title=t.market_title,
                    market_slug=t.market_slug,
                    outcome=t.outcome,
                    entry_ts=float(buy_ts),
                    exit_ts=float(t.ts),
                    size=float(take),
                    entry_price=float(buy_price),
                    exit_price=float(t.price),
                    hold_time_seconds=float(hold),
                    pnl_per_contract=float(pnl_per),
                    pnl_total=float(pnl_total),
                    captured_cents=float(captured_cents),
                    entry_time_to_expiry_seconds=entry_tte,
                )
            )

    out.sort(key=lambda r: r.entry_ts)
    return out, stats


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    p = float(p)
    if p <= 0:
        return float(xs[0])
    if p >= 100:
        return float(xs[-1])

    # Nearest-rank
    k = int(math.ceil((p / 100.0) * len(xs))) - 1
    k = max(0, min(len(xs) - 1, k))
    return float(xs[k])


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _fmt(v: float | None, *, digits: int = 2, suffix: str = "") -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{int(digits)}f}{suffix}"
    except Exception:
        return "-"


def _fmt_seconds(v: float | None) -> str:
    if v is None:
        return "-"
    v = float(v)
    if v < 0:
        return f"{v:.0f}s"
    if v < 120:
        return f"{v:.0f}s"
    if v < 7200:
        return f"{v/60.0:.1f}m"
    return f"{v/3600.0:.2f}h"


def print_summary(*, all_trades: list[Trade], btc_trades: list[Trade], round_trips: list[RoundTrip]) -> None:
    n_trades = len(all_trades)
    n_btc = len(btc_trades)
    n_rt = len(round_trips)

    pnls = [rt.pnl_total for rt in round_trips]
    holds = [rt.hold_time_seconds for rt in round_trips]
    captured = [rt.captured_cents for rt in round_trips]
    tte = [rt.entry_time_to_expiry_seconds for rt in round_trips if rt.entry_time_to_expiry_seconds is not None]

    wins = [p for p in pnls if p > 0]
    win_rate = (len(wins) / n_rt) if n_rt else None

    print("polymarket_wallet_analysis")
    print(f"  wallet_trades_total: {n_trades}")
    print(f"  btc_15m_trades:       {n_btc}")
    print(f"  round_trips:          {n_rt}")
    print(f"  win_rate:             {_fmt(win_rate * 100.0 if win_rate is not None else None, digits=1, suffix='%')}")
    print(f"  avg_pnl_per_trip:     {_fmt(_mean(pnls), digits=4)}")
    print(f"  avg_hold_time:        {_fmt_seconds(_mean(holds))}")
    print(f"  avg_entry_tte:        {_fmt_seconds(_mean([float(x) for x in tte]) if tte else None)}")
    print(f"  avg_captured_cents:   {_fmt(_mean(captured), digits=2)}")

    print("  hold_time_seconds:")
    print(f"    p50: {_fmt_seconds(_percentile(holds, 50))}   p90: {_fmt_seconds(_percentile(holds, 90))}")
    print("  captured_cents:")
    print(f"    p50: {_fmt(_percentile(captured, 50), digits=2)}   p90: {_fmt(_percentile(captured, 90), digits=2)}")


def write_roundtrips_csv(round_trips: list[RoundTrip], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    fields = [
        "market_key",
        "market_title",
        "market_slug",
        "outcome",
        "entry_ts",
        "exit_ts",
        "hold_time_seconds",
        "entry_price",
        "exit_price",
        "size",
        "pnl_per_contract",
        "pnl_total",
        "captured_cents",
        "entry_time_to_expiry_seconds",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rt in round_trips:
            w.writerow(
                {
                    "market_key": rt.market_key,
                    "market_title": rt.market_title,
                    "market_slug": rt.market_slug,
                    "outcome": rt.outcome,
                    "entry_ts": f"{rt.entry_ts:.3f}",
                    "exit_ts": f"{rt.exit_ts:.3f}",
                    "hold_time_seconds": f"{rt.hold_time_seconds:.3f}",
                    "entry_price": f"{rt.entry_price:.6f}",
                    "exit_price": f"{rt.exit_price:.6f}",
                    "size": f"{rt.size:.6f}",
                    "pnl_per_contract": f"{rt.pnl_per_contract:.6f}",
                    "pnl_total": f"{rt.pnl_total:.6f}",
                    "captured_cents": f"{rt.captured_cents:.4f}",
                    "entry_time_to_expiry_seconds": (
                        f"{rt.entry_time_to_expiry_seconds:.3f}" if rt.entry_time_to_expiry_seconds is not None else ""
                    ),
                }
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze Polymarket wallet trades (BTC 15m filter + FIFO round trips)")
    p.add_argument("--wallet", type=str, default=DEFAULT_WALLET)
    p.add_argument("--base-url", type=str, default=DATA_API_BASE_URL_DEFAULT)
    p.add_argument("--limit", type=int, default=200, help="Pagination page size")
    p.add_argument("--max-pages", type=int, default=None, help="Optional cap for debugging")
    p.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep between pages")
    p.add_argument(
        "--no-activity",
        action="store_true",
        help="Only fetch /trades (skip /activity). Useful if /activity duplicates trades.",
    )
    p.add_argument(
        "--source",
        type=str,
        default="activity",
        choices=["activity", "trades", "both"],
        help="Which Data API source(s) to pull from. 'activity' is recommended (supports start/end).",
    )
    p.add_argument(
        "--since-hours",
        type=float,
        default=24.0,
        help="Only pull rows from the last N hours (0 disables time filtering)",
    )
    p.add_argument(
        "--since-ts",
        type=float,
        default=None,
        help="Unix seconds lower bound (overrides --since-hours when provided)",
    )
    p.add_argument("--out-csv", type=str, default="runs/polymarket_roundtrips.csv")

    args = p.parse_args()

    now = time.time()
    since_ts: float | None = None
    if args.since_ts is not None:
        since_ts = float(args.since_ts)
    else:
        hrs = float(args.since_hours)
        if hrs and hrs > 0:
            since_ts = float(now - (hrs * 3600.0))

    trades = fetch_wallet_trades(
        wallet=str(args.wallet),
        base_url=str(args.base_url),
        limit=int(args.limit),
        sleep_seconds=float(args.sleep_seconds),
        max_pages=int(args.max_pages) if args.max_pages is not None else None,
        include_activity=(not bool(args.no_activity)) and (str(args.source) in ("activity", "both")),
        include_trades=(str(args.source) in ("trades", "both")),
        since_ts=since_ts,
        end_ts=float(now),
    )

    btc_trades = filter_btc_15m_trades(trades)
    round_trips, rt_stats = reconstruct_round_trips(btc_trades)

    print_summary(all_trades=trades, btc_trades=btc_trades, round_trips=round_trips)
    if any(int(v) > 0 for v in rt_stats.values()):
        print("  reconstruction_notes:")
        for k, v in rt_stats.items():
            if int(v) > 0:
                print(f"    {k}: {int(v)}")

    write_roundtrips_csv(round_trips, out_csv=str(args.out_csv))
    print(f"  wrote_csv: {str(args.out_csv)}")


if __name__ == "__main__":
    main()
