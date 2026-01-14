from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import os
import sys
import time
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Any

# Allow running this file directly ("python analytics/kalshi_activity_report.py") in a
# src-layout repo by ensuring the src/ root is on sys.path.
_THIS_FILE = Path(__file__).resolve()
_SRC_ROOT = _THIS_FILE.parents[1] / "src"  # .../tradebot/src
if (_SRC_ROOT / "tradebot").is_dir():
    src_root_str = str(_SRC_ROOT)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _write_csv_atomic(*, path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    """Write CSV atomically (temp file then replace) with simple retries.

    This mitigates Windows PermissionError when the destination file is briefly locked.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory so os.replace is atomic.
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        last_err: Exception | None = None
        for i in range(8):
            try:
                os.replace(str(tmp_path), str(path))
                return str(path)
            except PermissionError as e:
                last_err = e
                time.sleep(0.25 * (i + 1))

        # If we still can't replace (file locked by something like Excel), keep the tmp
        # contents by writing to a timestamped sibling name.
        fallback = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
        try:
            os.replace(str(tmp_path), str(fallback))
            return str(fallback)
        except Exception as e:
            # Give the caller a useful error that includes the original lock cause.
            raise PermissionError(
                f"Failed to write {path} (locked). Also failed to write fallback {fallback}."
            ) from (last_err or e)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def export_settlements_csv(*, settlements: list[dict[str, Any]], out_path: str) -> str:
    """Write `get_settlements` settlements to CSV (fields per docs) and return resolved path."""

    path = Path(out_path).expanduser().resolve()

    fieldnames = [
        "ticker",
        "event_ticker",
        "market_result",
        "yes_count",
        "yes_total_cost",
        "no_count",
        "no_total_cost",
        "revenue",
        "settled_time",
        "fee_cost",
        "value",
    ]

    rows: list[dict[str, Any]] = []
    for s in settlements:
        if not isinstance(s, dict):
            continue
        rows.append({k: s.get(k) for k in fieldnames})

    return _write_csv_atomic(path=path, fieldnames=fieldnames, rows=rows)


def export_fills_csv(*, fills: list[dict[str, Any]], out_path: str) -> str:
    """Write `get_fills` fills to CSV (fields per docs) and return resolved path."""

    path = Path(out_path).expanduser().resolve()

    fieldnames = [
        "fill_id",
        "trade_id",
        "order_id",
        "ticker",
        "market_ticker",
        "side",
        "action",
        "count",
        "price",
        "yes_price",
        "no_price",
        "yes_price_fixed",
        "no_price_fixed",
        "is_taker",
        "client_order_id",
        "created_time",
        "ts",
    ]

    rows: list[dict[str, Any]] = []
    for fill in fills:
        if not isinstance(fill, dict):
            continue
        rows.append({k: fill.get(k) for k in fieldnames})

    return _write_csv_atomic(path=path, fieldnames=fieldnames, rows=rows)


async def _page_all(
    fetch_page,
    *,
    limit: int,
    **kwargs,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        page = await fetch_page(limit=limit, cursor=cursor, **kwargs)
        cursor = page.get("cursor")
        # items key depends on endpoint; caller will extract
        items.append(page)
        if not cursor:
            break
    return items


async def fetch_settlements(
    client: KalshiClient, *, min_ts: int, max_ts: int, ticker: str | None
) -> list[dict[str, Any]]:
    pages = await _page_all(
        client.get_settlements,
        limit=200,
        ticker=ticker,
        min_ts=min_ts,
        max_ts=max_ts,
    )
    settlements: list[dict[str, Any]] = []
    for p in pages:
        settlements.extend(p.get("settlements", []) or [])
    return settlements


async def fetch_fills(client: KalshiClient, *, min_ts: int, max_ts: int, ticker: str | None) -> list[dict[str, Any]]:
    pages = await _page_all(
        client.get_fills,
        limit=200,
        ticker=ticker,
        min_ts=min_ts,
        max_ts=max_ts,
    )
    fills: list[dict[str, Any]] = []
    for p in pages:
        fills.extend(p.get("fills", []) or [])
    return fills


def _parse_iso_z(dt: str) -> datetime.datetime | None:
    if not dt:
        return None
    # Kalshi returns e.g. 2026-01-11T20:50:34.298935Z
    if dt.endswith("Z"):
        dt = dt[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(dt)
    except Exception:
        return None


def _to_float(x: Any) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x))
    except Exception:
        return 0.0


async def run(
    minutes: int,
    *,
    ticker: str | None,
    out_settlements_csv: str | None,
    out_fills_csv: str | None,
) -> None:
    now = _utcnow()
    start = now - datetime.timedelta(minutes=minutes)
    min_ts = int(start.timestamp())
    max_ts = int(now.timestamp())

    settings = Settings.load()
    client = KalshiClient.from_settings(settings)

    try:
        settlements = await fetch_settlements(client, min_ts=min_ts, max_ts=max_ts, ticker=ticker)
        fills: list[dict[str, Any]] = []
        if out_fills_csv:
            fills = await fetch_fills(client, min_ts=min_ts, max_ts=max_ts, ticker=ticker)

        if out_settlements_csv:
            wrote = export_settlements_csv(settlements=settlements, out_path=str(out_settlements_csv))
            print(f"\nWrote settlements CSV: {wrote}")

        if out_fills_csv:
            wrote = export_fills_csv(fills=fills, out_path=str(out_fills_csv))
            print(f"Wrote fills CSV: {wrote}")

        # ---- Settlements summary (direct fields) ----
        # Docs: revenue is an int (assumed cents); fee_cost is a dollars string.
        revenue_cents_total = 0
        fee_cost_total = 0.0
        by_ticker_count: dict[str, int] = defaultdict(int)
        by_ticker_revenue_cents: dict[str, int] = defaultdict(int)
        by_ticker_fee_cost: dict[str, float] = defaultdict(float)

        for s in settlements:
            t = str(s.get("ticker") or "(unknown)")
            by_ticker_count[t] += 1

            rev_cents = int(s.get("revenue") or 0)
            revenue_cents_total += rev_cents
            by_ticker_revenue_cents[t] += rev_cents

            fee = _to_float(s.get("fee_cost"))
            fee_cost_total += fee
            by_ticker_fee_cost[t] += fee

        print("")
        print(f"Kalshi settlements report: last {minutes} minutes")
        if ticker:
            print(f"Ticker filter: {ticker}")
        print(f"Window: {start.isoformat()} -> {now.isoformat()}")

        if out_fills_csv:
            print("")
            print("Fills")
            print(f"- records: {len(fills)}")

        print("")
        print("Settlements")
        print(f"- records: {len(settlements)}")
        print(f"- revenue_total: ${revenue_cents_total/100.0:.2f}")
        print(f"- fee_cost_total: ${fee_cost_total:.2f}")
        print(f"- net (revenue - fee_cost): ${(revenue_cents_total/100.0 - fee_cost_total):.2f}")

        print("")
        print("By ticker (settlements)")
        rows = sorted(by_ticker_count.keys(), key=lambda t: by_ticker_count[t], reverse=True)
        for t in rows[:15]:
            n = by_ticker_count[t]
            rev = by_ticker_revenue_cents[t] / 100.0
            fee = by_ticker_fee_cost[t]
            print(f"- {t}: records={n}, revenue=${rev:.2f}, fee_cost=${fee:.2f}, net=${(rev-fee):.2f}")

    finally:
        await client.aclose()


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze Kalshi portfolio activity (fills/settlements/positions)")
    p.add_argument("--minutes", type=int, default=90)
    p.add_argument("--ticker", type=str, default=None)
    p.add_argument(
        "--settlements-csv",
        dest="out_settlements_csv",
        type=str,
        default=None,
        help="Optional output CSV path for Kalshi get_settlements settlements payload (direct fields).",
    )
    p.add_argument(
        "--fills-csv",
        dest="out_fills_csv",
        type=str,
        default=None,
        help="Optional output CSV path for Kalshi get_fills fills payload (direct fields).",
    )
    args = p.parse_args()

    asyncio.run(
        run(
            args.minutes,
            ticker=args.ticker,
            out_settlements_csv=args.out_settlements_csv,
            out_fills_csv=args.out_fills_csv,
        )
    )


if __name__ == "__main__":
    main()
