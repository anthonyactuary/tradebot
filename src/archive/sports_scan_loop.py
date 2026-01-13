from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass

from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient
from tradebot.tools.arbitrage_scanner import scan_sports_arbitrage


@dataclass(frozen=True)
class ScanLoopResult:
    runs: int
    total_candidates: int
    runs_with_candidates: int
    max_candidates_in_run: int


async def run_scan_loop(
    *,
    client: KalshiClient,
    category: str,
    duration_seconds: int,
    interval_seconds: int,
    # scanner args
    min_edge_cents: int,
    min_liquidity: int,
    min_volume_24h: int,
    min_yes_bid: int,
    min_yes_ask: int,
    max_yes_ask: int,
    require_two_sided: bool,
    live_only: bool,
    live_window_hours: int,
) -> ScanLoopResult:
    end_at = time.monotonic() + max(0, int(duration_seconds))
    interval = max(0.0, float(interval_seconds))

    runs = 0
    total_candidates = 0
    runs_with_candidates = 0
    max_candidates_in_run = 0

    while time.monotonic() < end_at:
        runs += 1
        candidates = await scan_sports_arbitrage(
            client=client,
            category=category,
            min_edge_cents=min_edge_cents,
            min_liquidity=min_liquidity,
            min_volume_24h=min_volume_24h,
            min_yes_bid=min_yes_bid,
            min_yes_ask=min_yes_ask,
            max_yes_ask=max_yes_ask,
            require_two_sided=require_two_sided,
            live_only=live_only,
            live_window_hours=live_window_hours,
            print_candidates=False,
        )

        n = len(candidates)
        total_candidates += n
        if n > 0:
            runs_with_candidates += 1
        max_candidates_in_run = max(max_candidates_in_run, n)

        # Wait until next run, but don't oversleep past end.
        if interval > 0:
            remaining = end_at - time.monotonic()
            await asyncio.sleep(min(interval, max(0.0, remaining)))

    return ScanLoopResult(
        runs=runs,
        total_candidates=total_candidates,
        runs_with_candidates=runs_with_candidates,
        max_candidates_in_run=max_candidates_in_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated sports arbitrage scan (summary counts)")

    parser.add_argument("--duration-seconds", type=int, default=300, help="Total runtime (default: 300)")
    parser.add_argument("--interval-seconds", type=int, default=30, help="Sleep between scans (default: 30)")

    # Match tradebot scan defaults
    parser.add_argument("--category", default="Sports")
    parser.add_argument("--min-edge", type=int, default=1)
    parser.add_argument("--min-liquidity", type=int, default=1)
    parser.add_argument("--min-volume24h", type=int, default=1)
    parser.add_argument("--min-bid", type=int, default=1)
    parser.add_argument("--min-ask", type=int, default=2)
    parser.add_argument("--max-ask", type=int, default=98)
    parser.add_argument("--no-require-two-sided", action="store_true")

    parser.add_argument("--include-non-live", action="store_true")
    parser.add_argument("--live-window-hours", type=int, default=6)

    args = parser.parse_args()

    async def runner() -> None:
        settings = Settings.load()
        client = KalshiClient.from_settings(settings)
        try:
            result = await run_scan_loop(
                client=client,
                category=args.category,
                duration_seconds=args.duration_seconds,
                interval_seconds=args.interval_seconds,
                min_edge_cents=args.min_edge,
                min_liquidity=args.min_liquidity,
                min_volume_24h=args.min_volume24h,
                min_yes_bid=args.min_bid,
                min_yes_ask=args.min_ask,
                max_yes_ask=args.max_ask,
                require_two_sided=not args.no_require_two_sided,
                live_only=not args.include_non_live,
                live_window_hours=args.live_window_hours,
            )
        finally:
            await client.aclose()

        avg = (result.total_candidates / result.runs) if result.runs else 0.0
        print("\n=== Scan loop summary ===")
        print(f"runs={result.runs}")
        print(f"total_candidates={result.total_candidates}")
        print(f"runs_with_candidates={result.runs_with_candidates}")
        print(f"max_candidates_in_run={result.max_candidates_in_run}")
        print(f"avg_candidates_per_run={avg:.2f}")

    asyncio.run(runner())


if __name__ == "__main__":
    main()
