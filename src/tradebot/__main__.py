from __future__ import annotations

import argparse
import asyncio

from tradebot.app import run_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Tradebot (Kalshi)")
    parser.add_argument(
        "--mode",
        choices=["demo", "mm", "scan", "crypto"],
        default="demo",
        help="demo: public market data fetch; mm: market-maker loop (auth required); scan: sports arbitrage scan; crypto: crypto 15m up/down scan",
    )
    parser.add_argument(
        "--category",
        default="Sports",
        help="Series category to scan (default: sports)",
    )
    parser.add_argument(
        "--min-edge",
        type=int,
        default=1,
        help="Minimum theoretical edge in cents to print (default: 1)",
    )
    parser.add_argument(
        "--min-liquidity",
        type=int,
        default=1,
        help="Minimum market liquidity to include (default: 1)",
    )
    parser.add_argument(
        "--min-volume24h",
        type=int,
        default=1,
        help="Minimum 24h volume to include (default: 1)",
    )
    parser.add_argument(
        "--min-bid",
        type=int,
        default=1,
        help="Minimum YES bid in cents to include (default: 1)",
    )
    parser.add_argument(
        "--min-ask",
        type=int,
        default=2,
        help="Minimum YES ask in cents to include (default: 2)",
    )
    parser.add_argument(
        "--max-ask",
        type=int,
        default=98,
        help="Maximum YES ask in cents to include (default: 98)",
    )
    parser.add_argument(
        "--no-require-two-sided",
        action="store_true",
        help="If set, do not require both YES and NO bids (default: require)",
    )
    parser.add_argument(
        "--include-non-live",
        action="store_true",
        help="If set, include non-live sports markets (futures/awards/long-dated). Default: live-only.",
    )
    parser.add_argument(
        "--live-window-hours",
        type=int,
        default=12,
        help="When live-only, include markets with expected expiration within N hours (default: 12)",
    )

    parser.add_argument(
        "--crypto-assets",
        default="BTC,ETH,SOL",
        help="Comma-separated assets for crypto scan mode (default: BTC,ETH,SOL)",
    )
    parser.add_argument(
        "--crypto-horizon-minutes",
        type=int,
        default=60,
        help="For crypto scan mode, include markets expiring within N minutes (default: 60)",
    )
    parser.add_argument(
        "--crypto-per-asset",
        type=int,
        default=8,
        help="For crypto scan mode, print up to N markets per asset (default: 8)",
    )
    parser.add_argument(
        "--crypto-require-quotes",
        action="store_true",
        help="For crypto scan mode, only include markets with valid YES bid+ask quotes",
    )
    parser.add_argument(
        "--crypto-orderbook-levels",
        type=int,
        default=0,
        help="For crypto scan mode, print top N orderbook levels for YES/NO (default: 0)",
    )
    args = parser.parse_args()

    asyncio.run(
        run_app(
            mode=args.mode,
            category=args.category,
            min_edge_cents=args.min_edge,
            min_liquidity=args.min_liquidity,
            min_volume_24h=args.min_volume24h,
            min_yes_bid=args.min_bid,
            min_yes_ask=args.min_ask,
            max_yes_ask=args.max_ask,
            require_two_sided=not args.no_require_two_sided,
            live_only=not args.include_non_live,
            live_window_hours=args.live_window_hours,
            crypto_assets=args.crypto_assets,
            crypto_horizon_minutes=args.crypto_horizon_minutes,
            crypto_per_asset=args.crypto_per_asset,
            crypto_require_quotes=args.crypto_require_quotes,
            crypto_orderbook_levels=args.crypto_orderbook_levels,
        )
    )


if __name__ == "__main__":
    main()
