from __future__ import annotations

import asyncio
import logging
from typing import Sequence

from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient
from tradebot.kalshi.orderbook import compute_spread
from tradebot.services.whale_alerts import WhaleAlertService
from tradebot.strategies.market_maker import MarketMaker
from tradebot.tools.arbitrage_scanner import scan_sports_arbitrage
from tradebot.tools.crypto_scanner import scan_crypto_15m_markets


async def run_app(
    *,
    mode: str,
    category: str = "Sports",
    min_edge_cents: int = 1,
    min_liquidity: int = 1,
    min_volume_24h: int = 1,
    min_yes_bid: int = 1,
    min_yes_ask: int = 2,
    max_yes_ask: int = 98,
    require_two_sided: bool = True,
    live_only: bool = True,
    live_window_hours: int = 12,

    crypto_assets: str = "BTC,ETH,SOL",
    crypto_horizon_minutes: int = 60,
    crypto_per_asset: int = 8,
    crypto_require_quotes: bool = False,
    crypto_orderbook_levels: int = 0,
) -> None:
    settings = Settings.load()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Keep scan output readable (httpx can be very chatty at INFO).
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    client = KalshiClient.from_settings(settings)
    logging.getLogger("tradebot").info(
        "Kalshi env=%s base_url=%s", settings.kalshi_env, client.base_url
    )

    if mode == "scan":
        await scan_sports_arbitrage(
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
        )
        return

    if mode == "crypto":
        assets = [a.strip() for a in (crypto_assets or "").split(",") if a.strip()]
        await scan_crypto_15m_markets(
            client=client,
            assets=assets,
            horizon_minutes=crypto_horizon_minutes,
            per_asset=crypto_per_asset,
            min_liquidity=min_liquidity,
            min_volume_24h=min_volume_24h,
            require_quotes=crypto_require_quotes,
            orderbook_levels=crypto_orderbook_levels,
        )
        return

    tickers: Sequence[str] = settings.market_tickers
    if not tickers:
        # Minimal default: pick the first open market
        markets = await client.get_markets(limit=1, status="open")
        if not markets:
            raise RuntimeError("No open markets returned")
        tickers = [markets[0]["ticker"]]

    if mode == "demo":
        ticker = tickers[0]
        ob = await client.get_orderbook(ticker)
        spread = compute_spread(ob)
        trades = await client.get_trades(limit=5, ticker=ticker)

        logging.getLogger("tradebot").info(
            "Ticker=%s spread=%s recent_trades=%d", ticker, spread, len(trades)
        )
        return

    if mode == "mm":
        # Alert service runs alongside the market maker.
        whale_service = WhaleAlertService(client=client, settings=settings)
        mm = MarketMaker(client=client, settings=settings)

        await asyncio.gather(
            whale_service.run(tickers=tickers),
            mm.run(tickers=tickers),
        )
        return

    raise ValueError(f"Unknown mode: {mode}")
