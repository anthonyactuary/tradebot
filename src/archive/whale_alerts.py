from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Sequence

from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient
from tradebot.kalshi.orderbook import iter_levels


@dataclass
class WhaleAlertService:
    client: KalshiClient
    settings: Settings

    async def run(self, *, tickers: Sequence[str]) -> None:
        log = logging.getLogger("tradebot.whales")

        last_trade_seen_ts: dict[str, float] = {t: 0.0 for t in tickers}

        while True:
            for ticker in tickers:
                # 1) Whale in trades
                try:
                    trades = await self.client.get_trades(limit=50, ticker=ticker)
                    for tr in trades:
                        # created_time is an ISO string; we do a simple monotonic gate using arrival time.
                        count = int(tr.get("count") or 0)
                        if count >= self.settings.whale_trade_count:
                            log.warning(
                                "WHALE_TRADE ticker=%s count=%s price=%s trade_id=%s",
                                ticker,
                                count,
                                tr.get("price"),
                                tr.get("trade_id"),
                            )
                    last_trade_seen_ts[ticker] = time.time()
                except Exception as e:  # keep alerts alive
                    log.exception("trade alert error ticker=%s: %s", ticker, e)

                # 2) Whale in orderbook (large resting size at any level)
                try:
                    ob = await self.client.get_orderbook(ticker)
                    for side in ("yes", "no"):
                        for price, qty in iter_levels(ob, side=side):
                            if qty >= self.settings.whale_book_level_qty:
                                log.warning(
                                    "WHALE_BOOK ticker=%s side=%s price=%s qty=%s",
                                    ticker,
                                    side,
                                    price,
                                    qty,
                                )
                except Exception as e:
                    log.exception("orderbook alert error ticker=%s: %s", ticker, e)

            await asyncio.sleep(self.settings.poll_seconds)
