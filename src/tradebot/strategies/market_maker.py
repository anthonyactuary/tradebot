from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Sequence

from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient
from tradebot.kalshi.orderbook import compute_best_prices, compute_spread


@dataclass
class MarketMaker:
    client: KalshiClient
    settings: Settings

    async def run(self, *, tickers: Sequence[str]) -> None:
        log = logging.getLogger("tradebot.mm")

        # Basic sanity: verify auth works
        balance = await self.client.get_balance()
        log.info("Authenticated. Balance cents=%s", balance.get("balance"))

        # Optional: create an order group so a hard contracts limit can auto-cancel.
        og = await self.client.create_order_group(contracts_limit=max(1, self.settings.mm_order_count))
        order_group_id = og.get("order_group_id")
        log.info("Order group=%s", order_group_id)

        while True:
            for ticker in tickers:
                try:
                    ob = await self.client.get_orderbook(ticker)
                    spread = compute_spread(ob)
                    if spread is None:
                        continue
                    if spread < self.settings.mm_min_spread_cents:
                        continue

                    prices = compute_best_prices(ob)
                    if prices.best_yes_bid is None or prices.best_yes_ask is None:
                        continue

                    # Quote inside the spread on YES (buy at bid+improve, sell at ask-improve).
                    buy_price = min(99, prices.best_yes_bid + self.settings.mm_improve_cents)
                    sell_price = max(1, prices.best_yes_ask - self.settings.mm_improve_cents)

                    # Post-only to avoid crossing (maker behavior).
                    post_only = bool(self.settings.mm_post_only)

                    # Place YES buy
                    await self.client.create_order(
                        ticker=ticker,
                        side="yes",
                        action="buy",
                        count=self.settings.mm_order_count,
                        order_type="limit",
                        yes_price=buy_price,
                        client_order_id=str(uuid.uuid4()),
                        post_only=post_only,
                        order_group_id=order_group_id,
                    )

                    # Place YES sell
                    await self.client.create_order(
                        ticker=ticker,
                        side="yes",
                        action="sell",
                        count=self.settings.mm_order_count,
                        order_type="limit",
                        yes_price=sell_price,
                        client_order_id=str(uuid.uuid4()),
                        post_only=post_only,
                        order_group_id=order_group_id,
                    )

                    log.info(
                        "Quoted ticker=%s spread=%s buy_yes=%s sell_yes=%s",
                        ticker,
                        spread,
                        buy_price,
                        sell_price,
                    )

                except Exception as e:
                    log.exception("mm loop error ticker=%s: %s", ticker, e)

            await asyncio.sleep(self.settings.poll_seconds)
