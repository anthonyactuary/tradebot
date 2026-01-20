from __future__ import annotations

import datetime
import asyncio
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception

from tradebot.config import Settings, env_defaults
from tradebot.kalshi.signing import KalshiAuth, create_kalshi_signature, load_rsa_private_key_from_file


Json = dict[str, Any]


def _is_retryable_exception(exc: BaseException) -> bool:
    # Network / timeout errors are usually retryable.
    if isinstance(exc, httpx.RequestError):
        return True

    # Only retry HTTP status errors that are plausibly transient.
    if isinstance(exc, httpx.HTTPStatusError):
        status = getattr(exc.response, "status_code", None)
        if status == 429:
            return True
        if isinstance(status, int) and 500 <= status <= 599:
            return True
        return False

    return False


@dataclass
class KalshiClient:
    base_url: str
    auth: KalshiAuth | None = None

    _client: httpx.AsyncClient | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "KalshiClient":
        defaults = env_defaults(settings.kalshi_env)
        base_url = (settings.kalshi_base_url or defaults.base_url).rstrip("/")

        auth: KalshiAuth | None = None
        if settings.kalshi_key_id and settings.kalshi_private_key_path:
            private_key = load_rsa_private_key_from_file(settings.kalshi_private_key_path)
            auth = KalshiAuth(key_id=settings.kalshi_key_id, private_key=private_key)

        return cls(base_url=base_url, auth=auth)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
        return self._client

    def _signed_headers(self, *, method: str, path: str) -> dict[str, str]:
        if self.auth is None:
            raise RuntimeError("Authenticated endpoint called but no KALSHI auth configured")
        timestamp_ms = str(int(datetime.datetime.now().timestamp() * 1000))
        sig = create_kalshi_signature(
            self.auth.private_key,
            timestamp_ms=timestamp_ms,
            method=method,
            path=path,
        )
        return {
            "KALSHI-ACCESS-KEY": self.auth.key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }

    @retry(
        wait=wait_exponential_jitter(initial=0.2, max=2.0),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(_is_retryable_exception),
    )
    async def request(
        self,
        method: Literal["GET", "POST", "DELETE", "PUT"],
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        auth_required: bool = False,
    ) -> Json:
        headers: dict[str, str] = {}
        if auth_required:
            headers.update(self._signed_headers(method=method, path=path))
        if json is not None:
            headers.setdefault("Content-Type", "application/json")

        url = f"{self.base_url}{path}"
        resp = await self._get_client().request(method, url, params=params, json=json, headers=headers)

        # Be polite with rate limits: if we get a 429, pause briefly and let tenacity retry.
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            delay = 1.0
            if retry_after:
                try:
                    delay = float(retry_after)
                except Exception:
                    delay = 1.0
            await asyncio.sleep(max(0.5, min(delay, 10.0)))

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Attach the response body to aid debugging (Kalshi often returns useful JSON here).
            body_preview = ""
            try:
                body_preview = resp.text
            except Exception:
                body_preview = ""
            if body_preview:
                raise httpx.HTTPStatusError(
                    f"{e} | body={body_preview}",
                    request=e.request,
                    response=e.response,
                ) from None
            raise
        # Some endpoints (especially DELETEs) may return 204 or an empty body.
        # Treat that as success with an empty payload.
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError:
            return {}

    # -------- Public endpoints --------

    async def get_series_list(
        self,
        *,
        category: str | None = None,
        tags: str | None = None,
        include_volume: bool = False,
        include_product_metadata: bool = False,
    ) -> list[Json]:
        params: dict[str, Any] = {
            "include_volume": include_volume,
            "include_product_metadata": include_product_metadata,
        }
        if category:
            params["category"] = category
        if tags:
            params["tags"] = tags

        data = await self.request("GET", "/trade-api/v2/series", params=params, auth_required=False)
        return list(data.get("series") or [])

    async def get_markets_page(
        self,
        *,
        limit: int = 1000,
        cursor: str | None = None,
        status: str | None = None,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        tickers: str | None = None,
        mve_filter: str | None = None,
    ) -> Json:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if tickers:
            params["tickers"] = tickers
        if mve_filter:
            params["mve_filter"] = mve_filter

        return await self.request("GET", "/trade-api/v2/markets", params=params, auth_required=False)

    async def get_markets(self, *, limit: int = 100, status: str | None = None) -> list[Json]:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        data = await self.request("GET", "/trade-api/v2/markets", params=params, auth_required=False)
        return list(data.get("markets") or [])

    async def get_filters_by_sport(self) -> Json:
        return await self.request("GET", "/trade-api/v2/search/filters_by_sport", auth_required=False)

    async def get_tags_by_categories(self) -> Json:
        return await self.request(
            "GET", "/trade-api/v2/search/tags_by_categories", auth_required=False
        )

    async def get_market(self, ticker: str) -> Json:
        """Get a single market by ticker. Returns market data including BBO."""
        return await self.request(
            "GET",
            f"/trade-api/v2/markets/{ticker}",
            auth_required=False,
        )

    async def get_orderbook(self, ticker: str) -> Json:
        return await self.request(
            "GET",
            f"/trade-api/v2/markets/{ticker}/orderbook",
            auth_required=False,
        )

    async def get_trades(
        self,
        *,
        limit: int = 100,
        cursor: str | None = None,
        ticker: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
    ) -> list[Json]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if ticker:
            params["ticker"] = ticker
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts
        data = await self.request("GET", "/trade-api/v2/markets/trades", params=params, auth_required=False)
        return list(data.get("trades") or [])

    # -------- Trading endpoints --------

    async def get_balance(self) -> Json:
        return await self.request("GET", "/trade-api/v2/portfolio/balance", auth_required=True)

    async def create_order(
        self,
        *,
        ticker: str,
        side: Literal["yes", "no"],
        action: Literal["buy", "sell"],
        count: int,
        order_type: Literal["limit", "market"] = "limit",
        yes_price: int | None = None,
        no_price: int | None = None,
        client_order_id: str | None = None,
        post_only: bool | None = None,
        reduce_only: bool | None = None,
        time_in_force: Literal["fill_or_kill", "good_till_canceled", "immediate_or_cancel"] | None = None,
        order_group_id: str | None = None,
    ) -> Json:
        body: dict[str, Any] = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": order_type,
        }
        if client_order_id:
            body["client_order_id"] = client_order_id
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        if post_only is not None:
            body["post_only"] = post_only
        if reduce_only is not None:
            body["reduce_only"] = reduce_only
        if time_in_force is not None:
            body["time_in_force"] = time_in_force
        if order_group_id is not None:
            body["order_group_id"] = order_group_id

        return await self.request(
            "POST",
            "/trade-api/v2/portfolio/orders",
            json=body,
            auth_required=True,
        )

    async def cancel_order(self, order_id: str) -> Json:
        return await self.request(
            "DELETE",
            f"/trade-api/v2/portfolio/orders/{order_id}",
            auth_required=True,
        )

    async def create_order_group(self, *, contracts_limit: int) -> Json:
        return await self.request(
            "POST",
            "/trade-api/v2/portfolio/order_groups/create",
            json={"contracts_limit": contracts_limit},
            auth_required=True,
        )

    async def get_positions(
        self,
        *,
        ticker: str | None = None,
        event_ticker: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        count_filter: str | None = None,
    ) -> Json:
        """Get portfolio positions.
        
        Args:
            ticker: Filter by market ticker
            event_ticker: Filter by event ticker (comma-separated, max 10)
            limit: Number of results (1-1000, default 100)
            cursor: Pagination cursor
            count_filter: Filter non-zero fields ('position', 'total_traded')
            
        Returns:
            {market_positions: [...], event_positions: [...], cursor: str}
            
            market_positions contains:
            - ticker: market ticker
            - position: current position (positive=long, negative=short)
            - market_exposure: exposure in cents
            - realized_pnl: realized P&L in cents
            - resting_orders_count: number of resting orders
        """
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if cursor:
            params["cursor"] = cursor
        if count_filter:
            params["count_filter"] = count_filter
            
        return await self.request(
            "GET",
            "/trade-api/v2/portfolio/positions",
            params=params,
            auth_required=True,
        )

    async def get_fills(
        self,
        *,
        ticker: str | None = None,
        order_id: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> Json:
        """Get order fills (matched trades).
        
        Args:
            ticker: Filter by market ticker
            order_id: Filter by order ID
            min_ts: Filter after this Unix timestamp
            max_ts: Filter before this Unix timestamp
            limit: Number of results (1-200, default 100)
            cursor: Pagination cursor
            
        Returns:
            {fills: [...], cursor: str}
            
            fills contains:
            - fill_id, trade_id, order_id
            - ticker, side, action, count, price
            - is_taker: whether we took liquidity
            - created_time: timestamp
        """
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if order_id:
            params["order_id"] = order_id
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts
        if cursor:
            params["cursor"] = cursor
            
        return await self.request(
            "GET",
            "/trade-api/v2/portfolio/fills",
            params=params,
            auth_required=True,
        )

    async def get_settlements(
        self,
        *,
        ticker: str | None = None,
        event_ticker: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> Json:
        """Get portfolio settlements (historical).

        Docs: https://docs.kalshi.com/api-reference/portfolio/get-settlements
        """
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts
        if cursor:
            params["cursor"] = cursor

        return await self.request(
            "GET",
            "/trade-api/v2/portfolio/settlements",
            params=params,
            auth_required=True,
        )

    async def get_orders_page(
        self,
        *,
        ticker: str | None = None,
        event_ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> Json:
        """List portfolio orders (paged).

        Note: This endpoint is not used by the bot during trading, but is helpful
        for post-run analysis and debugging.
        """
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor

        return await self.request(
            "GET",
            "/trade-api/v2/portfolio/orders",
            params=params,
            auth_required=True,
        )
