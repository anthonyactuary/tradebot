from __future__ import annotations

import argparse
import asyncio
import datetime
import logging
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient
from tradebot.kalshi.orderbook import compute_best_prices


# ----------------------------
# Core math / probability model
# ----------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class FairProbabilityModel:
    """Simple, deterministic p_fair estimator.

    We intentionally avoid any complex predictors.
    This uses ONLY contract price dynamics (not underlying BTC/ETH/SOL spot).

    Signals:
    - Start at 0.5
    - Momentum: short-term return of the mid price
    - Mild mean reversion: penalize recent overextension away from 0.5

    Output is clamped to [0.05, 0.95] for safety.
    """

    alpha: float = 1.25
    beta: float = 0.75

    def compute(self, *, mid_now: float, mid_prev: float | None, ema_mid: float | None) -> float:
        base = 0.5

        # Short-term return in probability-space.
        # mid is already in [0,1], so returns are small.
        short_term_return = 0.0
        if mid_prev is not None and mid_prev > 0:
            short_term_return = (mid_now - mid_prev) / max(1e-6, mid_prev)

        # Overextension: distance from an EMA anchor.
        # If we don't have an EMA yet, use 0.5 as the anchor.
        anchor = ema_mid if ema_mid is not None else 0.5
        recent_overextension = mid_now - anchor

        p = base + (self.alpha * short_term_return) - (self.beta * recent_overextension)
        return _clamp(p, 0.05, 0.95)


# ----------------------------
# Volatility & spread selection
# ----------------------------

@dataclass
class VolatilityEstimator:
    """Estimate short-term volatility from mid-price returns.

    Uses a small rolling window of returns and reports an RMS value.
    This is fast, stable, and easy to reason about.
    """

    window: int = 20
    _returns: deque[float] = field(default_factory=lambda: deque(maxlen=20))

    def update(self, *, mid_now: float, mid_prev: float | None) -> float:
        if mid_prev is not None and mid_prev > 0:
            r = (mid_now - mid_prev) / max(1e-6, mid_prev)
            self._returns.append(r)
        return self.value()

    def value(self) -> float:
        if not self._returns:
            return 0.0
        ms = sum(r * r for r in self._returns) / len(self._returns)
        return math.sqrt(ms)


@dataclass
class SpreadPolicy:
    """Volatility-aware spread adjustment.

    - Low vol => tighter spread
    - High vol => wider spread

    We express spreads in probability units (0..1), then convert to cents.
    """

    base_spread: float = 0.05  # 5 cents
    vol_k: float = 8.0
    min_spread: float = 0.04
    max_spread: float = 0.12

    def compute(self, *, volatility: float) -> float:
        spread = self.base_spread * (1.0 + self.vol_k * volatility)
        return _clamp(spread, self.min_spread, self.max_spread)


# ----------------------------
# Risk limits / sizing
# ----------------------------

@dataclass
class RiskLimits:
    bankroll_dollars: float = 100.0

    # Hard caps (requirements)
    max_exposure_per_market_dollars: float = 10.0
    max_exposure_total_dollars: float = 25.0

    # Order sizing rules
    max_single_order_pct_bankroll: float = 0.02  # 2%
    base_order_dollars: float = 1.50  # default 1-2 dollars

    # As we approach expiration, we stop quoting.
    stop_quoting_final_seconds: int = 60


def _max_loss_yes_buy(*, yes_price_cents: int, count: int) -> float:
    # Buy YES at p: worst-case loss = p per contract (if it expires 0).
    return (yes_price_cents / 100.0) * count


def _max_loss_yes_sell(*, yes_price_cents: int, count: int) -> float:
    # Sell YES at p: worst-case loss = (1 - p) per contract (if it expires 1).
    return ((100 - yes_price_cents) / 100.0) * count


@dataclass
class ExposureTracker:
    """Conservative exposure proxy.

    We do NOT rely on portfolio positions because this repo doesn't currently
    implement a positions endpoint. Instead, we bound exposure using the
    worst-case loss of our *resting orders*.

    This is intentionally conservative: it helps ensure we don't blow up even
    if multiple orders get filled quickly.
    """

    limits: RiskLimits

    # Track open (resting) orders we placed.
    # Map: ticker -> set(order_ids)
    open_orders: dict[str, set[str]] = field(default_factory=dict)

    # Track worst-case exposure contribution per order_id.
    # Map: order_id -> dollars risk
    order_risk: dict[str, float] = field(default_factory=dict)

    def total_open_risk(self) -> float:
        return float(sum(self.order_risk.values()))

    def open_risk_for_ticker(self, ticker: str) -> float:
        ids = self.open_orders.get(ticker) or set()
        return float(sum(self.order_risk.get(oid, 0.0) for oid in ids))

    def can_add_risk(self, *, ticker: str, add_risk: float) -> bool:
        if add_risk <= 0:
            return True
        if self.open_risk_for_ticker(ticker) + add_risk > self.limits.max_exposure_per_market_dollars:
            return False
        if self.total_open_risk() + add_risk > self.limits.max_exposure_total_dollars:
            return False
        return True

    def register_order(self, *, ticker: str, order_id: str, risk_dollars: float) -> None:
        self.open_orders.setdefault(ticker, set()).add(order_id)
        self.order_risk[order_id] = float(max(0.0, risk_dollars))

    def forget_order(self, *, ticker: str, order_id: str) -> None:
        if ticker in self.open_orders:
            self.open_orders[ticker].discard(order_id)
            if not self.open_orders[ticker]:
                self.open_orders.pop(ticker, None)
        self.order_risk.pop(order_id, None)


# ----------------------------
# Market discovery helpers
# ----------------------------

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


def _discover_series_ticker(asset: str) -> str:
    # Kalshi naming convention we observed: KXBTC15M / KXETH15M / KXSOL15M
    return f"KX{asset.upper()}15M"


async def pick_active_markets(
    *,
    client: KalshiClient,
    assets: list[str],
    horizon_minutes: int,
    per_asset: int,
) -> dict[str, list[dict[str, Any]]]:
    """Return a small list of soon-to-expire markets for each asset."""

    now = datetime.datetime.now(datetime.timezone.utc)
    horizon = datetime.timedelta(minutes=max(1, horizon_minutes))

    out: dict[str, list[dict[str, Any]]] = {}
    for asset in assets:
        series_ticker = _discover_series_ticker(asset)
        page = await client.get_markets_page(
            limit=1000,
            status="open",
            series_ticker=series_ticker,
            mve_filter="exclude",
        )
        markets = list(page.get("markets") or [])

        # Filter to near-term contracts.
        keep: list[tuple[datetime.datetime, dict[str, Any]]] = []
        for m in markets:
            exp = _to_dt(m.get("expected_expiration_time"))
            if exp is None:
                continue
            if exp < now:
                continue
            if exp - now > horizon:
                continue
            keep.append((exp, m))

        keep.sort(key=lambda t: t[0])
        out[asset] = [m for _, m in keep[: max(1, per_asset)]]

    return out


# ----------------------------
# The actual strategy
# ----------------------------

@dataclass
class Crypto15mMarketMaker:
    client: KalshiClient
    limits: RiskLimits

    model: FairProbabilityModel = field(default_factory=FairProbabilityModel)
    vol: VolatilityEstimator = field(default_factory=VolatilityEstimator)
    spread_policy: SpreadPolicy = field(default_factory=SpreadPolicy)

    # Per-ticker state
    prev_mid: dict[str, float] = field(default_factory=dict)
    ema_mid: dict[str, float] = field(default_factory=dict)

    exposure: ExposureTracker = field(init=False)

    def __post_init__(self) -> None:
        self.exposure = ExposureTracker(self.limits)

    async def _cancel_known_orders(self, *, ticker: str, log: logging.Logger) -> None:
        # We don't have a "list orders" endpoint wired, so we cancel only what we created.
        # This is a conservative approach that prevents stale quotes from lingering.
        ids = list(self.exposure.open_orders.get(ticker) or [])
        for oid in ids:
            try:
                await self.client.cancel_order(oid)
            except Exception:
                # If already filled/canceled, cancel may fail. We still forget it.
                pass
            finally:
                self.exposure.forget_order(ticker=ticker, order_id=oid)

    def _choose_size(self, *, volatility: float) -> int:
        # Bankroll protection:
        # - Default order size: ~$1-$2
        # - Never > 2% of bankroll
        # - Reduce size automatically when vol is high
        max_single = self.limits.bankroll_dollars * self.limits.max_single_order_pct_bankroll
        base = min(self.limits.base_order_dollars, max_single)

        # Simple reduction: if volatility is elevated, reduce size.
        # This is deliberately blunt.
        if volatility >= 0.02:
            base *= 0.5
        if volatility >= 0.05:
            base *= 0.5

        # Convert dollars to contracts (each contract notionally $1).
        # Always at least 1 contract.
        return max(1, int(round(base)))

    def _round_price_cents(self, p: float) -> int:
        # p is probability in [0,1]. Convert to cents and round.
        cents = int(round(p * 100.0))
        return max(1, min(99, cents))

    def _apply_inventory_pressure(
        self,
        *,
        p_fair: float,
        spread: float,
        # We do not have true inventory; use open risk as a conservative proxy.
        open_risk: float,
    ) -> tuple[float, float]:
        """Shift bid/ask slightly based on current open risk.

        If we already have a lot of open risk in this market, we quote less aggressively:
        - lower the bid
        - raise the ask

        This avoids piling into exposure.
        """

        pressure = _clamp(open_risk / max(1e-6, self.limits.max_exposure_per_market_dollars), 0.0, 1.0)

        # At full pressure, shift each side by up to 1 cent (0.01) away from mid.
        shift = 0.01 * pressure

        bid = p_fair - (spread / 2.0) - shift
        ask = p_fair + (spread / 2.0) + shift
        return bid, ask

    async def quote_one(
        self,
        *,
        ticker: str,
        expected_expiration_time: datetime.datetime | None,
        log: logging.Logger,
    ) -> None:
        # 1) Time-to-resolution logic: stop quoting in final moments.
        now = datetime.datetime.now(datetime.timezone.utc)
        if expected_expiration_time is not None:
            seconds_left = int((expected_expiration_time - now).total_seconds())
            if seconds_left <= self.limits.stop_quoting_final_seconds:
                await self._cancel_known_orders(ticker=ticker, log=log)
                log.info("Skip quoting (near expiry) ticker=%s seconds_left=%s", ticker, seconds_left)
                return

        # 2) Pull best prices from orderbook.
        ob = await self.client.get_orderbook(ticker)
        prices = compute_best_prices(ob)
        if prices.best_yes_bid is None or prices.best_yes_ask is None:
            return

        # Mid probability as our basic state variable.
        mid_now = (prices.best_yes_bid + prices.best_yes_ask) / 200.0

        # Update EMA anchor (mean reversion reference).
        prev_ema = self.ema_mid.get(ticker)
        ema = mid_now if prev_ema is None else (0.9 * prev_ema + 0.1 * mid_now)
        self.ema_mid[ticker] = ema

        # Momentum uses previous mid.
        mid_prev = self.prev_mid.get(ticker)
        self.prev_mid[ticker] = mid_now

        # 3) Estimate volatility and spread.
        volatility = self.vol.update(mid_now=mid_now, mid_prev=mid_prev)
        spread = self.spread_policy.compute(volatility=volatility)

        # 4) Compute fair probability.
        p_fair = self.model.compute(mid_now=mid_now, mid_prev=mid_prev, ema_mid=ema)

        # 5) Cancel stale orders from prior iteration.
        await self._cancel_known_orders(ticker=ticker, log=log)

        # 6) Choose size with risk-first sizing.
        count = self._choose_size(volatility=volatility)

        # 7) Inventory / exposure pressure.
        open_risk = self.exposure.open_risk_for_ticker(ticker)
        bid_p, ask_p = self._apply_inventory_pressure(p_fair=p_fair, spread=spread, open_risk=open_risk)

        # Convert to cents.
        bid_c = self._round_price_cents(bid_p)
        ask_c = self._round_price_cents(ask_p)

        # Never cross the spread (maker behavior).
        if bid_c >= ask_c:
            # If our model got too tight, widen by 1 cent.
            bid_c = max(1, min(bid_c, ask_c - 1))

        # Also never cross the current book.
        # - We place bid <= best_yes_bid (or slightly below) to avoid taking.
        # - We place ask >= best_yes_ask (or slightly above) to avoid taking.
        bid_c = min(bid_c, prices.best_yes_bid)
        ask_c = max(ask_c, prices.best_yes_ask)
        if bid_c >= ask_c:
            return

        # 8) Risk checks using worst-case loss bounds.
        # We treat each order as standalone risk.
        buy_risk = _max_loss_yes_buy(yes_price_cents=bid_c, count=count)
        sell_risk = _max_loss_yes_sell(yes_price_cents=ask_c, count=count)

        # Enforce hard exposure caps.
        if not self.exposure.can_add_risk(ticker=ticker, add_risk=buy_risk + sell_risk):
            log.info(
                "Risk cap reached ticker=%s open_risk=%.2f add_risk=%.2f total_open_risk=%.2f",
                ticker,
                self.exposure.open_risk_for_ticker(ticker),
                (buy_risk + sell_risk),
                self.exposure.total_open_risk(),
            )
            return

        post_only = True  # market-making, never take

        # 9) Place a YES bid and a YES ask.
        # We use unique client_order_id for idempotency/traceability.
        buy_resp = await self.client.create_order(
            ticker=ticker,
            side="yes",
            action="buy",
            count=count,
            order_type="limit",
            yes_price=bid_c,
            client_order_id=str(uuid.uuid4()),
            post_only=post_only,
        )
        buy_order_id = str(buy_resp.get("order_id") or buy_resp.get("id") or "")
        if buy_order_id:
            self.exposure.register_order(ticker=ticker, order_id=buy_order_id, risk_dollars=buy_risk)

        sell_resp = await self.client.create_order(
            ticker=ticker,
            side="yes",
            action="sell",
            count=count,
            order_type="limit",
            yes_price=ask_c,
            client_order_id=str(uuid.uuid4()),
            post_only=post_only,
        )
        sell_order_id = str(sell_resp.get("order_id") or sell_resp.get("id") or "")
        if sell_order_id:
            self.exposure.register_order(ticker=ticker, order_id=sell_order_id, risk_dollars=sell_risk)

        log.info(
            "Quoted ticker=%s mid=%.3f p_fair=%.3f vol=%.4f spread=%.3f bid=%dc ask=%dc size=%d open_risk=%.2f",
            ticker,
            mid_now,
            p_fair,
            volatility,
            spread,
            bid_c,
            ask_c,
            count,
            self.exposure.open_risk_for_ticker(ticker),
        )

    async def run(
        self,
        *,
        assets: list[str],
        horizon_minutes: int,
        per_asset: int,
        poll_seconds: float,
    ) -> None:
        log = logging.getLogger("tradebot.crypto_mm")

        # Basic sanity: verify auth works.
        bal = await self.client.get_balance()
        log.info("Authenticated. Balance cents=%s", bal.get("balance"))

        # Main loop.
        while True:
            started = time.monotonic()

            try:
                selection = await pick_active_markets(
                    client=self.client,
                    assets=assets,
                    horizon_minutes=horizon_minutes,
                    per_asset=per_asset,
                )

                for asset, markets in selection.items():
                    for m in markets:
                        ticker = str(m.get("ticker"))
                        exp = _to_dt(m.get("expected_expiration_time"))
                        if not ticker:
                            continue
                        try:
                            await self.quote_one(ticker=ticker, expected_expiration_time=exp, log=log)
                        except Exception as e:
                            # Never let one market crash the bot.
                            log.exception("quote error asset=%s ticker=%s: %s", asset, ticker, e)

            except Exception as e:
                log.exception("crypto mm loop error: %s", e)

            # Keep to a steady cadence.
            elapsed = time.monotonic() - started
            sleep_for = max(0.0, float(poll_seconds) - elapsed)
            await asyncio.sleep(sleep_for)


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservative market-maker for Kalshi crypto 15m markets")
    parser.add_argument("--assets", default="BTC,ETH,SOL", help="Comma-separated assets (default: BTC,ETH,SOL)")
    parser.add_argument("--horizon-minutes", type=int, default=60, help="Pick markets expiring within N minutes")
    parser.add_argument("--per-asset", type=int, default=1, help="How many markets per asset to quote")
    parser.add_argument("--poll-seconds", type=float, default=3.0, help="Loop cadence")

    parser.add_argument("--bankroll", type=float, default=100.0, help="Total bankroll dollars")
    parser.add_argument("--base-order", type=float, default=1.5, help="Base order size in dollars (1-2 recommended)")

    args = parser.parse_args()

    settings = Settings.load()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Keep HTTP logs quiet.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    limits = RiskLimits(
        bankroll_dollars=float(args.bankroll),
        base_order_dollars=float(args.base_order),
    )

    client = KalshiClient.from_settings(settings)
    bot = Crypto15mMarketMaker(client=client, limits=limits)

    assets = [a.strip().upper() for a in (args.assets or "").split(",") if a.strip()]

    async def runner() -> None:
        try:
            await bot.run(
                assets=assets,
                horizon_minutes=int(args.horizon_minutes),
                per_asset=int(args.per_asset),
                poll_seconds=float(args.poll_seconds),
            )
        finally:
            await client.aclose()

    asyncio.run(runner())


if __name__ == "__main__":
    main()
