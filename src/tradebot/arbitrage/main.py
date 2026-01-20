"""
Main entrypoint for BTC 15m Delayed Update Arbitrage strategy.

Run:
    python -m tradebot.arbitrage.main
    python -m tradebot.arbitrage.main --dry-run
    python -m tradebot.arbitrage.main --dry-run --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient
from tradebot.arbitrage.spot import (
    SyntheticSpot,
    EwmaVariance,
    VolConfig,
    PFairEstimator,
    now_ms,
)
from tradebot.arbitrage.strategy import (
    StrategyConfig,
    MarketInfo,
    RiskManager,
    DelayedUpdateArbStrategy,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# Silence noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ---------------------------
# Configuration
# ---------------------------

@dataclass(frozen=True)
class MainConfig:
    """Top-level configuration."""
    # Polling intervals
    spot_poll_interval_s: float = 0.25      # How often to update spot prices
    strategy_poll_interval_s: float = 0.35  # How often to tick strategy
    
    # Market discovery
    series_ticker: str = "KXBTC15M"
    min_seconds_to_expiry: int = 90
    max_seconds_to_expiry: int = 900  # 15 minutes


CONFIG = MainConfig()
STRATEGY_CONFIG = StrategyConfig()
VOL_CONFIG = VolConfig()


# ---------------------------
# Spot Feed
# ---------------------------

# Re-use existing exchange fetchers from kalshi_market_poll
from tradebot.tools.kalshi_market_poll import (
    fetch_coinbase_best_bid_ask,
    fetch_kraken_best_bid_ask,
)


async def fetch_coinbase_mid() -> Optional[float]:
    """Fetch current BTC mid price from Coinbase."""
    _bid, _ask, mid = await fetch_coinbase_best_bid_ask()
    return mid


async def fetch_kraken_mid() -> Optional[float]:
    """Fetch current BTC mid price from Kraken."""
    _bid, _ask, mid = await fetch_kraken_best_bid_ask()
    return mid


async def spot_feed_loop(spot: SyntheticSpot, vol: EwmaVariance) -> None:
    """
    Continuously update spot prices and volatility estimate.
    
    This loop should run as fast as possible (every 200-500ms) to detect
    when spot moves before Kalshi updates.
    """
    log.info("Starting spot feed loop (interval=%.2fs)", CONFIG.spot_poll_interval_s)
    
    while True:
        try:
            # Fetch spot prices from exchanges (parallel)
            cb_mid, kr_mid = await asyncio.gather(
                fetch_coinbase_mid(),
                fetch_kraken_mid(),
            )
            
            if cb_mid is not None:
                spot.update_coinbase_mid(cb_mid)
            if kr_mid is not None:
                spot.update_kraken_mid(kr_mid)

            # Update volatility estimate
            mid = spot.mid()
            if mid is not None:
                vol.update(now_ms(), mid)
                
        except Exception as e:
            log.warning("SPOT_FEED_ERROR error=%s", e)

        await asyncio.sleep(CONFIG.spot_poll_interval_s)


# ---------------------------
# Market Discovery
# ---------------------------

async def discover_active_markets(kalshi: KalshiClient) -> List[MarketInfo]:
    """
    Discover active BTC 15m markets from Kalshi.
    
    Returns markets that are:
    - In the KXBTC15M series
    - Status is active/initialized
    - Within trading window (not too early, not too late)
    """
    try:
        now_ts = int(time.time())
        min_close = now_ts + CONFIG.min_seconds_to_expiry
        max_close = now_ts + CONFIG.max_seconds_to_expiry
        
        resp = await kalshi.get_markets_page(
            limit=50,
            series_ticker=CONFIG.series_ticker,
            mve_filter="exclude",
        )
        
        markets = resp.get("markets", [])
        result: List[MarketInfo] = []
        
        for m in markets:
            status = m.get("status", "")
            if status not in {"active", "initialized", "open"}:
                continue
                
            ticker = m.get("ticker", "")
            if not ticker:
                continue
            
            # Parse close time
            close_time_str = m.get("close_time", "")
            if not close_time_str:
                continue
            try:
                # Parse ISO format: "2026-01-18T19:00:00Z"
                from datetime import datetime, timezone
                close_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                close_ts_ms = int(close_dt.timestamp() * 1000)
            except Exception:
                continue
            
            # Check if within trading window
            close_ts_s = close_ts_ms // 1000
            if close_ts_s < min_close or close_ts_s > max_close:
                continue
            
            # Parse strike from yes_sub_title: "Price to beat: $95,344.22"
            strike = m.get("floor_strike")
            if strike is None:
                strike = m.get("custom_strike")
            if strike is None:
                # Parse from yes_sub_title
                sub = m.get("yes_sub_title", "")
                if "Price to beat:" in sub:
                    import re
                    match = re.search(r"\$([\d,]+\.?\d*)", sub)
                    if match:
                        try:
                            strike = float(match.group(1).replace(",", ""))
                        except Exception:
                            pass
            if strike is None:
                continue
            try:
                strike_f = float(strike)
            except Exception:
                continue
            
            result.append(MarketInfo(
                ticker=ticker,
                strike=strike_f,
                expiry_ts_ms=close_ts_ms,
            ))
        
        return result
        
    except Exception as e:
        log.warning("MARKET_DISCOVERY_ERROR error=%s", e)
        return []


# ---------------------------
# Strategy Loop
# ---------------------------

async def strategy_loop(kalshi: KalshiClient, strat: DelayedUpdateArbStrategy) -> None:
    """
    Main strategy loop: discover markets and tick strategy.
    """
    log.info("Starting strategy loop (interval=%.2fs)", CONFIG.strategy_poll_interval_s)
    
    while True:
        try:
            markets = await discover_active_markets(kalshi)
            
            if markets:
                log.debug("Found %d active markets", len(markets))
                
            for m in markets:
                try:
                    await strat.tick_market(m)
                except Exception as e:
                    log.warning("TICK_ERROR %s error=%s", m.ticker, e)
                    
        except Exception as e:
            log.warning("STRATEGY_LOOP_ERROR error=%s", e)

        await asyncio.sleep(CONFIG.strategy_poll_interval_s)


# ---------------------------
# Main
# ---------------------------

async def main(dry_run: bool = False, verbose: bool = False) -> None:
    mode_str = "DRY_RUN" if dry_run else "LIVE"
    log.info("Starting BTC 15m Delayed Update Arb mode=%s", mode_str)
    log.info("Strategy config: %s", STRATEGY_CONFIG)
    
    if dry_run:
        log.info("*** DRY RUN MODE - NO REAL ORDERS WILL BE PLACED ***")
    
    # Initialize components
    spot = SyntheticSpot()
    vol = EwmaVariance(cfg=VOL_CONFIG)
    p_est = PFairEstimator(spot=spot, vol=vol)
    
    # Initialize Kalshi client
    settings = Settings.load()
    kalshi = KalshiClient.from_settings(settings)
    
    # Initialize strategy
    risk = RiskManager(STRATEGY_CONFIG)
    strat = DelayedUpdateArbStrategy(
        cfg=STRATEGY_CONFIG,
        kalshi=kalshi,
        p_est=p_est,
        risk=risk,
        dry_run=dry_run,
        verbose=verbose,
    )
    
    try:
        # Run both loops concurrently
        await asyncio.gather(
            spot_feed_loop(spot, vol),
            strategy_loop(kalshi, strat),
        )
    finally:
        await kalshi.aclose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BTC 15m Delayed Update Arbitrage Bot"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no real orders)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose decision logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(main(dry_run=args.dry_run, verbose=args.verbose))
    except KeyboardInterrupt:
        log.info("Stopped")
