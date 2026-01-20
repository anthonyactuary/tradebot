"""Debug market polling issue"""

import asyncio
import datetime as dt
from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient
from tradebot.tools.kalshi_market_poll import pick_active_markets, poll_once


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


async def debug_markets():
    settings = Settings.load()
    client = KalshiClient.from_settings(settings)
    
    try:
        now = _utcnow()
        print(f"Current UTC time: {now.isoformat()}")
        print(f"Current local time: {dt.datetime.now().isoformat()}")
        
        # Test pick_active_markets with the fix
        print(f"\n--- Testing pick_active_markets() ---")
        markets = await pick_active_markets(
            client,
            asset="BTC",
            horizon_minutes=60,
            limit_markets=5,
            min_seconds_to_expiry=0,
        )
        
        print(f"Found {len(markets)} active markets:")
        for m in markets:
            ticker = m.get("ticker", "")
            status = m.get("status", "")
            close_time = m.get("close_time", "")
            print(f"  {ticker} | status={status} | close={close_time}")
        
        # Also test poll_once
        print(f"\n--- Testing poll_once() ---")
        snapshots = await poll_once(
            client,
            asset="BTC",
            horizon_minutes=60,
            limit_markets=3,
        )
        
        print(f"Got {len(snapshots)} snapshots:")
        for snap in snapshots:
            print(f"  {snap.ticker} | TTE={snap.seconds_to_expiry}s | strike=${snap.price_to_beat}")
            print(f"    YES: bid={snap.best_yes_bid} ask={snap.best_yes_ask}")
            print(f"    NO:  bid={snap.best_no_bid} ask={snap.best_no_ask}")
        
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(debug_markets())
