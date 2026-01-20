import asyncio
from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient

async def main():
    settings = Settings()
    kalshi = KalshiClient.from_settings(settings)
    
    # Get OPEN markets only
    markets = await kalshi.get_markets_page(series_ticker='KXBTC15M', status='open')
    btc = markets.get('markets', [])
    
    print(f'Found {len(btc)} OPEN markets')
    for m in btc[:1]:
        ticker = m['ticker']
        print(f'\n{ticker}')
        
        # Use the new get_market endpoint
        mkt_resp = await kalshi.get_market(ticker)
        mkt = mkt_resp.get('market', {})
        
        print(f'BBO from market endpoint:')
        print(f'  yes_bid: {mkt.get("yes_bid")} cents = ${mkt.get("yes_bid_dollars")}')
        print(f'  yes_ask: {mkt.get("yes_ask")} cents = ${mkt.get("yes_ask_dollars")}')
        print(f'  no_bid: {mkt.get("no_bid")} cents = ${mkt.get("no_bid_dollars")}')
        print(f'  no_ask: {mkt.get("no_ask")} cents = ${mkt.get("no_ask_dollars")}')
        
        # Verify: yes_bid + no_ask should ~ 100, yes_ask + no_bid should ~ 100
        yes_bid = mkt.get("yes_bid", 0)
        no_ask = mkt.get("no_ask", 0)
        yes_ask = mkt.get("yes_ask", 0)
        no_bid = mkt.get("no_bid", 0)
        
        print(f'\nSanity checks:')
        print(f'  yes_bid + no_ask = {yes_bid} + {no_ask} = {yes_bid + no_ask} (should be ~100)')
        print(f'  yes_ask + no_bid = {yes_ask} + {no_bid} = {yes_ask + no_bid} (should be ~100)')

asyncio.run(main())
