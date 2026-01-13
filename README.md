# Tradebot (Kalshi)

Clean, minimal scaffold for a Kalshi trading bot:
- Public market data ingest (markets, orderbook, trades)
- Whale/large-trade alerts
- Market-making loop (quotes both sides)
- Signed authentication for trading endpoints

## Safety / disclaimer
This is tooling to interact with the Kalshi API. It does **not** guarantee profits. Market making and "arbitrage" carry risk (inventory, adverse selection, fees, outages). Start in demo, add strict risk limits, and monitor closely.

## Setup
1) Create a demo account: https://demo.kalshi.co/
2) Generate an API key + download the `.key` file (private key).
3) Create `.env` from `.env.example`.

## Install
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run (demo, public-only)
Fetch one open market + orderbook + recent trades:
```bash
python -m tradebot
```

## Run (demo, authenticated)
Set `KALSHI_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH` in `.env`, then:
```bash
python -m tradebot --mode mm
```

## Notes
- Kalshi signing: signature is base64(RSA-PSS-SHA256(timestamp + METHOD + path_without_query)).
- Demo base URL: `https://demo-api.kalshi.co`
- API paths include `/trade-api/v2/...`
