"""Local analytics package for the Kalshi BTC 15m trading bot.

Non-goals:
- No Kalshi API calls inside analytics logic.
- No database.
- No order execution.

The Streamlit dashboard (see `analytics/app.py`) reads two CSV files:
- fills CSV: one row per fill
- settlements CSV: one row per resolved market

All functions are intended to be pure and testable.
"""
