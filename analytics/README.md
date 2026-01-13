# Kalshi BTC 15m Analytics Dashboard

Local-only Streamlit dashboard that reads:
- fills CSV (`--fills_csv`)
- settlements CSV (`--settlements_csv`)

No database. No live trading. No Kalshi API calls in analytics modules.

## Install

- `pip install -r requirements.txt`
- Ensure `streamlit` and `pandas` are installed.

## Run

From repo root:

- `streamlit run analytics/app.py -- --fills_csv runs/kalshi_fills.csv --settlements_csv runs/kalshi_settlements.csv`

One-click (VS Code):

- Run [analytics/run_app.py](analytics/run_app.py) with the Python "Run" button.

Optional auto-refresh:

- `streamlit run analytics/app.py -- --auto_refresh_seconds 30`

Optional refresh button command (example using the archived Kalshi report script):

If you leave `--refresh_cmd` blank, the dashboard will try to run the default refresh:

- `python analytics/kalshi_activity_report.py --minutes 1440 --fills-csv ... --settlements-csv ...`

If you want a custom refresh command, pass it explicitly:

- `streamlit run analytics/app.py -- \
    --fills_csv runs/kalshi_fills.csv \
    --settlements_csv runs/kalshi_settlements.csv \
    --refresh_cmd "python src/archive/kalshi_activity_report.py --minutes 1440 --fills-csv runs/kalshi_fills.csv --settlements-csv runs/kalshi_settlements.csv"`
