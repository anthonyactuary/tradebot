from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

# Streamlit typically adds the script directory (analytics/) to sys.path, which
# breaks absolute imports like `from analytics...` because it would look for
# analytics/analytics/. Add repo root explicitly so imports resolve.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from analytics.equity_curve import build_equity_curve
from analytics.kpis import compute_kpis
from analytics.kalshi_metrics_fixed import (
    build_per_market_metrics,
    build_summary_kpis,
    load_fills_export,
    load_settlements_export,
)
from analytics.load_csvs import load_csvs
from analytics.roundtrips import reconstruct_roundtrips


def _atomic_write_csv(df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)


def _default_refresh_argv(*, fills_csv: str, settlements_csv: str) -> list[str] | None:
    # Prefer the local script colocated with the dashboard.
    here = Path(__file__).resolve().parent
    candidates = [
        here / "kalshi_activity_report.py",
        (here.parent / "src" / "archive" / "kalshi_activity_report.py"),
    ]

    script_path: Path | None = None
    for c in candidates:
        if c.exists():
            script_path = c
            break
    if script_path is None:
        return None

    # 1440 minutes = last 24h by default.
    return [
        sys.executable,
        str(script_path),
        "--minutes",
        "1440",
        "--fills-csv",
        str(fills_csv),
        "--settlements-csv",
        str(settlements_csv),
    ]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--fills_csv", type=str, default=os.environ.get("KALSHI_FILLS_CSV", "runs/kalshi_fills.csv"))
    p.add_argument(
        "--settlements_csv",
        type=str,
        default=os.environ.get("KALSHI_SETTLEMENTS_CSV", "runs/kalshi_settlements.csv"),
    )

    # Refresh button runs an external script to regenerate CSVs.
    p.add_argument(
        "--refresh_cmd",
        type=str,
        default=None,
        help="Shell command to run to refresh CSVs (e.g. python src/archive/kalshi_activity_report.py ...)",
    )
    p.add_argument(
        "--auto_refresh_seconds",
        type=int,
        default=int(os.environ.get("KALSHI_DASH_REFRESH_SECONDS", "30")),
    )

    # Streamlit passes app args after a '--' separator.
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    return p.parse_args(argv)


def _autorefresh(seconds: int) -> None:
    """Client-side refresh without extra deps."""

    seconds = int(seconds)
    if seconds <= 0:
        return

    st.components.v1.html(
        f"""
        <script>
          setTimeout(function() {{ window.location.reload(); }}, {seconds * 1000});
        </script>
        """,
        height=0,
    )


def _run_refresh_cmd(cmd: str) -> tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        ok = proc.returncode == 0
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return ok, out.strip()
    except Exception as e:
        return False, str(e)


def _run_refresh_argv(argv: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(argv, shell=False, capture_output=True, text=True, timeout=180)
        ok = proc.returncode == 0
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return ok, out.strip()
    except Exception as e:
        return False, str(e)


def main() -> None:
    st.set_page_config(page_title="Kalshi BTC 15m Bot Analytics", layout="wide")

    args = _parse_args()

    st.title("Kalshi BTC 15-minute Bot Performance")

    with st.sidebar:
        st.header("Data")
        fills_csv = st.text_input("fills_csv", value=str(args.fills_csv))
        settlements_csv = st.text_input("settlements_csv", value=str(args.settlements_csv))

        st.header("Filters")
        tz_mode = st.selectbox("Date range timezone", options=["UTC", "Mountain (MST/MDT)"], index=0)
        date_range = st.date_input("Date range", value=())
        side_filter = st.selectbox("Side", options=["ALL", "YES", "NO"], index=0)
        ticker_filter = st.text_input("Ticker contains", value="")
        strategy_filter = st.text_input("Strategy tag contains", value="")

        fee_source = st.selectbox("Fee source", options=["settlement", "fills", "both"], index=0)

        st.header("Refresh")
        auto_refresh = st.number_input(
            "Auto refresh seconds (0 disables)",
            min_value=0,
            max_value=3600,
            value=int(args.auto_refresh_seconds),
            step=5,
        )

        refresh_cmd = st.text_input(
            "Refresh command (optional)",
            value=str(args.refresh_cmd or ""),
            help="If blank, Refresh runs the default kalshi_activity_report.py command (last 24h).",
        )

        do_refresh = st.button("Refresh stats")

    # Where we export per-market metrics on refresh.
    per_market_export_path = Path("runs") / "kalshi_per_market_metrics_fixed.csv"

    if do_refresh:
        with st.spinner("Running refresh..."):
            if refresh_cmd.strip():
                ok, out = _run_refresh_cmd(refresh_cmd.strip())
            else:
                argv = _default_refresh_argv(fills_csv=str(fills_csv), settlements_csv=str(settlements_csv))
                if argv is None:
                    ok, out = False, "Default refresh script not found at src/archive/kalshi_activity_report.py"
                else:
                    ok, out = _run_refresh_argv(argv)
        st.sidebar.success("Refresh complete" if ok else "Refresh failed")
        if out:
            st.sidebar.code(out[:4000])
        if ok:
            st.session_state["_export_per_market_after_refresh"] = True

    _autorefresh(int(auto_refresh))

    # Load CSVs every render (no caching)
    loaded = load_csvs(fills_csv=str(fills_csv), settlements_csv=str(settlements_csv))

    # Reconstruct round trips (used for trade-level tables/plots)
    roundtrips, open_positions, diag = reconstruct_roundtrips(fills=loaded.fills, settlements=loaded.settlements)

    # Apply global filters on closed trades (by exit_time)
    import pandas as pd  # type: ignore

    filtered = roundtrips.copy()
    filtered_settlements = loaded.settlements.copy()
    filtered_fills = loaded.fills.copy()

    tz_name = "UTC" if tz_mode == "UTC" else "America/Denver"

    # Prepare settlements tz timestamps for filtering.
    if not filtered_settlements.empty and "settled_ts" in filtered_settlements.columns:
        filtered_settlements["settled_ts"] = pd.to_datetime(filtered_settlements["settled_ts"], utc=True, errors="coerce")
        filtered_settlements = filtered_settlements.dropna(subset=["settled_ts"]).copy()
        filtered_settlements["_settled_time_tz"] = filtered_settlements["settled_ts"].dt.tz_convert(tz_name)

    # Apply date-range filter (timezone-aware) to settlements.
    start_ts = end_ts = None
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
        if start_d and end_d:
            start_ts = pd.Timestamp(start_d).tz_localize(tz_name)
            end_ts = pd.Timestamp(end_d).tz_localize(tz_name) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            if not filtered_settlements.empty and "_settled_time_tz" in filtered_settlements.columns:
                filtered_settlements = filtered_settlements[
                    (filtered_settlements["_settled_time_tz"] >= start_ts)
                    & (filtered_settlements["_settled_time_tz"] <= end_ts)
                ]

    # Apply ticker filter to settlements.
    if ticker_filter.strip() and not filtered_settlements.empty and "ticker" in filtered_settlements.columns:
        filtered_settlements = filtered_settlements[
            filtered_settlements["ticker"].astype(str).str.contains(ticker_filter.strip(), case=False, na=False)
        ]

    # Align fills to the filtered settlement markets (one row per market).
    if not filtered_settlements.empty and "market_ticker" in filtered_settlements.columns and not filtered_fills.empty:
        keep = set(filtered_settlements["market_ticker"].astype(str).tolist())
        if "market_ticker" in filtered_fills.columns:
            filtered_fills = filtered_fills[filtered_fills["market_ticker"].astype(str).isin(keep)]

    # Roundtrip-level filters (optional, used for trade-level plots/tables).
    if not filtered.empty:
        filtered["exit_time"] = pd.to_datetime(filtered["exit_time"], utc=True, errors="coerce")
        filtered["_exit_time_tz"] = filtered["exit_time"].dt.tz_convert(tz_name)

        if start_ts is not None and end_ts is not None:
            filtered = filtered[(filtered["_exit_time_tz"] >= start_ts) & (filtered["_exit_time_tz"] <= end_ts)]

        if side_filter != "ALL":
            filtered = filtered[filtered["side"].astype(str).str.upper() == side_filter]

        if ticker_filter.strip():
            filtered = filtered[filtered["ticker"].astype(str).str.contains(ticker_filter.strip(), case=False, na=False)]

        if strategy_filter.strip() and "strategy_tag" in filtered.columns:
            filtered = filtered[
                filtered["strategy_tag"].astype(str).str.contains(strategy_filter.strip(), case=False, na=False)
            ]

    # Per-market metrics/KPIs (paired-payout settlement accounting + FIFO early-sell saved/lost)
    # IMPORTANT: `load_csvs()` normalizes settlements and drops the yes/no count/cost columns.
    # For paired-payout accounting we reload the raw Kalshi exports from disk.
    raw_settlements = load_settlements_export(str(settlements_csv))
    raw_fills = load_fills_export(str(fills_csv))

    # Keep alignment with the dashboard's settlement-based filters by restricting to the
    # tickers present in `filtered_settlements`.
    keep_tickers = set()
    if not filtered_settlements.empty:
        if "market_ticker" in filtered_settlements.columns:
            keep_tickers = set(filtered_settlements["market_ticker"].astype(str).tolist())
        elif "ticker" in filtered_settlements.columns:
            keep_tickers = set(filtered_settlements["ticker"].astype(str).tolist())

    if keep_tickers:
        raw_settlements = raw_settlements[raw_settlements["ticker"].astype(str).isin(keep_tickers)].copy()
        raw_fills = raw_fills[raw_fills["ticker"].astype(str).isin(keep_tickers)].copy()

    per_market = build_per_market_metrics(settlements=raw_settlements, fills=raw_fills)

    # Reattach settled_ts (for equity curve + day aggregation) from the normalized settlements.
    if not filtered_settlements.empty and "settled_ts" in filtered_settlements.columns:
        settle_map = filtered_settlements[["market_ticker", "settled_ts"]].copy()
        settle_map = settle_map.rename(columns={"market_ticker": "ticker"})
        per_market = per_market.merge(settle_map, on="ticker", how="left")

    summary = build_summary_kpis(per_market)

    # If the user just refreshed successfully, export per-market metrics for offline inspection.
    if st.session_state.pop("_export_per_market_after_refresh", False):
        try:
            _atomic_write_csv(per_market, per_market_export_path)
            st.sidebar.info(f"Exported per-market metrics: {per_market_export_path}")
        except Exception as e:
            st.sidebar.warning(f"Failed to export per-market metrics: {e}")

    # Total fees from settlements (source-of-truth)
    fees_total_usd = 0.0
    if not filtered_settlements.empty and "fee_cost_usd" in filtered_settlements.columns:
        fees_total_usd = float(pd.to_numeric(filtered_settlements["fee_cost_usd"], errors="coerce").fillna(0.0).sum())

    # Realized equity curve and drawdown from per-market realized PnL.
    eq_settle = pd.DataFrame(columns=["exit_time", "net_pnl_usd", "equity", "peak", "drawdown"])
    max_dd_usd = 0.0
    if not per_market.empty and "settled_ts" in per_market.columns:
        s = per_market.copy()
        s["settled_ts"] = pd.to_datetime(s["settled_ts"], utc=True, errors="coerce")
        s = s.dropna(subset=["settled_ts"]).sort_values("settled_ts")
        pnl = pd.to_numeric(s.get("realized_pnl_usd"), errors="coerce").fillna(0.0).astype(float)
        eq_settle = pd.DataFrame({"exit_time": s["settled_ts"], "net_pnl_usd": pnl})
        eq_settle["equity"] = eq_settle["net_pnl_usd"].cumsum()
        eq_settle["peak"] = eq_settle["equity"].cummax()
        eq_settle["drawdown"] = eq_settle["equity"] - eq_settle["peak"]
        dd_min = float(eq_settle["drawdown"].min()) if not eq_settle.empty else 0.0
        max_dd_usd = abs(dd_min)

    # ---- Summary tiles ----
    realized_total = float(summary.get("total_realized_pnl") or 0.0)
    realized_net_total = float(realized_total - fees_total_usd)

    a1, a2, a3, a4, a5, a6, a7 = st.columns(7)
    a1.metric("Realized Net P&L", f"${realized_net_total:,.2f}")
    a2.metric("Realized P&L", f"${realized_total:,.2f}")
    a3.metric("Total fees", f"${fees_total_usd:,.2f}")
    a4.metric(
        "Win rate",
        "-" if summary.get("win_rate") is None else f"{float(summary['win_rate'])*100.0:.1f}%",
    )
    a5.metric(
        "Profit factor",
        "-" if summary.get("profit_factor") is None else f"{float(summary['profit_factor']):.2f}",
    )
    a6.metric("Max drawdown", f"${max_dd_usd:,.2f}")
    a7.metric(
        "Median P&L",
        "-" if summary.get("median_pnl") is None else f"${float(summary['median_pnl']):,.2f}",
    )

    b1, b2, b3 = st.columns(3)
    b1.metric("Early sell contracts", f"{int(summary.get('total_early_sell_contracts') or 0)}")
    b2.metric("Early sell saved", f"${float(summary.get('total_early_sell_saved_usd') or 0.0):,.2f}")
    b3.metric("Early sell lost", f"${float(summary.get('total_early_sell_lost_usd') or 0.0):,.2f}")

    st.divider()

    # ---- Charts ----
    left, right = st.columns(2)

    with left:
        st.subheader("Equity curve (realized)")
        if eq_settle.empty:
            st.info("No closed trades in the selected filters.")
        else:
            import altair as alt  # type: ignore

            plot_df = eq_settle.copy()
            plot_df["equity_pos"] = plot_df["equity"].where(plot_df["equity"] >= 0)
            plot_df["equity_neg"] = plot_df["equity"].where(plot_df["equity"] < 0)

            base = alt.Chart(plot_df).encode(x=alt.X("exit_time:T", title=None))
            pos = base.mark_line(color="green").encode(y=alt.Y("equity_pos:Q", title=None))
            neg = base.mark_line(color="red").encode(y=alt.Y("equity_neg:Q", title=None))

            st.altair_chart((pos + neg).properties(height=280), use_container_width=True)

    with right:
        st.subheader("P&L by day")
        if per_market.empty or "settled_ts" not in per_market.columns:
            st.info("No settled markets in the selected filters.")
        else:
            tmp = per_market.copy()
            tmp["settled_ts"] = pd.to_datetime(tmp["settled_ts"], utc=True, errors="coerce")
            tmp = tmp.dropna(subset=["settled_ts"]).copy()
            tz_name = "UTC" if tz_mode == "UTC" else "America/Denver"
            tmp["day"] = tmp["settled_ts"].dt.tz_convert(tz_name).dt.floor("D")
            tmp["realized_pnl_usd"] = pd.to_numeric(tmp.get("realized_pnl_usd"), errors="coerce").fillna(0.0).astype(float)
            by_day = tmp.groupby("day", as_index=True)["realized_pnl_usd"].sum().sort_index()
            st.bar_chart(by_day)

    st.subheader("Rolling win rate (last 20 trades)")
    if filtered.empty:
        st.info("No closed trades.")
    else:
        tmp = filtered.sort_values("exit_time").copy()
        tmp["win"] = (tmp["net_pnl_usd"].astype(float) > 0).astype(int)
        tmp["roll_win_rate"] = tmp["win"].rolling(20, min_periods=5).mean()
        st.line_chart(tmp.set_index("exit_time")["roll_win_rate"])

    st.divider()

    st.subheader("Per-market metrics")
    if per_market.empty:
        st.info("No settled markets in the selected filters.")
    else:
        cols = [
            c
            for c in [
                "ticker",
                "market_result",
                "realized_pnl_usd",
                "is_win",
                "gross_payout_usd",
                "gross_cost_usd",
                "fee_cost",
                "yes_count",
                "no_count",
                "paired_count",
                "net_count",
                "early_sell_contracts",
                "early_sell_saved_usd",
                "early_sell_lost_usd",
            ]
            if c in per_market.columns
        ]
        st.dataframe(
            per_market[cols],
            use_container_width=True,
        )

    with st.expander("Diagnostics"):
        st.json({"load": loaded.diagnostics, "reconstruction": asdict(diag)})


if __name__ == "__main__":
    main()
