"""
Time-of-Day Analysis

Analyze trading performance by hour of day (ET timezone).
Parse hour from ticker like: KXBTC15M-26JAN172045-45 -> 20:45 on Jan 17
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re


def parse_hour_from_ticker(ticker: str) -> int:
    """
    Extract hour from ticker like KXBTC15M-26JAN172045-45
    Format: KXBTC15M-26JAN17HHMM-MM
    The HHMM after the date is the market close time in ET
    """
    # Match pattern: date followed by 4 digits (HHMM)
    match = re.search(r'-26JAN(\d{2})(\d{2})(\d{2})-', ticker)
    if match:
        day = int(match.group(1))
        hour = int(match.group(2))
        minute = int(match.group(3))
        return hour, day
    return None, None


def main():
    # Load trades
    trades_path = Path(__file__).parent / "best_config_trades.csv"
    df = pd.read_csv(trades_path)
    
    print("=" * 80)
    print("TIME-OF-DAY ANALYSIS")
    print("=" * 80)
    
    # Parse hour from ticker
    df["hour"], df["day"] = zip(*df["ticker"].apply(parse_hour_from_ticker))
    df = df.dropna(subset=["hour"])
    df["hour"] = df["hour"].astype(int)
    df["day"] = df["day"].astype(int)
    
    print(f"\nTotal trades: {len(df)}")
    print(f"Date range: Jan {df['day'].min()}-{df['day'].max()}, 2026")
    print(f"Hours covered: {sorted(df['hour'].unique())}")
    
    # Group by hour
    print("\n" + "=" * 80)
    print("PERFORMANCE BY HOUR (ET)")
    print("=" * 80)
    
    hourly = df.groupby("hour").agg({
        "pnl": ["count", "sum", "mean"],
        "win": "mean",
        "entry_price": "mean",
    }).round(4)
    
    hourly.columns = ["trades", "total_pnl", "avg_pnl", "win_rate", "avg_entry"]
    hourly["win_rate"] = (hourly["win_rate"] * 100).round(1)
    hourly = hourly.sort_index()
    
    print(f"\n{'Hour':>6} {'Trades':>8} {'Win%':>8} {'Total P&L':>12} {'Avg P&L':>10} {'Avg Entry':>10}")
    print("-" * 60)
    
    for hour, row in hourly.iterrows():
        # Classify time period
        if 6 <= hour < 12:
            period = "Morning"
        elif 12 <= hour < 18:
            period = "Afternoon"
        elif 18 <= hour < 24:
            period = "Evening"
        else:
            period = "Night"
        
        marker = " *" if row["total_pnl"] < 0 else ""
        print(f"{hour:>4}:00 {row['trades']:>8.0f} {row['win_rate']:>7.1f}% "
              f"${row['total_pnl']:>+10.2f} ${row['avg_pnl']:>+9.4f} ${row['avg_entry']:>9.3f}{marker}")
    
    # By time period
    print("\n" + "=" * 80)
    print("PERFORMANCE BY TIME PERIOD (ET)")
    print("=" * 80)
    
    def get_period(hour):
        if 6 <= hour < 12:
            return "Morning (6-12 ET)"
        elif 12 <= hour < 18:
            return "Afternoon (12-18 ET)"
        elif 18 <= hour < 24:
            return "Evening (18-24 ET)"
        else:
            return "Night (0-6 ET)"
    
    df["period"] = df["hour"].apply(get_period)
    
    period_stats = df.groupby("period").agg({
        "pnl": ["count", "sum", "mean"],
        "win": "mean",
    }).round(4)
    
    period_stats.columns = ["trades", "total_pnl", "avg_pnl", "win_rate"]
    period_stats["win_rate"] = (period_stats["win_rate"] * 100).round(1)
    
    # Sort by time of day
    period_order = ["Night (0-6 ET)", "Morning (6-12 ET)", "Afternoon (12-18 ET)", "Evening (18-24 ET)"]
    period_stats = period_stats.reindex([p for p in period_order if p in period_stats.index])
    
    print(f"\n{'Period':<25} {'Trades':>8} {'Win%':>8} {'Total P&L':>12} {'Avg P&L':>10}")
    print("-" * 70)
    
    for period, row in period_stats.iterrows():
        marker = " **WORST**" if row["total_pnl"] == period_stats["total_pnl"].min() else ""
        marker = " **BEST**" if row["total_pnl"] == period_stats["total_pnl"].max() else marker
        print(f"{period:<25} {row['trades']:>8.0f} {row['win_rate']:>7.1f}% "
              f"${row['total_pnl']:>+10.2f} ${row['avg_pnl']:>+9.4f}{marker}")
    
    # By day
    print("\n" + "=" * 80)
    print("PERFORMANCE BY DAY")
    print("=" * 80)
    
    daily = df.groupby("day").agg({
        "pnl": ["count", "sum", "mean"],
        "win": "mean",
    }).round(4)
    
    daily.columns = ["trades", "total_pnl", "avg_pnl", "win_rate"]
    daily["win_rate"] = (daily["win_rate"] * 100).round(1)
    
    print(f"\n{'Day':>8} {'Trades':>8} {'Win%':>8} {'Total P&L':>12} {'Avg P&L':>10}")
    print("-" * 50)
    
    for day, row in daily.iterrows():
        print(f"Jan {day:>2} {row['trades']:>8.0f} {row['win_rate']:>7.1f}% "
              f"${row['total_pnl']:>+10.2f} ${row['avg_pnl']:>+9.4f}")
    
    # Worst hours detail
    print("\n" + "=" * 80)
    print("WORST PERFORMING HOURS (sorted by total P&L)")
    print("=" * 80)
    
    hourly_sorted = hourly.sort_values("total_pnl")
    
    print(f"\n{'Hour':>6} {'Trades':>8} {'Win%':>8} {'P&L':>12} {'Comment':<30}")
    print("-" * 70)
    
    for hour, row in hourly_sorted.head(8).iterrows():
        if 6 <= hour < 9:
            comment = "Early morning - markets opening"
        elif 9 <= hour < 16:
            comment = "US market hours"
        elif 16 <= hour < 20:
            comment = "After hours / early evening"
        elif 20 <= hour < 24:
            comment = "Late evening"
        else:
            comment = "Overnight"
        
        print(f"{hour:>4}:00 {row['trades']:>8.0f} {row['win_rate']:>7.1f}% "
              f"${row['total_pnl']:>+10.2f}  {comment}")
    
    # Statistical significance caveat
    print("\n" + "=" * 80)
    print("DATA QUALITY NOTE")
    print("=" * 80)
    
    min_trades_per_hour = hourly["trades"].min()
    max_trades_per_hour = hourly["trades"].max()
    
    print(f"""
Trades per hour: {min_trades_per_hour:.0f} - {max_trades_per_hour:.0f}

With only {len(df)} total trades over {df['day'].nunique()} days:
- Each hour has ~{len(df) / df['hour'].nunique():.0f} trades on average
- This is NOT enough for statistical significance
- Patterns you see could easily be random noise

Recommendation: Collect 2-4 weeks of data before drawing conclusions
about time-of-day performance.
""")


if __name__ == "__main__":
    main()
