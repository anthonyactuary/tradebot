"""
Poll-and-Enter-Once Backtest with Optimization Analysis

More realistic simulation: for each ticker, scan from high TTE to low TTE 
(simulating polling every 5 seconds), enter ONCE when edge first appears, 
then hold until expiry.

Includes parameter sweeps for:
- Edge thresholds (5%, 7%, 10% for high TTE)
- Start TTE (when to begin polling)
- YES-only vs both sides
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import math
from scipy import stats


SLIPPAGE = 0.003


def kalshi_taker_fee(price: float, contracts: int = 1) -> float:
    """
    Calculate Kalshi taker fee.
    Formula: ceil(0.07 * C * P * (1-P) * 100) / 100
    
    Args:
        price: Contract price in [0, 1]
        contracts: Number of contracts
    
    Returns:
        Fee in dollars (e.g., 0.02 for 2 cents)
    """
    if price <= 0 or price >= 1:
        return 0.0
    raw_fee = 0.07 * contracts * price * (1 - price)
    # Round up to the cent
    return math.ceil(raw_fee * 100) / 100


def kalshi_maker_fee(price: float, contracts: int = 1) -> float:
    """
    Calculate Kalshi maker fee.
    Formula: ceil(0.0175 * C * P * (1-P) * 100) / 100
    """
    if price <= 0 or price >= 1:
        return 0.0
    raw_fee = 0.0175 * contracts * price * (1 - price)
    return math.ceil(raw_fee * 100) / 100


@dataclass
class BacktestConfig:
    """Configuration for a backtest run"""
    high_tte_edge: float = 0.05  # Edge threshold for TTE >= 600
    mid_tte_edge: float = 0.025  # Edge threshold for TTE >= 360
    low_tte_edge: float = 0.01   # Edge threshold for TTE < 360
    start_tte: int = 720         # Start polling at this TTE (720 = earliest)
    yes_only: bool = False       # Only take YES positions
    
    def get_min_edge(self, tte: int) -> float:
        """Get minimum edge threshold for a given TTE"""
        if tte >= 600:
            return self.high_tte_edge
        elif tte >= 360:
            return self.mid_tte_edge
        else:
            return self.low_tte_edge
    
    def __str__(self):
        side = "YES-only" if self.yes_only else "Both"
        return f"TTE≤{self.start_tte}, edge={self.high_tte_edge*100:.0f}%/{self.mid_tte_edge*100:.1f}%/{self.low_tte_edge*100:.0f}%, {side}"


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    config: BacktestConfig
    trades_df: pd.DataFrame
    n_trades: int
    n_skipped: int
    total_pnl: float
    win_rate: float
    avg_pnl: float
    avg_ev: float
    yes_pnl: float
    no_pnl: float
    yes_trades: int
    no_trades: int


def compute_pvalues(trades_df: pd.DataFrame) -> dict:
    """
    Compute statistical significance p-values for backtest results.
    
    Returns:
        Dictionary with p-values and related statistics
    """
    if len(trades_df) == 0:
        return {
            "n_trades": 0,
            "n_wins": 0,
            "win_rate": 0,
            "binom_pvalue": 1.0,
            "win_rate_ci_low": 0,
            "win_rate_ci_high": 0,
            "total_pnl": 0,
            "mean_pnl": 0,
            "std_pnl": 0,
            "t_stat": 0,
            "t_pvalue": 1.0,
            "trades_for_p05": 0,
            "significant_wr": False,
            "significant_pnl": False,
        }
    
    n_trades = len(trades_df)
    n_wins = int(trades_df["win"].sum())
    win_rate = n_wins / n_trades
    
    # Binomial test: Is win rate > 50%?
    binom_result = stats.binomtest(n_wins, n_trades, p=0.5, alternative='greater')
    ci = binom_result.proportion_ci(confidence_level=0.95)
    
    # T-test: Is mean PnL > 0?
    total_pnl = trades_df["pnl"].sum()
    mean_pnl = trades_df["pnl"].mean()
    std_pnl = trades_df["pnl"].std()
    
    if std_pnl > 0:
        se_pnl = std_pnl / np.sqrt(n_trades)
        t_stat = mean_pnl / se_pnl
        t_pvalue = 1 - stats.t.cdf(t_stat, df=n_trades-1)
    else:
        t_stat = 0
        t_pvalue = 1.0 if mean_pnl <= 0 else 0.0
    
    # Power analysis: trades needed for p<0.05 at current win rate
    trades_for_p05 = 0
    if win_rate > 0.5:
        for n in range(50, 2000, 10):
            wins_at_rate = int(n * win_rate)
            result = stats.binomtest(wins_at_rate, n, p=0.5, alternative='greater')
            if result.pvalue < 0.05:
                trades_for_p05 = n
                break
    
    return {
        "n_trades": n_trades,
        "n_wins": n_wins,
        "win_rate": win_rate,
        "binom_pvalue": binom_result.pvalue,
        "win_rate_ci_low": ci.low,
        "win_rate_ci_high": ci.high,
        "total_pnl": total_pnl,
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "t_stat": t_stat,
        "t_pvalue": t_pvalue,
        "trades_for_p05": trades_for_p05,
        "significant_wr": binom_result.pvalue < 0.05,
        "significant_pnl": t_pvalue < 0.05,
    }


def run_backtest(df: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    """Run a single backtest with the given configuration"""
    trades = []
    n_tickers = df["ticker"].nunique()
    
    for ticker, group in df.groupby("ticker"):
        # Filter to start_tte
        group = group[group["tte_bucket"] <= config.start_tte]
        group = group.sort_values("tte_bucket", ascending=False)  # High to low
        
        if len(group) == 0:
            continue
            
        y_true = group["y_true"].iloc[0]
        
        entered = False
        for _, row in group.iterrows():
            if entered:
                break
                
            tte = row["tte_bucket"]
            p_cal = row["p_cal"]
            market_p_yes = row["market_p_yes"]
            market_p_no = row["market_p_no"]
            
            if pd.isna(market_p_yes) or pd.isna(market_p_no):
                continue
            
            ev_yes = p_cal - market_p_yes - SLIPPAGE
            ev_no = (1 - p_cal) - market_p_no - SLIPPAGE
            min_edge = config.get_min_edge(tte)
            
            # Check YES
            if ev_yes >= min_edge:
                entry_price = market_p_yes + SLIPPAGE
                fee = kalshi_taker_fee(entry_price)
                # P&L: win pays (1 - entry), lose pays (-entry), minus fee
                gross_pnl = (1 - entry_price) if y_true == 1 else -entry_price
                pnl = gross_pnl - fee
                trades.append({
                    "ticker": ticker,
                    "tte_bucket": tte,
                    "side": "YES",
                    "p_cal": p_cal,
                    "entry_price": entry_price,
                    "fee": fee,
                    "ev": ev_yes,
                    "min_edge": min_edge,
                    "win": int(y_true == 1),
                    "gross_pnl": gross_pnl,
                    "pnl": pnl,
                    "y_true": y_true,
                })
                entered = True
            # Check NO (unless yes_only)
            elif not config.yes_only and ev_no >= min_edge:
                entry_price = market_p_no + SLIPPAGE
                fee = kalshi_taker_fee(entry_price)
                gross_pnl = (1 - entry_price) if y_true == 0 else -entry_price
                pnl = gross_pnl - fee
                trades.append({
                    "ticker": ticker,
                    "tte_bucket": tte,
                    "side": "NO",
                    "p_cal": p_cal,
                    "entry_price": entry_price,
                    "fee": fee,
                    "ev": ev_no,
                    "min_edge": min_edge,
                    "win": int(y_true == 0),
                    "gross_pnl": gross_pnl,
                    "pnl": pnl,
                    "y_true": y_true,
                })
                entered = True
    
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        return BacktestResult(
            config=config, trades_df=trades_df, n_trades=0, n_skipped=n_tickers,
            total_pnl=0, win_rate=0, avg_pnl=0, avg_ev=0,
            yes_pnl=0, no_pnl=0, yes_trades=0, no_trades=0
        )
    
    yes_df = trades_df[trades_df["side"] == "YES"]
    no_df = trades_df[trades_df["side"] == "NO"]
    
    return BacktestResult(
        config=config,
        trades_df=trades_df,
        n_trades=len(trades_df),
        n_skipped=n_tickers - len(trades_df),
        total_pnl=trades_df["pnl"].sum(),
        win_rate=trades_df["win"].mean() * 100,
        avg_pnl=trades_df["pnl"].mean(),
        avg_ev=trades_df["ev"].mean(),
        yes_pnl=yes_df["pnl"].sum() if len(yes_df) > 0 else 0,
        no_pnl=no_df["pnl"].sum() if len(no_df) > 0 else 0,
        yes_trades=len(yes_df),
        no_trades=len(no_df),
    )


def print_detailed_results(result: BacktestResult):
    """Print detailed breakdown of a backtest result"""
    trades_df = result.trades_df
    
    if len(trades_df) == 0:
        print("No trades executed")
        return
    
    print(f"\nOverall ({result.n_trades} trades):")
    print(f"  Total P&L:     {result.total_pnl:+.2f} units")
    print(f"  Win rate:      {result.win_rate:.1f}%")
    print(f"  Avg P&L/trade: {result.avg_pnl:+.4f}")
    print(f"  Avg EV:        {result.avg_ev:+.4f}")
    
    # Statistical significance
    pvals = compute_pvalues(trades_df)
    print("\n" + "-" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("-" * 70)
    sig_wr = "YES" if pvals["significant_wr"] else "NO"
    sig_pnl = "YES" if pvals["significant_pnl"] else "NO"
    print(f"  Win Rate Test (H0: WR=50%):  p = {pvals['binom_pvalue']:.4f} [Significant: {sig_wr}]")
    print(f"  Win Rate 95% CI:             [{pvals['win_rate_ci_low']*100:.1f}%, {pvals['win_rate_ci_high']*100:.1f}%]")
    print(f"  Profit Test (H0: PnL=0):     p = {pvals['t_pvalue']:.4f} [Significant: {sig_pnl}]")
    if pvals["trades_for_p05"] > 0:
        print(f"  Trades needed for p<0.05:    ~{pvals['trades_for_p05']} (at {pvals['win_rate']*100:.1f}% WR)")
    
    # By side
    print("\n" + "-" * 70)
    print("BY SIDE")
    print("-" * 70)
    for side in ["YES", "NO"]:
        side_df = trades_df[trades_df["side"] == side]
        if len(side_df) > 0:
            print(f"{side}: {len(side_df)} trades, {side_df['win'].mean()*100:.1f}% win, "
                  f"P&L: {side_df['pnl'].sum():+.2f}, Avg: {side_df['pnl'].mean():+.4f}")
    
    # By TTE - YES vs NO
    print("\n" + "-" * 70)
    print("ENTRY TTE - YES vs NO")
    print("-" * 70)
    print(f"{'TTE':>5} | {'YES':^22} | {'NO':^22} | Net PnL")
    print("-" * 70)
    for tte in sorted(trades_df["tte_bucket"].unique(), reverse=True):
        tte_df = trades_df[trades_df["tte_bucket"] == tte]
        yes_df = tte_df[tte_df["side"] == "YES"]
        no_df = tte_df[tte_df["side"] == "NO"]
        
        yes_wr = yes_df["win"].mean() * 100 if len(yes_df) > 0 else 0
        no_wr = no_df["win"].mean() * 100 if len(no_df) > 0 else 0
        yes_str = f"{len(yes_df):>3} @ {yes_wr:>2.0f}% = {yes_df['pnl'].sum():+.2f}"
        no_str = f"{len(no_df):>3} @ {no_wr:>2.0f}% = {no_df['pnl'].sum():+.2f}"
        net = tte_df["pnl"].sum()
        print(f"{tte:>5} | {yes_str:>22} | {no_str:>22} | {net:+.2f}")
    
    print("-" * 70)
    print(f"TOTAL | YES: {result.yes_pnl:+.2f} ({result.yes_trades}) | "
          f"NO: {result.no_pnl:+.2f} ({result.no_trades}) | Net: {result.total_pnl:+.2f}")


def run_optimization_sweep(df: pd.DataFrame) -> List[BacktestResult]:
    """Run parameter sweep across different configurations"""
    results = []
    
    # Parameter grids
    high_tte_edges = [0.05, 0.07, 0.10]
    start_ttes = [720, 660, 600, 540, 480, 360]
    yes_only_options = [False, True]
    
    for high_edge in high_tte_edges:
        for start_tte in start_ttes:
            for yes_only in yes_only_options:
                config = BacktestConfig(
                    high_tte_edge=high_edge,
                    start_tte=start_tte,
                    yes_only=yes_only,
                )
                result = run_backtest(df, config)
                results.append(result)
    
    return results


def print_optimization_summary(results: List[BacktestResult]):
    """Print summary table of all optimization results"""
    print("\n" + "=" * 100)
    print("OPTIMIZATION SWEEP RESULTS")
    print("=" * 100)
    
    # Sort by total P&L descending
    results_sorted = sorted(results, key=lambda r: r.total_pnl, reverse=True)
    
    print(f"{'Config':<40} {'Trades':>6} {'Win%':>6} {'P&L':>8} {'WR p-val':>9} {'PnL p-val':>10}")
    print("-" * 100)
    
    for r in results_sorted:
        side_str = "YES" if r.config.yes_only else "Both"
        config_str = f"TTE<={r.config.start_tte}, {r.config.high_tte_edge*100:.0f}% edge, {side_str}"
        pvals = compute_pvalues(r.trades_df)
        wr_sig = "*" if pvals["significant_wr"] else ""
        pnl_sig = "*" if pvals["significant_pnl"] else ""
        print(f"{config_str:<40} {r.n_trades:>6} {r.win_rate:>5.1f}% {r.total_pnl:>+8.2f} "
              f"{pvals['binom_pvalue']:>8.4f}{wr_sig} {pvals['t_pvalue']:>9.4f}{pnl_sig}")
    
    print("\n  * = statistically significant at p<0.05")
    
    # Find best configs
    print("\n" + "-" * 100)
    print("TOP 5 CONFIGURATIONS:")
    print("-" * 100)
    for i, r in enumerate(results_sorted[:5], 1):
        pvals = compute_pvalues(r.trades_df)
        print(f"  {i}. {r.config}")
        print(f"     P&L: {r.total_pnl:+.2f}, {r.win_rate:.1f}% win, {r.n_trades} trades")
        print(f"     p-values: WR={pvals['binom_pvalue']:.4f}, PnL={pvals['t_pvalue']:.4f}")
    
    # Best YES-only
    yes_only_results = [r for r in results_sorted if r.config.yes_only]
    if yes_only_results:
        best_yes = yes_only_results[0]
        pvals = compute_pvalues(best_yes.trades_df)
        print(f"\nBest YES-only: {best_yes.config}")
        print(f"  P&L: {best_yes.total_pnl:+.2f}, Win: {best_yes.win_rate:.1f}%, Trades: {best_yes.n_trades}")
        print(f"  p-values: WR={pvals['binom_pvalue']:.4f}, PnL={pvals['t_pvalue']:.4f}")
    
    # Best Both sides
    both_results = [r for r in results_sorted if not r.config.yes_only]
    if both_results:
        best_both = both_results[0]
        pvals = compute_pvalues(best_both.trades_df)
        print(f"\nBest Both-sides: {best_both.config}")
        print(f"  P&L: {best_both.total_pnl:+.2f}, Win: {best_both.win_rate:.1f}%, Trades: {best_both.n_trades}")
        print(f"  p-values: WR={pvals['binom_pvalue']:.4f}, PnL={pvals['t_pvalue']:.4f}")


def print_edge_comparison(df: pd.DataFrame):
    """Compare different edge thresholds for high TTE"""
    print("\n" + "=" * 70)
    print("EDGE THRESHOLD COMPARISON (TTE >= 600)")
    print("=" * 70)
    
    for yes_only in [False, True]:
        mode = "YES-only" if yes_only else "Both sides"
        print(f"\n{mode}:")
        print(f"  {'Edge':>6} {'Trades':>8} {'Win%':>7} {'P&L':>10} {'Avg P&L':>10}")
        print("  " + "-" * 45)
        
        for edge in [0.03, 0.05, 0.07, 0.10, 0.12]:
            config = BacktestConfig(high_tte_edge=edge, yes_only=yes_only)
            result = run_backtest(df, config)
            print(f"  {edge*100:>5.0f}% {result.n_trades:>8} {result.win_rate:>6.1f}% "
                  f"{result.total_pnl:>+10.2f} {result.avg_pnl:>+10.4f}")


def print_start_tte_comparison(df: pd.DataFrame):
    """Compare different start TTEs"""
    print("\n" + "=" * 70)
    print("START TTE COMPARISON (when to begin polling)")
    print("=" * 70)
    
    for yes_only in [False, True]:
        mode = "YES-only" if yes_only else "Both sides"
        print(f"\n{mode} (7% edge for TTE>=600):")
        print(f"  {'Start TTE':>10} {'Trades':>8} {'Win%':>7} {'P&L':>10} {'Avg P&L':>10}")
        print("  " + "-" * 50)
        
        for start_tte in [720, 680, 640, 600, 540, 480, 420, 360, 300]:
            config = BacktestConfig(high_tte_edge=0.07, start_tte=start_tte, yes_only=yes_only)
            result = run_backtest(df, config)
            print(f"  {start_tte:>10} {result.n_trades:>8} {result.win_rate:>6.1f}% "
                  f"{result.total_pnl:>+10.2f} {result.avg_pnl:>+10.4f}")


def main():
    # Load data
    data_path = Path(__file__).parent / "calibration_backtest.csv"
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows")
    print(f"Unique tickers: {df['ticker'].nunique()}")
    print(f"Slippage: {SLIPPAGE*100:.1f}%")
    print(f"Fee model: Kalshi taker (0.07 * P * (1-P), ceil to cent)")
    print(f"  Example fees: P=0.50 -> ${kalshi_taker_fee(0.50):.2f}, P=0.20 -> ${kalshi_taker_fee(0.20):.2f}, P=0.80 -> ${kalshi_taker_fee(0.80):.2f}")
    
    # =========================================================================
    # 1. Run baseline (original config)
    # =========================================================================
    print("\n" + "=" * 70)
    print("BASELINE: Both sides, 5% edge, TTE <= 720")
    print("=" * 70)
    baseline_config = BacktestConfig(high_tte_edge=0.05, start_tte=720, yes_only=False)
    baseline_result = run_backtest(df, baseline_config)
    print_detailed_results(baseline_result)
    
    # =========================================================================
    # 2. Edge threshold comparison
    # =========================================================================
    print_edge_comparison(df)
    
    # =========================================================================
    # 3. Start TTE comparison
    # =========================================================================
    print_start_tte_comparison(df)
    
    # =========================================================================
    # 4. Full optimization sweep
    # =========================================================================
    results = run_optimization_sweep(df)
    print_optimization_summary(results)
    
    # =========================================================================
    # 5. Recommended config with detailed breakdown
    # =========================================================================
    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIG: YES-only, TTE <= 360, 7% edge")
    print("=" * 70)
    recommended_config = BacktestConfig(
        high_tte_edge=0.07,
        start_tte=360,
        yes_only=True,
    )
    recommended_result = run_backtest(df, recommended_config)
    print_detailed_results(recommended_result)
    
    # Save best trades
    if len(recommended_result.trades_df) > 0:
        output_path = Path(__file__).parent / "best_config_trades.csv"
        recommended_result.trades_df.to_csv(output_path, index=False)
        print(f"\nTrades saved to: {output_path}")


if __name__ == "__main__":
    main()
