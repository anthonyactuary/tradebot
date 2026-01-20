"""
Financial Model: Compound Growth Projection

Projects account growth using calibration backtest results.
Configurable parameters:
- Starting balance
- Position sizing (% of account per trade)
- Time horizon
- Liquidity cap
- Win rate degradation factor
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import math


# =============================================================================
# CONFIGURATION - ADJUST THESE
# =============================================================================
@dataclass
class ModelConfig:
    """Configuration for financial model"""
    starting_balance: float = 100.0      # Initial account balance ($)
    position_pct: float = 0.10          # Bet 10% of account per trade
    liquidity_cap: float = 150.0        # Max $ per trade (market liquidity)
    win_rate_degradation: float = 0.95  # Assume 5% worse than backtest in real world
    backtest_days: int = 3              # Days of data in backtest (Jan 17-19, 2026)
    trading_time_factor: float = 0.5    # 0.5 = trade 12 hours/day, 1.0 = 24hr


@dataclass 
class TradeStats:
    """Statistics from backtest trades"""
    n_trades: int
    win_rate: float
    avg_entry_price: float
    avg_fee: float
    trades_per_day: float
    avg_win_pnl: float
    avg_loss_pnl: float


def load_trade_stats(trades_path: Path, config: ModelConfig) -> Tuple[TradeStats, pd.DataFrame]:
    """Load trades and compute statistics"""
    df = pd.read_csv(trades_path)
    
    wins = df[df["win"] == 1]
    losses = df[df["win"] == 0]
    
    avg_win_pnl = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss_pnl = abs(losses["pnl"].mean()) if len(losses) > 0 else 0
    
    raw_trades_per_day = len(df) / config.backtest_days
    
    stats = TradeStats(
        n_trades=len(df),
        win_rate=df["win"].mean(),
        avg_entry_price=df["entry_price"].mean(),
        avg_fee=df["fee"].mean() if "fee" in df.columns else 0.02,
        trades_per_day=raw_trades_per_day * config.trading_time_factor,
        avg_win_pnl=avg_win_pnl,
        avg_loss_pnl=avg_loss_pnl,
    )
    
    return stats, df


def run_compounding_projection(
    stats: TradeStats, 
    config: ModelConfig,
    months: int,
    win_rate_override: float = None,
) -> dict:
    """
    Run deterministic compounding projection using expected value.
    
    Args:
        stats: Trade statistics from backtest
        config: Model configuration
        months: Number of months to project
        win_rate_override: Override win rate (for sensitivity analysis)
    
    Returns:
        Dictionary with projection results
    """
    win_rate = win_rate_override if win_rate_override else stats.win_rate * config.win_rate_degradation
    
    days = months * 30
    n_trades = int(stats.trades_per_day * days)
    
    balance = config.starting_balance
    balances = [balance]
    capped_count = 0
    
    for _ in range(n_trades):
        # Position size: X% of balance, capped at liquidity limit
        position = min(balance * config.position_pct, config.liquidity_cap)
        if position >= config.liquidity_cap:
            capped_count += 1
        
        n_contracts = position / stats.avg_entry_price
        fee = stats.avg_fee * n_contracts
        
        # Expected profit per trade
        exp_win = (1 - stats.avg_entry_price) * n_contracts - fee
        exp_loss = -stats.avg_entry_price * n_contracts - fee
        exp_profit = win_rate * exp_win + (1 - win_rate) * exp_loss
        
        balance += exp_profit
        balance = max(balance, 0.01)  # Floor at 1 cent
        balances.append(balance)
    
    return {
        "months": months,
        "days": days,
        "n_trades": n_trades,
        "final_balance": balance,
        "roi": (balance - config.starting_balance) / config.starting_balance * 100,
        "capped_pct": capped_count / n_trades * 100 if n_trades > 0 else 0,
        "daily_profit": (balance - config.starting_balance) / days if days > 0 else 0,
        "balances": balances,
        "win_rate_used": win_rate,
    }


def run_linear_projection(stats: TradeStats, config: ModelConfig, months: int) -> dict:
    """
    Run linear (no compounding) projection - flat bet sizing.
    Good for sanity checking the compounding results.
    """
    days = months * 30
    n_trades = int(stats.trades_per_day * days)
    
    # Flat betting: always bet X% of INITIAL balance
    bet_size = config.starting_balance * config.position_pct
    n_contracts = bet_size / stats.avg_entry_price
    
    # Expected profit per trade
    exp_profit_per_trade = stats.win_rate * stats.avg_win_pnl - (1 - stats.win_rate) * stats.avg_loss_pnl
    profit_per_trade = exp_profit_per_trade * (n_contracts / (bet_size / stats.avg_entry_price))
    
    total_profit = profit_per_trade * n_trades
    final_balance = config.starting_balance + total_profit
    
    return {
        "months": months,
        "n_trades": n_trades,
        "bet_size": bet_size,
        "profit_per_trade": profit_per_trade,
        "total_profit": total_profit,
        "final_balance": final_balance,
        "roi": total_profit / config.starting_balance * 100,
    }


def calculate_kelly(stats: TradeStats) -> dict:
    """Calculate Kelly Criterion for optimal position sizing"""
    p = stats.win_rate
    q = 1 - p
    
    b = stats.avg_win_pnl / stats.avg_loss_pnl if stats.avg_loss_pnl > 0 else 0
    
    kelly = (p * b - q) / b if b > 0 else 0
    kelly = max(0, kelly)
    
    return {
        "win_prob": p,
        "avg_win": stats.avg_win_pnl,
        "avg_loss": stats.avg_loss_pnl,
        "win_loss_ratio": b,
        "full_kelly": kelly,
        "half_kelly": kelly * 0.5,
        "quarter_kelly": kelly * 0.25,
    }


def main():
    # ==========================================================================
    # Configuration
    # ==========================================================================
    config = ModelConfig(
        starting_balance=100.0,
        position_pct=0.10,
        liquidity_cap=150.0,
        win_rate_degradation=0.95,
        backtest_days=3,
        trading_time_factor=0.5,
    )
    
    # ==========================================================================
    # Load trade data
    # ==========================================================================
    trades_path = Path(__file__).parent / "best_config_trades.csv"
    
    if not trades_path.exists():
        print(f"Error: {trades_path} not found. Run poll_once_backtest.py first.")
        return
    
    stats, trades_df = load_trade_stats(trades_path, config)
    
    # ==========================================================================
    # Print header and statistics
    # ==========================================================================
    print("=" * 70)
    print("FINANCIAL MODEL: COMPOUND GROWTH PROJECTION")
    print("=" * 70)
    
    print(f"\n--- BACKTEST STATISTICS ({stats.n_trades} trades over {config.backtest_days} days) ---")
    print(f"  Win rate: {stats.win_rate*100:.1f}%")
    print(f"  Avg entry price: ${stats.avg_entry_price:.3f}")
    print(f"  Avg fee per trade: ${stats.avg_fee:.3f}")
    print(f"  Avg win P&L: ${stats.avg_win_pnl:.4f}")
    print(f"  Avg loss P&L: ${stats.avg_loss_pnl:.4f}")
    print(f"  Trades/day (24hr): {stats.n_trades / config.backtest_days:.1f}")
    print(f"  Trades/day (adjusted): {stats.trades_per_day:.1f}")
    
    print(f"\n--- MODEL CONFIGURATION ---")
    print(f"  Starting balance: ${config.starting_balance:.2f}")
    print(f"  Position size: {config.position_pct*100:.0f}% of account")
    print(f"  Liquidity cap: ${config.liquidity_cap:.0f}/trade")
    print(f"  Win rate degradation: {config.win_rate_degradation*100:.0f}% of backtest")
    print(f"  Adjusted win rate: {stats.win_rate * config.win_rate_degradation * 100:.1f}%")
    
    # ==========================================================================
    # Kelly Criterion
    # ==========================================================================
    print(f"\n{'='*70}")
    print("KELLY CRITERION")
    print(f"{'='*70}")
    
    kelly = calculate_kelly(stats)
    print(f"\n  Win probability: {kelly['win_prob']*100:.1f}%")
    print(f"  Avg win: ${kelly['avg_win']:.4f}")
    print(f"  Avg loss: ${kelly['avg_loss']:.4f}")
    print(f"  Win/Loss ratio: {kelly['win_loss_ratio']:.2f}")
    print(f"\n  Full Kelly: {kelly['full_kelly']*100:.1f}%")
    print(f"  Half Kelly (recommended): {kelly['half_kelly']*100:.1f}%")
    print(f"  Quarter Kelly (conservative): {kelly['quarter_kelly']*100:.1f}%")
    
    # ==========================================================================
    # Compounding Projection
    # ==========================================================================
    print(f"\n{'='*70}")
    print("COMPOUNDING PROJECTION (with liquidity cap)")
    print(f"{'='*70}")
    print(f"\n{'Months':>6} {'Trades':>7} {'Balance':>12} {'ROI':>10} {'Daily $':>10} {'Capped':>8}")
    print("-" * 60)
    
    for months in [1, 2, 3, 6, 9, 12]:
        result = run_compounding_projection(stats, config, months)
        print(f"{months:>6} {result['n_trades']:>7} ${result['final_balance']:>11.2f} "
              f"{result['roi']:>+9.1f}% ${result['daily_profit']:>9.2f} {result['capped_pct']:>7.1f}%")
    
    # ==========================================================================
    # Win Rate Sensitivity
    # ==========================================================================
    print(f"\n{'='*70}")
    print("WIN RATE SENSITIVITY (6 months, compounding)")
    print(f"{'='*70}")
    print(f"\n{'Win Rate':>12} {'Final $':>14} {'ROI':>12} {'Daily $':>12}")
    print("-" * 55)
    
    for wr_mult in [0.90, 0.95, 1.00, 1.02, 1.05]:
        test_wr = stats.win_rate * wr_mult
        result = run_compounding_projection(stats, config, 6, win_rate_override=test_wr)
        
        label = f"{test_wr*100:.1f}%"
        if wr_mult == 1.0:
            label += " (backtest)"
        elif wr_mult == config.win_rate_degradation:
            label += " (default)"
        
        print(f"{label:>12} ${result['final_balance']:>13.2f} "
              f"{result['roi']:>+11.1f}% ${result['daily_profit']:>11.2f}")
    
    # ==========================================================================
    # Kelly vs Fixed Position Comparison
    # ==========================================================================
    print(f"\n{'='*70}")
    print("KELLY VS FIXED POSITION SIZING (6 months, compounding)")
    print(f"{'='*70}")
    
    position_sizes = [
        ("Your 10%", 0.10),
        ("Quarter Kelly", kelly['quarter_kelly']),
        ("Half Kelly", kelly['half_kelly']),
        ("Full Kelly", kelly['full_kelly']),
    ]
    
    print(f"\n{'Strategy':>16} {'Pos %':>8} {'Final $':>12} {'ROI':>10} {'Daily $':>10}")
    print("-" * 60)
    
    for name, pos_pct in position_sizes:
        if pos_pct <= 0:
            print(f"{name:>16} {'N/A':>8} {'N/A':>12} {'N/A':>10} {'N/A':>10}")
            continue
            
        temp_config = ModelConfig(
            starting_balance=config.starting_balance,
            position_pct=pos_pct,
            liquidity_cap=config.liquidity_cap,
            win_rate_degradation=config.win_rate_degradation,
            backtest_days=config.backtest_days,
            trading_time_factor=config.trading_time_factor,
        )
        result = run_compounding_projection(stats, temp_config, 6)
        print(f"{name:>16} {pos_pct*100:>7.1f}% ${result['final_balance']:>11.2f} "
              f"{result['roi']:>+9.1f}% ${result['daily_profit']:>9.2f}")
    
    # ==========================================================================
    # Position Size Sensitivity
    # ==========================================================================
    print(f"\n{'='*70}")
    print("POSITION SIZE SENSITIVITY (6 months, compounding)")
    print(f"{'='*70}")
    print(f"\n{'Position %':>12} {'Final $':>14} {'ROI':>12} {'Risk Level':>14}")
    print("-" * 55)
    
    for pos_pct in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        temp_config = ModelConfig(
            starting_balance=config.starting_balance,
            position_pct=pos_pct,
            liquidity_cap=config.liquidity_cap,
            win_rate_degradation=config.win_rate_degradation,
            backtest_days=config.backtest_days,
            trading_time_factor=config.trading_time_factor,
        )
        result = run_compounding_projection(stats, temp_config, 6)
        
        risk = "Conservative" if pos_pct <= 0.05 else "Moderate" if pos_pct <= 0.10 else "Aggressive" if pos_pct <= 0.15 else "High Risk"
        print(f"{pos_pct*100:>11.0f}% ${result['final_balance']:>13.2f} "
              f"{result['roi']:>+11.1f}% {risk:>14}")
    
    # ==========================================================================
    # Linear Projection (sanity check)
    # ==========================================================================
    print(f"\n{'='*70}")
    print("LINEAR PROJECTION (No Compounding - Sanity Check)")
    print(f"{'='*70}")
    print(f"\nFlat ${config.starting_balance * config.position_pct:.0f} bets (no scaling with account growth)")
    print(f"\n{'Months':>6} {'Trades':>7} {'Profit':>12} {'Balance':>12} {'ROI':>10}")
    print("-" * 50)
    
    for months in [1, 3, 6, 12]:
        result = run_linear_projection(stats, config, months)
        print(f"{months:>6} {result['n_trades']:>7} ${result['total_profit']:>+11.2f} "
              f"${result['final_balance']:>11.2f} {result['roi']:>+9.1f}%")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    result_6mo = run_compounding_projection(stats, config, 6)
    result_linear_6mo = run_linear_projection(stats, config, 6)
    
    print(f"\n6-Month Projection (starting with ${config.starting_balance}):")
    print(f"  With compounding: ${result_6mo['final_balance']:.2f} ({result_6mo['roi']:+.0f}% ROI)")
    print(f"  Without compounding: ${result_linear_6mo['final_balance']:.2f} ({result_linear_6mo['roi']:+.0f}% ROI)")
    print(f"  Expected daily profit: ${result_6mo['daily_profit']:.2f}")
    print(f"  Trades per day: {stats.trades_per_day:.1f}")


if __name__ == "__main__":
    main()
