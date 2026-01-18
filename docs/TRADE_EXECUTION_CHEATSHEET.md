# BTC 15m Trader - Trade Execution Cheat Sheet

This document covers **all conditions** that determine whether a trade will be placed. Organized by stage in the trading pipeline.

---

## 📊 STAGE 1: Market Discovery & Filtering

| Config | Default | What It Does |
|--------|---------|--------------|
| `asset` | `"BTC"` | Asset to trade (filters KXBTC15M markets) |
| `horizon_minutes` | `60` | Only poll markets expiring within this window |
| `limit_markets` | `2` | Max number of active markets to track per poll |
| `min_seconds_to_expiry` | `30` | Skip markets with TTE below this (too late to trade) |

---

## 📈 STAGE 2: Spot Price Selection

| Config | Default | What It Does |
|--------|---------|--------------|
| `use_kraken_composite` | `False` | If True, averages Coinbase + Kraken for composite mid |

**Spot Source Priority Chain:**
1. `coinbase_vwap_60s` → Used when TTE ≤ 60s (matches BRTI settlement)
2. `composite_mid` → Only if `use_kraken_composite=True` and both exchanges available
3. `coinbase_mid` → Coinbase orderbook midpoint (default)
4. `coinbase_ticker` → Fallback to ticker price

---

## 🧮 STAGE 3: Signal Generation (Edge Calculation)

| Concept | Formula | Notes |
|---------|---------|-------|
| `p_yes` | Model probability | From ML model inference |
| `p_no` | `1 - p_yes` | Complementary probability |
| `market_p_yes` | Kalshi YES ask / 100 | Market-implied probability |
| `market_p_no` | Kalshi NO ask / 100 | Market-implied probability |
| `EV_yes` | `p_yes - market_p_yes` | Raw edge for YES |
| `EV_no` | `p_no - market_p_no` | Raw edge for NO |
| `EV_yes_after_fees` | `EV_yes - fee_yes` | Fee-adjusted edge |
| `EV_no_after_fees` | `EV_no - fee_no` | Fee-adjusted edge |

**Decision Rule:**
- Buy YES if `EV_yes_after_fees > threshold`
- Buy NO if `EV_no_after_fees > threshold`
- If both qualify → pick higher EV
- If neither qualifies → no trade (`side=None`)

| Config | Default | What It Does |
|--------|---------|--------------|
| `threshold` | `0.0` | Minimum EV after fees to generate a decision |
| `fee_mode` | `"taker"` | Fee assumption: `"taker"` or `"maker"` |

---

## 💰 STAGE 4: Position Sizing (Kelly)

| Config | Default | What It Does |
|--------|---------|--------------|
| `use_kalshi_balance_as_bankroll` | `True` | Use Kalshi available balance as bankroll |
| `bankroll_usd` | `None` | Manual bankroll override (if set, ignores Kalshi balance) |
| `kelly_mult` | `0.1` | Fractional Kelly multiplier (0.1 = 1/10th Kelly) |
| `kelly_max_fraction` | `0.1` | Cap Kelly fraction at this max |

**Kelly Formula (binary):**
```
kelly_full = (p_win - price) / (1 - price)
kelly_fraction = min(kelly_full * kelly_mult, kelly_max_fraction)
contracts = floor(bankroll * kelly_fraction / effective_cost)
```

**⛔ BLOCK if:** `contracts <= 0`

---

## 🚦 STAGE 5: Pre-Order Execution Gates

### 5.1 Trading Paused Backoff
**Triggered by:** Kalshi returns `trading_is_paused` error  
**Effect:** Blocks ALL orders for 60 seconds  
**Reason:** `trading_paused_backoff`

---

### 5.2 Re-Entry After Flatten Block

| Config | Default | What It Does |
|--------|---------|--------------|
| `allow_reentry_after_flatten` | `True` | If False, once you flatten a ticker, NO re-entry until expiry |

**⛔ BLOCK if:** You flattened this ticker previously AND `allow_reentry_after_flatten=False`  
**Reason:** `reentry_after_flatten`

---

### 5.3 Pending Flip Side Mismatch

**⛔ BLOCK if:** A pending flip-reentry exists but the new decision side doesn't match  
**Reason:** `pending_flip_side_mismatch`

---

### 5.4 Entry Rate Limiting

| Config | Default | What It Does |
|--------|---------|--------------|
| `min_seconds_between_entry_orders` | `20` | Minimum seconds between entry attempts per ticker |

**⛔ BLOCK if:** Last entry order was < `min_seconds_between_entry_orders` ago  
**Applies to:** All entry modes (taker_ioc AND maker_only)  
**Reason:** `entry_rate_limit`

---

### 5.5 Maker-Only Position Exists

| Config | Default | What It Does |
|--------|---------|--------------|
| `entry_mode` | `"taker_ioc"` | Entry execution mode |

**⛔ BLOCK if:** `entry_mode="maker_only"` AND you already have a position in this ticker  
**Reason:** `maker_position_exists`

---

### 5.6 Flip/Flatten Logic

When you hold YES and model says NO (or vice versa):

| Config | Default | What It Does |
|--------|---------|--------------|
| `exit_delta` | `0.10` | Minimum probability shift to exit (p_held must be < 0.5 - exit_delta) |
| `catastrophic_exit_delta` | `0.18` | Emergency exit threshold (bypasses min_hold_seconds) |
| `min_hold_seconds` | `300` | Minimum hold time before flattening (unless catastrophic) |
| `flatten_tier_enabled` | `True` | Enable tiered flatten (half first, then remainder) |
| `flatten_tier_seconds` | `30` | Seconds to wait between tier 1 and tier 2 |

**Flatten Method:** Buys opposite side instead of selling current side (lower fees!)
- Example: Holding YES @ 80% market → Buy NO @ 20¢ instead of Sell YES @ 78¢
- Fee savings: ~75% lower fees on the flatten transaction

**Tiered Flatten Flow:**
1. **Tier 1:** Flip signal detected → flatten HALF of position
2. **Wait:** `flatten_tier_seconds` (30s default)
3. **Tier 2:** If flip signal still valid → flatten remaining half → arm for re-entry

**⛔ FLIP BLOCK if:**
1. `p_held >= 0.5 - exit_delta` (probability hasn't moved enough against you)
2. `held_for < min_hold_seconds` AND `p_held >= 0.5 - catastrophic_exit_delta` (not catastrophic yet)

**Note:** Flatten/exit is NEVER blocked by entry gates - only re-entry is gated.

---

### 5.7 Spot-Strike Sanity Check

| Config | Default | What It Does |
|--------|---------|--------------|
| `spot_strike_sanity_enabled` | `True` | Enable spot vs strike deviation check |
| `max_spot_strike_deviation_fraction` | `0.02` | Max allowed \|spot - strike\| / spot (2%) |

**⛔ BLOCK if:** `|spot - strike| / spot > max_spot_strike_deviation_fraction`  
**Purpose:** Catches bad strike data early in market  
**Reason:** `spot_strike_deviation`

---

### 5.8 Dead Zone Guard

| Config | Default | What It Does |
|--------|---------|--------------|
| `dead_zone` | `0.05` | Minimum distance from 50% to enter |

**⛔ BLOCK if:** `|p_side - 0.5| < dead_zone` (model is too uncertain)  
**Applies to:** Entries only (not exits)  
**Reason:** `deadzone`

---

### 5.9 Minimum Entry Edge (Dynamic by TTE)

| Config | Default | What It Does |
|--------|---------|--------------|
| `min_entry_edge` | `0.025` | EV threshold when TTE ≥ 5 minutes (2.5%) |
| `min_entry_edge_late` | `0.05` | EV threshold when TTE < 5 minutes (5%) |
| `entry_edge_tte_threshold` | `300` | TTE threshold in seconds (5 minutes) |

**Logic:**
```python
effective_edge = min_entry_edge_late if TTE < 300 else min_entry_edge
```

**⛔ BLOCK if:** `EV_after_fees <= effective_edge`  
**Applies to:** Entries only (not exits)  
**Reason:** `min_entry_edge`

---

### 5.10 Monotonicity Guard 🆕

| Config | Default | What It Does |
|--------|---------|--------------|
| `monotonicity_guard_enabled` | `True` | Enable late-market microstructure guard |
| `monotonicity_tte_threshold` | `360` | TTE threshold (6 minutes) |
| `monotonicity_market_confidence` | `0.80` | Market must be ≥ 80% confident to trigger |
| `monotonicity_model_diff` | `0.05` | Model must agree within 5% to block |

**⛔ BLOCK if ALL of:**
1. `TTE < 360 seconds` (within 6 minutes of expiry)
2. Market confident: `market_p_yes >= 0.80` OR `market_p_no >= 0.80`
3. Model agrees: `|p_model - market_p| < 0.05`

**Purpose:** Avoids microstructure noise trades when outcome is near-certain  
**Applies to:** Entries only (not exits)  
**Reason:** `monotonicity_guard`

---

### 5.11 Max Time to Expiry Gate

| Config | Default | What It Does |
|--------|---------|--------------|
| `max_seconds_to_expiry_to_trade` | `840` | Max TTE to place trades (14 minutes) |

**⛔ BLOCK if:** `TTE > max_seconds_to_expiry_to_trade`  
**Purpose:** Don't trade too early in market when data may be stale

---

### 5.12 Maker-Only Spread Guard

| Config | Default | What It Does |
|--------|---------|--------------|
| `max_entry_spread_cents` | `5` | Max bid-ask spread for maker entry |
| `maker_improve_cents` | `0` | How much to improve on best bid |

**⛔ BLOCK if:** `entry_mode="maker_only"` AND `spread > max_entry_spread_cents`  
**Reason:** Wide spread = risky maker entry

---

### 5.13 Risk Limits

| Config | Default | What It Does |
|--------|---------|--------------|
| `max_total_abs_contracts` | `7` | Max total contracts across all tickers |
| `max_total_exposure_usd` | `7.0` | Max total $ exposure across all tickers |
| `max_ticker_abs_contracts` | `7` | Max contracts per individual ticker |
| `max_ticker_exposure_usd` | `7.0` | Max $ exposure per individual ticker |

**⛔ BLOCK if ANY limit exceeded after projected trade**  
**Reason:** `risk_limits`

---

## 🔄 STAGE 6: Order Placement

| Config | Default | What It Does |
|--------|---------|--------------|
| `enable_execution` | `True` | Master switch for order placement |
| `dry_run` | `False` | If True, logs orders but doesn't submit |
| `time_in_force` | `"immediate_or_cancel"` | Order TIF (IoC default for taker) |
| `max_orders_per_poll` | `1` | Max orders placed per polling cycle |

---

## 📋 Quick Reference: All Block Reasons

| Reason Code | Gate | Applies To |
|-------------|------|------------|
| `trading_paused_backoff` | 5.1 | All orders |
| `reentry_after_flatten` | 5.2 | Entries |
| `pending_flip_side_mismatch` | 5.3 | Entries |
| `entry_rate_limit` | 5.4 | Entries |
| `maker_position_exists` | 5.5 | Entries (maker_only) |
| `spot_strike_deviation` | 5.7 | Entries |
| `deadzone` | 5.8 | Entries |
| `min_entry_edge` | 5.9 | Entries |
| `monotonicity_guard` | 5.10 | Entries |
| `risk_limits` | 5.13 | Entries |

---

## 🎯 Decision Flow Summary

```
Market Poll → Spot Price → Model Inference → Edge Calc → Kelly Sizing
                                                              ↓
                                            ┌─────────────────┴─────────────────┐
                                            │      EXECUTION GATES              │
                                            │  (all must pass for entry)        │
                                            ├───────────────────────────────────┤
                                            │ ✓ Not trading_paused              │
                                            │ ✓ Re-entry allowed                │
                                            │ ✓ Rate limit passed               │
                                            │ ✓ No existing position (maker)    │
                                            │ ✓ Spot-strike sanity              │
                                            │ ✓ Outside dead zone               │
                                            │ ✓ Above min_entry_edge            │
                                            │ ✓ Monotonicity guard passed       │
                                            │ ✓ Within max TTE                  │
                                            │ ✓ Spread OK (maker)               │
                                            │ ✓ Risk limits OK                  │
                                            └───────────────────────────────────┘
                                                              ↓
                                                       PLACE ORDER
```

---

## 📁 File Reference

| File | Purpose |
|------|---------|
| [run_btc15m_trader.py](../src/tradebot/tools/run_btc15m_trader.py) | Config definitions, main loop |
| [order_execution.py](../src/tradebot/tools/order_execution.py) | All execution gates & order placement |
| [trade_edge_calculation.py](../src/tradebot/tools/trade_edge_calculation.py) | EV calculation & trade decision |
| [btc15m_trade_signal.py](../src/tradebot/tools/btc15m_trade_signal.py) | Signal assembly & logging |
| [position_size.py](../src/tradebot/tools/position_size.py) | Kelly sizing logic |
| [inventory_check.py](../src/tradebot/tools/inventory_check.py) | Position & exposure tracking |
| [btc15m_live_inference.py](../src/tradebot/tools/btc15m_live_inference.py) | Model inference & spot selection |
| [kalshi_market_poll.py](../src/tradebot/tools/kalshi_market_poll.py) | Market polling & price fetching |

---

*Last updated: January 2026*
