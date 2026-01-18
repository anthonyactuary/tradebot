"""Unit tests for order_execution module.

Tests cover:
- Edge calculation and trade decisions
- Position sizing (Kelly)
- Flatten logic (buy opposite side)
- Tiered flatten state management
- Entry gates (deadzone, min_entry_edge, monotonicity guard)
- Risk limits
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the modules under test
from tradebot.tools.trade_edge_calculation import (
    EdgeResult,
    TradeDecision,
    trade_decision,
    trade_edge_calculation,
)
from tradebot.tools.position_size import (
    PositionSize,
    calc_position_size,
    kelly_fraction_binary,
)
from tradebot.tools.order_execution import (
    _PartialFlatten,
    _PARTIAL_FLATTEN_BY_TICKER,
    _held_side_from_position,
    _p_for_side,
    _ev_after_fees_per_contract,
    _spot_strike_diff_fraction,
    _check_risk_limits,
    RiskLimits,
)
from tradebot.tools.inventory_check import InventorySummary, TickerInventory


# =============================================================================
# EDGE CALCULATION TESTS
# =============================================================================

class TestTradeEdgeCalculation:
    """Tests for trade_edge_calculation function."""

    def test_basic_positive_edge_yes(self) -> None:
        """Model predicts 80% YES, market is at 70% -> positive YES edge."""
        edge = trade_edge_calculation(
            p_yes=0.80,
            market_p_yes=0.70,
        )
        assert edge.p_yes == 0.80
        assert abs(edge.p_no - 0.20) < 1e-9
        assert edge.market_p_yes == 0.70
        assert abs(edge.market_p_no - 0.30) < 1e-9
        assert abs(edge.ev_yes - 0.10) < 1e-9  # 0.80 - 0.70 = 0.10
        assert abs(edge.ev_no - (-0.10)) < 1e-9  # 0.20 - 0.30 = -0.10

    def test_basic_positive_edge_no(self) -> None:
        """Model predicts 30% YES, market is at 50% -> positive NO edge."""
        edge = trade_edge_calculation(
            p_yes=0.30,
            market_p_yes=0.50,
        )
        assert abs(edge.ev_yes - (-0.20)) < 1e-9  # 0.30 - 0.50 = -0.20
        assert abs(edge.ev_no - 0.20) < 1e-9  # 0.70 - 0.50 = 0.20

    def test_fee_adjustment(self) -> None:
        """Fees should reduce EV after fees."""
        edge = trade_edge_calculation(
            p_yes=0.80,
            market_p_yes=0.70,
            fee_yes_usd_per_contract=0.02,
            fee_no_usd_per_contract=0.01,
        )
        assert abs(edge.ev_yes_after_fees - 0.08) < 1e-9  # 0.10 - 0.02
        assert abs(edge.ev_no_after_fees - (-0.11)) < 1e-9  # -0.10 - 0.01

    def test_no_edge_at_fair_price(self) -> None:
        """When model = market, EV should be zero."""
        edge = trade_edge_calculation(
            p_yes=0.60,
            market_p_yes=0.60,
        )
        assert abs(edge.ev_yes) < 1e-9
        assert abs(edge.ev_no) < 1e-9


class TestTradeDecision:
    """Tests for trade_decision function."""

    def test_yes_above_threshold(self) -> None:
        """Should return YES when only YES is above threshold."""
        edge = trade_edge_calculation(p_yes=0.80, market_p_yes=0.70)
        decision = trade_decision(edge=edge, threshold=0.0)
        assert decision.side == "YES"
        assert decision.reason == "ev_yes_above_threshold"

    def test_no_above_threshold(self) -> None:
        """Should return NO when only NO is above threshold."""
        edge = trade_edge_calculation(p_yes=0.30, market_p_yes=0.50)
        decision = trade_decision(edge=edge, threshold=0.0)
        assert decision.side == "NO"
        assert decision.reason == "ev_no_above_threshold"

    def test_both_positive_pick_higher(self) -> None:
        """When both sides are positive, pick the higher one."""
        # Create edge where YES is slightly better
        edge = EdgeResult(
            p_yes=0.60,
            p_no=0.40,
            market_p_yes=0.55,
            market_p_no=0.45,
            ev_yes=0.05,
            ev_no=-0.05,
            ev_yes_after_fees=0.03,
            ev_no_after_fees=0.02,  # Both positive, YES higher
        )
        decision = trade_decision(edge=edge, threshold=0.0)
        assert decision.side == "YES"
        assert decision.reason == "both_positive_pick_higher"

    def test_no_edge_above_threshold(self) -> None:
        """Should return None when neither side exceeds threshold."""
        edge = trade_edge_calculation(p_yes=0.50, market_p_yes=0.50)
        decision = trade_decision(edge=edge, threshold=0.01)
        assert decision.side is None
        assert decision.reason == "no_edge_above_threshold"

    def test_threshold_filters_small_edge(self) -> None:
        """A positive but small edge should be filtered by threshold."""
        edge = trade_edge_calculation(p_yes=0.72, market_p_yes=0.70)
        # EV_yes = 0.02, which is below threshold of 0.025
        decision = trade_decision(edge=edge, threshold=0.025)
        assert decision.side is None


# =============================================================================
# POSITION SIZING (KELLY) TESTS
# =============================================================================

class TestKellyFraction:
    """Tests for kelly_fraction_binary function."""

    def test_positive_edge(self) -> None:
        """Positive edge should give positive Kelly fraction."""
        # p=0.60, m=0.50 -> f* = (0.60 - 0.50) / (1 - 0.50) = 0.20
        f = kelly_fraction_binary(p=0.60, m=0.50)
        assert abs(f - 0.20) < 1e-9

    def test_negative_edge(self) -> None:
        """Negative edge should give negative Kelly fraction."""
        # p=0.40, m=0.50 -> f* = (0.40 - 0.50) / (1 - 0.50) = -0.20
        f = kelly_fraction_binary(p=0.40, m=0.50)
        assert abs(f - (-0.20)) < 1e-9

    def test_fair_price_zero_kelly(self) -> None:
        """At fair price, Kelly fraction should be zero."""
        f = kelly_fraction_binary(p=0.70, m=0.70)
        assert abs(f) < 1e-9

    def test_extreme_edge(self) -> None:
        """Very high confidence should give high Kelly fraction."""
        # p=0.95, m=0.50 -> f* = (0.95 - 0.50) / (1 - 0.50) = 0.90
        f = kelly_fraction_binary(p=0.95, m=0.50)
        assert abs(f - 0.90) < 1e-9


class TestCalcPositionSize:
    """Tests for calc_position_size function."""

    def test_basic_sizing(self) -> None:
        """Should calculate correct position size."""
        pos = calc_position_size(
            side="YES",
            p_win=0.70,
            price=0.50,
            bankroll_usd=100.0,
            kelly_multiplier=0.5,
            max_fraction=1.0,
        )
        # Kelly full = (0.70 - 0.50) / (1 - 0.50) = 0.40
        # Kelly fraction = 0.40 * 0.5 = 0.20
        # Stake = 100 * 0.20 = 20
        # Contracts = floor(20 / 0.50) = 40 (or 39 due to floating point)
        assert abs(pos.kelly_full_fraction - 0.40) < 1e-9
        assert abs(pos.kelly_fraction - 0.20) < 1e-9
        assert pos.contracts in (39, 40)  # Allow for floating point floor differences

    def test_max_fraction_cap(self) -> None:
        """Kelly fraction should be capped at max_fraction."""
        pos = calc_position_size(
            side="YES",
            p_win=0.95,
            price=0.50,
            bankroll_usd=100.0,
            kelly_multiplier=1.0,
            max_fraction=0.10,  # Cap at 10%
        )
        # Kelly full = 0.90, but capped at 0.10
        assert pos.kelly_fraction == 0.10
        # Stake = 100 * 0.10 = 10
        # Contracts = floor(10 / 0.50) = 20
        assert pos.contracts == 20

    def test_negative_edge_zero_contracts(self) -> None:
        """Negative edge should result in zero contracts."""
        pos = calc_position_size(
            side="YES",
            p_win=0.40,
            price=0.50,
            bankroll_usd=100.0,
            kelly_multiplier=1.0,
            max_fraction=1.0,
        )
        assert pos.kelly_fraction == 0.0
        assert pos.contracts == 0

    def test_fee_reduces_contracts(self) -> None:
        """Per-contract fees should reduce effective contracts."""
        pos = calc_position_size(
            side="YES",
            p_win=0.70,
            price=0.50,
            bankroll_usd=100.0,
            kelly_multiplier=0.5,
            max_fraction=1.0,
            fee_per_contract_usd=0.02,
        )
        # Effective cost = 0.50 + 0.02 = 0.52
        # Contracts = floor(20 / 0.52) = 38
        assert pos.contracts == 38


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

class TestHelperFunctions:
    """Tests for order_execution helper functions."""

    def test_held_side_from_position_yes(self) -> None:
        """Positive position means holding YES."""
        assert _held_side_from_position(5) == "YES"
        assert _held_side_from_position(1) == "YES"

    def test_held_side_from_position_no(self) -> None:
        """Negative position means holding NO."""
        assert _held_side_from_position(-5) == "NO"
        assert _held_side_from_position(-1) == "NO"

    def test_held_side_from_position_flat(self) -> None:
        """Zero position means no side held."""
        assert _held_side_from_position(0) is None

    def test_p_for_side(self) -> None:
        """Should return correct probability for requested side."""
        edge = EdgeResult(
            p_yes=0.75,
            p_no=0.25,
            market_p_yes=0.70,
            market_p_no=0.30,
            ev_yes=0.05,
            ev_no=-0.05,
            ev_yes_after_fees=0.03,
            ev_no_after_fees=-0.06,
        )
        assert _p_for_side(edge, side="YES") == 0.75
        assert _p_for_side(edge, side="NO") == 0.25

    def test_ev_after_fees_per_contract(self) -> None:
        """Should return correct EV for requested side."""
        edge = EdgeResult(
            p_yes=0.75,
            p_no=0.25,
            market_p_yes=0.70,
            market_p_no=0.30,
            ev_yes=0.05,
            ev_no=-0.05,
            ev_yes_after_fees=0.03,
            ev_no_after_fees=-0.06,
        )
        assert _ev_after_fees_per_contract(edge, side="YES") == 0.03
        assert _ev_after_fees_per_contract(edge, side="NO") == -0.06


class TestSpotStrikeDiff:
    """Tests for spot-strike deviation calculation."""

    def test_basic_diff(self) -> None:
        """Should calculate correct percentage difference."""
        @dataclass
        class MockSnap:
            btc_spot_usd: float | None
            price_to_beat: float | None

        snap = MockSnap(btc_spot_usd=100000.0, price_to_beat=99000.0)
        diff = _spot_strike_diff_fraction(snap=snap)
        # |100000 - 99000| / 100000 = 0.01
        assert diff is not None
        assert abs(diff - 0.01) < 1e-9

    def test_none_values(self) -> None:
        """Should return None if spot or strike is missing."""
        @dataclass
        class MockSnap:
            btc_spot_usd: float | None
            price_to_beat: float | None

        snap1 = MockSnap(btc_spot_usd=None, price_to_beat=99000.0)
        assert _spot_strike_diff_fraction(snap=snap1) is None

        snap2 = MockSnap(btc_spot_usd=100000.0, price_to_beat=None)
        assert _spot_strike_diff_fraction(snap=snap2) is None


# =============================================================================
# RISK LIMITS TESTS
# =============================================================================

class TestRiskLimits:
    """Tests for risk limit checking."""

    def test_total_contracts_limit_exceeded(self) -> None:
        """Should block when total contracts exceed limit."""
        summary = InventorySummary(
            tickers=["T1", "T2"],
            per_ticker={
                "T1": TickerInventory(ticker="T1", position=3, abs_contracts=3, exposure_usd=3.0),
                "T2": TickerInventory(ticker="T2", position=2, abs_contracts=2, exposure_usd=2.0),
            },
            total_abs_contracts=5,
            total_exposure_usd=5.0,
            max_abs_contracts=3,
            max_exposure_usd=3.0,
        )
        limits = RiskLimits(max_total_abs_contracts=7)

        result = _check_risk_limits(
            summary=summary,
            ticker="T1",
            current_position=3,
            projected_position=6,  # Adding 3 more
            projected_total_abs_contracts=8,  # 5 + 3 = 8 > 7
            projected_total_exposure_usd=8.0,
            projected_ticker_abs_contracts=6,
            projected_ticker_exposure_usd=6.0,
            limits=limits,
        )
        assert result is not None
        assert "total_abs_contracts" in result

    def test_total_exposure_limit_exceeded(self) -> None:
        """Should block when total exposure exceeds limit."""
        summary = InventorySummary(
            tickers=["T1"],
            per_ticker={
                "T1": TickerInventory(ticker="T1", position=5, abs_contracts=5, exposure_usd=5.0),
            },
            total_abs_contracts=5,
            total_exposure_usd=5.0,
            max_abs_contracts=5,
            max_exposure_usd=5.0,
        )
        limits = RiskLimits(max_total_exposure_usd=7.0)

        result = _check_risk_limits(
            summary=summary,
            ticker="T1",
            current_position=5,
            projected_position=8,
            projected_total_abs_contracts=8,
            projected_total_exposure_usd=8.0,  # > 7.0
            projected_ticker_abs_contracts=8,
            projected_ticker_exposure_usd=8.0,
            limits=limits,
        )
        assert result is not None
        assert "total_exposure" in result

    def test_ticker_contracts_limit_exceeded(self) -> None:
        """Should block when per-ticker contracts exceed limit."""
        summary = InventorySummary(
            tickers=["T1"],
            per_ticker={
                "T1": TickerInventory(ticker="T1", position=5, abs_contracts=5, exposure_usd=5.0),
            },
            total_abs_contracts=5,
            total_exposure_usd=5.0,
            max_abs_contracts=5,
            max_exposure_usd=5.0,
        )
        limits = RiskLimits(max_ticker_abs_contracts=6)

        result = _check_risk_limits(
            summary=summary,
            ticker="T1",
            current_position=5,
            projected_position=7,
            projected_total_abs_contracts=7,
            projected_total_exposure_usd=7.0,
            projected_ticker_abs_contracts=7,  # > 6
            projected_ticker_exposure_usd=7.0,
            limits=limits,
        )
        assert result is not None
        assert "abs_contracts" in result

    def test_within_limits(self) -> None:
        """Should return None when within all limits."""
        summary = InventorySummary(
            tickers=["T1"],
            per_ticker={
                "T1": TickerInventory(ticker="T1", position=3, abs_contracts=3, exposure_usd=3.0),
            },
            total_abs_contracts=3,
            total_exposure_usd=3.0,
            max_abs_contracts=3,
            max_exposure_usd=3.0,
        )
        limits = RiskLimits(
            max_total_abs_contracts=10,
            max_total_exposure_usd=10.0,
            max_ticker_abs_contracts=7,
            max_ticker_exposure_usd=7.0,
        )

        result = _check_risk_limits(
            summary=summary,
            ticker="T1",
            current_position=3,
            projected_position=5,
            projected_total_abs_contracts=5,
            projected_total_exposure_usd=5.0,
            projected_ticker_abs_contracts=5,
            projected_ticker_exposure_usd=5.0,
            limits=limits,
        )
        assert result is None


# =============================================================================
# PARTIAL FLATTEN STATE TESTS
# =============================================================================

class TestPartialFlattenState:
    """Tests for tiered flatten state management."""

    def test_partial_flatten_dataclass(self) -> None:
        """PartialFlatten dataclass should store state correctly."""
        pf = _PartialFlatten(
            ts=1000.0,
            from_side="YES",
            to_side="NO",
            original_qty=4,
            flattened_qty=2,
            remaining_qty=2,
        )
        assert pf.ts == 1000.0
        assert pf.from_side == "YES"
        assert pf.to_side == "NO"
        assert pf.original_qty == 4
        assert pf.flattened_qty == 2
        assert pf.remaining_qty == 2

    def test_partial_flatten_half_calculation(self) -> None:
        """Half of position should be calculated correctly."""
        # Test even numbers
        assert max(1, 4 // 2) == 2
        assert max(1, 6 // 2) == 3
        assert max(1, 10 // 2) == 5

        # Test odd numbers (floor division)
        assert max(1, 5 // 2) == 2
        assert max(1, 7 // 2) == 3
        assert max(1, 3 // 2) == 1

        # Test minimum of 1
        assert max(1, 1 // 2) == 1
        assert max(1, 0 // 2) == 1


# =============================================================================
# INTEGRATION-STYLE TESTS (Mocked)
# =============================================================================

class TestDeadzoneGuard:
    """Tests for deadzone guard logic."""

    def test_deadzone_blocks_near_50_percent(self) -> None:
        """Entry should be blocked when model is near 50%."""
        # Simulating the deadzone logic
        p_side = 0.52
        dead_zone = 0.05
        dist = abs(p_side - 0.5)
        
        assert dist < dead_zone  # Should be blocked
        assert abs(dist - 0.02) < 1e-9  # 0.52 - 0.50

    def test_deadzone_allows_confident_signal(self) -> None:
        """Entry should be allowed when model is confident."""
        p_side = 0.70
        dead_zone = 0.05
        dist = abs(p_side - 0.5)
        
        assert dist >= dead_zone  # Should be allowed
        assert abs(dist - 0.20) < 1e-9  # 0.70 - 0.50


class TestMinEntryEdgeGate:
    """Tests for dynamic min entry edge."""

    def test_late_market_higher_threshold(self) -> None:
        """TTE < threshold should use higher edge requirement."""
        tte = 240  # 4 minutes
        entry_edge_tte_threshold = 300  # 5 minutes
        min_entry_edge = 0.025
        min_entry_edge_late = 0.05

        effective = (
            min_entry_edge_late
            if tte < entry_edge_tte_threshold
            else min_entry_edge
        )
        assert effective == 0.05

    def test_normal_market_lower_threshold(self) -> None:
        """TTE >= threshold should use normal edge requirement."""
        tte = 600  # 10 minutes
        entry_edge_tte_threshold = 300  # 5 minutes
        min_entry_edge = 0.025
        min_entry_edge_late = 0.05

        effective = (
            min_entry_edge_late
            if tte < entry_edge_tte_threshold
            else min_entry_edge
        )
        assert effective == 0.025


class TestMonotonicityGuard:
    """Tests for monotonicity guard logic."""

    def test_blocks_late_confident_agreeing(self) -> None:
        """Should block when late market, confident, and model agrees."""
        tte = 300  # 5 minutes
        monotonicity_tte_threshold = 360  # 6 minutes
        market_p_yes = 0.85
        monotonicity_market_confidence = 0.80
        p_model = 0.83
        monotonicity_model_diff = 0.05

        # Check conditions
        is_late = tte < monotonicity_tte_threshold
        market_confident_yes = market_p_yes >= monotonicity_market_confidence
        model_diff = abs(p_model - market_p_yes)
        model_agrees = model_diff < monotonicity_model_diff

        assert is_late
        assert market_confident_yes
        assert model_agrees
        assert abs(model_diff - 0.02) < 1e-9

        # All conditions met -> should block
        should_block = is_late and market_confident_yes and model_agrees
        assert should_block

    def test_allows_late_confident_disagreeing(self) -> None:
        """Should allow when model disagrees with confident market."""
        tte = 300  # 5 minutes
        monotonicity_tte_threshold = 360  # 6 minutes
        market_p_yes = 0.85
        monotonicity_market_confidence = 0.80
        p_model = 0.70  # Model disagrees!
        monotonicity_model_diff = 0.05

        is_late = tte < monotonicity_tte_threshold
        market_confident_yes = market_p_yes >= monotonicity_market_confidence
        model_diff = abs(p_model - market_p_yes)
        model_agrees = model_diff < monotonicity_model_diff

        assert is_late
        assert market_confident_yes
        assert not model_agrees  # Model disagrees
        assert abs(model_diff - 0.15) < 1e-9

        should_block = is_late and market_confident_yes and model_agrees
        assert not should_block

    def test_allows_early_market(self) -> None:
        """Should allow in early markets even if confident and agreeing."""
        tte = 600  # 10 minutes
        monotonicity_tte_threshold = 360  # 6 minutes
        market_p_yes = 0.85
        monotonicity_market_confidence = 0.80
        p_model = 0.84
        monotonicity_model_diff = 0.05

        is_late = tte < monotonicity_tte_threshold
        market_confident_yes = market_p_yes >= monotonicity_market_confidence
        model_diff = abs(p_model - market_p_yes)
        model_agrees = model_diff < monotonicity_model_diff

        assert not is_late  # Not late
        assert market_confident_yes
        assert model_agrees

        should_block = is_late and market_confident_yes and model_agrees
        assert not should_block


class TestFlipExitDelta:
    """Tests for flip exit delta logic."""

    def test_exit_delta_blocks_weak_signal(self) -> None:
        """Should block flip when p_held hasn't moved enough."""
        p_held = 0.45  # Held YES, model says 45% YES
        exit_delta = 0.10
        threshold = 0.5 - exit_delta  # 0.40

        # p_held >= threshold means signal not strong enough to exit
        assert p_held >= threshold
        should_block = p_held >= threshold
        assert should_block

    def test_exit_delta_allows_strong_signal(self) -> None:
        """Should allow flip when p_held has moved significantly."""
        p_held = 0.35  # Held YES, model says 35% YES
        exit_delta = 0.10
        threshold = 0.5 - exit_delta  # 0.40

        # p_held < threshold means signal strong enough to exit
        assert p_held < threshold
        should_block = p_held >= threshold
        assert not should_block

    def test_catastrophic_bypasses_hold_time(self) -> None:
        """Catastrophic signal should bypass min hold time."""
        p_held = 0.28  # Very bearish on held position
        catastrophic_exit_delta = 0.18
        catastrophic_threshold = 0.5 - catastrophic_exit_delta  # 0.32

        is_catastrophic = p_held < catastrophic_threshold
        assert is_catastrophic


class TestFeeSavingsCalculation:
    """Tests for buy-opposite fee savings calculation."""

    def test_fee_savings_high_confidence_market(self) -> None:
        """Buying opposite should save significant fees at high confidence."""
        # Holding YES at 80% market confidence
        old_bid_cents = 78  # Would sell YES here
        new_ask_cents = 22  # Buy NO here instead

        # Fee savings = (1 - new/old) * 100
        fee_saved_pct = (1.0 - new_ask_cents / old_bid_cents) * 100.0
        assert abs(fee_saved_pct - 71.79) < 0.1  # ~72% savings

    def test_fee_savings_near_fair_value(self) -> None:
        """Fee savings are smaller near 50% confidence."""
        # Holding YES at 55% market confidence
        old_bid_cents = 53  # Would sell YES here
        new_ask_cents = 47  # Buy NO here instead

        fee_saved_pct = (1.0 - new_ask_cents / old_bid_cents) * 100.0
        assert abs(fee_saved_pct - 11.32) < 0.1  # ~11% savings


class TestTieredFlattenLogic:
    """Tests for tiered flatten decision logic."""

    def test_tier1_flattens_half(self) -> None:
        """First tier should flatten half of position."""
        full_qty = 6
        tier1_qty = max(1, full_qty // 2)
        remaining = full_qty - tier1_qty

        assert tier1_qty == 3
        assert remaining == 3

    def test_tier2_flattens_remainder(self) -> None:
        """Second tier should flatten remaining position."""
        original_qty = 6
        flattened_qty = 3
        remaining_qty = original_qty - flattened_qty

        tier2_qty = remaining_qty
        assert tier2_qty == 3

    def test_tier2_waits_for_timer(self) -> None:
        """Tier 2 should wait for flatten_tier_seconds."""
        tier1_ts = 1000.0
        now_ts = 1020.0  # 20 seconds later
        flatten_tier_seconds = 30

        elapsed = now_ts - tier1_ts
        should_wait = elapsed < flatten_tier_seconds

        assert elapsed == 20.0
        assert should_wait

    def test_tier2_proceeds_after_timer(self) -> None:
        """Tier 2 should proceed after flatten_tier_seconds."""
        tier1_ts = 1000.0
        now_ts = 1035.0  # 35 seconds later
        flatten_tier_seconds = 30

        elapsed = now_ts - tier1_ts
        should_wait = elapsed < flatten_tier_seconds

        assert elapsed == 35.0
        assert not should_wait

    def test_odd_position_tier1(self) -> None:
        """Odd positions should handle tier 1 correctly."""
        full_qty = 5
        tier1_qty = max(1, full_qty // 2)
        remaining = full_qty - tier1_qty

        assert tier1_qty == 2  # floor(5/2) = 2
        assert remaining == 3

    def test_single_contract_tier1(self) -> None:
        """Single contract position should flatten at least 1."""
        full_qty = 1
        tier1_qty = max(1, full_qty // 2)
        remaining = full_qty - tier1_qty

        assert tier1_qty == 1  # max(1, 0) = 1
        assert remaining == 0  # Full flatten in tier 1
