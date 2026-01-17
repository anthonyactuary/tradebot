"""Fit probability calibrators from a CSV of logged predictions.

This script reads a CSV containing raw model predictions and true outcomes,
then fits time-bucketed calibrators and saves them to disk.

Expected CSV columns:
    - p_raw_up: Raw probability of price going up (from model)
    - y_true: True outcome (1 if price went up, 0 otherwise)
    - tte_seconds: Time to expiry in seconds at prediction time

Usage:
    python -m tradebot.models.fit_calibrator_from_csv \
        --input predictions.csv \
        --out-dir calibration/ \
        --method platt

    # With isotonic regression instead:
    python -m tradebot.models.fit_calibrator_from_csv \
        --input predictions.csv \
        --out-dir calibration/ \
        --method isotonic
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any


def _require_deps() -> tuple[Any, Any]:
    """Import required dependencies with clear error messages."""
    try:
        import numpy as np
    except ImportError:
        print("ERROR: numpy is required. Install with: pip install numpy", file=sys.stderr)
        sys.exit(1)

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is required. Install with: pip install pandas", file=sys.stderr)
        sys.exit(1)

    return np, pd


def _print_bucket_stats(
    name: str,
    p_raw: "Any",
    y: "Any",
    calibrator: Any,
) -> None:
    """Print calibration diagnostics for a bucket."""
    np, _ = _require_deps()

    n = len(p_raw)
    if n == 0:
        print(f"  {name}: No samples")
        return

    # Basic stats
    mean_p = float(np.mean(p_raw))
    mean_y = float(np.mean(y))
    print(f"  {name}: {n} samples, mean_p_raw={mean_p:.4f}, base_rate={mean_y:.4f}")

    if calibrator is not None and calibrator.is_fitted:
        p_cal = calibrator.predict(p_raw)
        mean_p_cal = float(np.mean(p_cal))

        # Brier scores
        brier_raw = float(np.mean((p_raw - y) ** 2))
        brier_cal = float(np.mean((p_cal - y) ** 2))

        print(f"    Brier raw: {brier_raw:.6f}")
        print(f"    Brier cal: {brier_cal:.6f}")
        print(f"    Improvement: {(brier_raw - brier_cal) / brier_raw * 100:.2f}%")
        print(f"    mean_p_cal: {mean_p_cal:.4f}")

        if hasattr(calibrator, "a") and hasattr(calibrator, "b"):
            print(f"    Platt params: a={calibrator.a:.4f}, b={calibrator.b:.4f}")
        elif hasattr(calibrator, "n_knots"):
            print(f"    Isotonic knots: {calibrator.n_knots}")


def _print_reliability_table(
    name: str,
    p_raw: "Any",
    y: "Any",
    calibrator: Any,
    n_bins: int = 10,
) -> None:
    """Print reliability diagram as a table."""
    np, _ = _require_deps()

    if len(p_raw) == 0:
        return

    print(f"\n  {name} Reliability (raw vs calibrated):")
    print(f"  {'Bin':>8} {'Count':>6} {'AvgRaw':>8} {'AvgCal':>8} {'Empirical':>10}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")

    p_cal = calibrator.predict(p_raw) if (calibrator and calibrator.is_fitted) else p_raw

    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (p_raw >= lo) & (p_raw < hi) if i < n_bins - 1 else (p_raw >= lo) & (p_raw <= hi)
        count = int(mask.sum())

        if count > 0:
            avg_raw = float(np.mean(p_raw[mask]))
            avg_cal = float(np.mean(p_cal[mask]))
            empirical = float(np.mean(y[mask]))
            bin_label = f"[{lo:.1f},{hi:.1f})"
            print(f"  {bin_label:>8} {count:>6} {avg_raw:>8.4f} {avg_cal:>8.4f} {empirical:>10.4f}")
        else:
            bin_label = f"[{lo:.1f},{hi:.1f})"
            print(f"  {bin_label:>8} {count:>6} {'--':>8} {'--':>8} {'--':>10}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit probability calibrators from CSV of predictions"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input CSV with columns: p_raw_up, y_true, tte_seconds",
    )
    parser.add_argument(
        "--out-dir", "-o",
        type=str,
        default="./calibration",
        help="Output directory for calibrator artifacts (default: ./calibration)",
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["platt", "isotonic"],
        default="platt",
        help="Calibration method (default: platt)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples per bucket to fit calibrator (default: 10)",
    )
    args = parser.parse_args()

    np, pd = _require_deps()

    # Import calibrator
    from tradebot.models.prob_calibration_bucketed import (
        BucketedCalibrator,
        FINAL_BUCKET_THRESHOLD_SECONDS,
    )

    # Load data
    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)

    required_cols = ["p_raw_up", "y_true", "tte_seconds"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # Extract arrays
    p_raw = df["p_raw_up"].values.astype(float)
    y_true = df["y_true"].values.astype(int)
    tte_seconds = df["tte_seconds"].values.astype(float)

    print(f"Loaded {len(df)} samples")
    print(f"  p_raw_up range: [{p_raw.min():.4f}, {p_raw.max():.4f}]")
    print(f"  y_true mean (base rate): {y_true.mean():.4f}")
    print(f"  tte_seconds range: [{tte_seconds.min():.0f}, {tte_seconds.max():.0f}]")

    # Create and fit bucketed calibrator
    print(f"\nFitting {args.method} calibrators...")
    calibrator = BucketedCalibrator(method=args.method)
    counts = calibrator.fit(
        p_raw, y_true, tte_seconds,
        min_samples_per_bucket=args.min_samples,
    )

    print(f"\nBucket statistics:")
    print(f"  Early (tte > {FINAL_BUCKET_THRESHOLD_SECONDS}s): {counts['early']} samples")
    print(f"  Final60 (tte <= {FINAL_BUCKET_THRESHOLD_SECONDS}s): {counts['final60']} samples")

    # Split data for diagnostics
    mask_early = tte_seconds > FINAL_BUCKET_THRESHOLD_SECONDS
    mask_final60 = ~mask_early

    print(f"\nCalibration results:")
    _print_bucket_stats("Early", p_raw[mask_early], y_true[mask_early], calibrator.early_cal)
    _print_bucket_stats("Final60", p_raw[mask_final60], y_true[mask_final60], calibrator.final60_cal)

    # Reliability tables
    if counts["early"] >= args.min_samples:
        _print_reliability_table("Early", p_raw[mask_early], y_true[mask_early], calibrator.early_cal)
    if counts["final60"] >= args.min_samples:
        _print_reliability_table("Final60", p_raw[mask_final60], y_true[mask_final60], calibrator.final60_cal)

    # Overall calibrated performance
    print(f"\nOverall performance:")
    p_cal = calibrator.predict(p_raw, tte_seconds)
    brier_raw = float(np.mean((p_raw - y_true) ** 2))
    brier_cal = float(np.mean((p_cal - y_true) ** 2))
    print(f"  Brier raw:        {brier_raw:.6f}")
    print(f"  Brier calibrated: {brier_cal:.6f}")
    if brier_raw > 0:
        print(f"  Improvement:      {(brier_raw - brier_cal) / brier_raw * 100:.2f}%")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    calibrator.save(args.out_dir)
    print(f"\nSaved calibrator to: {args.out_dir}/")
    print(f"  - meta.json")
    if calibrator.early_cal and calibrator.early_cal.is_fitted:
        print(f"  - early.json")
    if calibrator.final60_cal and calibrator.final60_cal.is_fitted:
        print(f"  - final60.json")


if __name__ == "__main__":
    main()
