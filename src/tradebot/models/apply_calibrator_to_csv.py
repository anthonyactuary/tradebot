"""Apply fitted calibrators to a CSV and output calibrated probabilities.

This script reads a CSV containing raw predictions, applies a fitted
bucketed calibrator, and outputs a new CSV with calibrated probabilities
along with diagnostic statistics.

Expected input CSV columns:
    - p_raw_up: Raw probability of price going up (required)
    - tte_seconds: Time to expiry in seconds (required)
    - y_true: True outcome (optional, for diagnostics)

Output CSV adds columns:
    - p_cal_up: Calibrated probability
    - p_cal_bucket: Which calibrator was used ("early" or "final60")

Usage:
    python -m tradebot.models.apply_calibrator_to_csv \
        --input predictions.csv \
        --calibrator-dir calibration/ \
        --output predictions_calibrated.csv

    # Just print diagnostics without output:
    python -m tradebot.models.apply_calibrator_to_csv \
        --input predictions.csv \
        --calibrator-dir calibration/ \
        --diagnostics-only
"""

from __future__ import annotations

import argparse
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


def _compute_brier(p: "Any", y: "Any") -> float:
    """Compute Brier score."""
    np, _ = _require_deps()
    return float(np.mean((p - y) ** 2))


def _print_reliability_table(
    label: str,
    p_raw: "Any",
    p_cal: "Any",
    y: "Any",
    n_bins: int = 10,
) -> None:
    """Print reliability comparison table."""
    np, _ = _require_deps()

    print(f"\n{label} Reliability Table ({n_bins} bins):")
    print(f"{'Bin':>12} {'Count':>6} {'AvgRaw':>8} {'AvgCal':>8} {'Empirical':>10} {'ErrRaw':>8} {'ErrCal':>8}")
    print(f"{'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right edge only for last bin
        if i < n_bins - 1:
            mask = (p_raw >= lo) & (p_raw < hi)
        else:
            mask = (p_raw >= lo) & (p_raw <= hi)

        count = int(mask.sum())
        if count > 0:
            avg_raw = float(np.mean(p_raw[mask]))
            avg_cal = float(np.mean(p_cal[mask]))
            empirical = float(np.mean(y[mask]))
            err_raw = abs(avg_raw - empirical)
            err_cal = abs(avg_cal - empirical)
            bin_label = f"[{lo:.1f}, {hi:.1f})"
            print(f"{bin_label:>12} {count:>6} {avg_raw:>8.4f} {avg_cal:>8.4f} {empirical:>10.4f} {err_raw:>8.4f} {err_cal:>8.4f}")
        else:
            bin_label = f"[{lo:.1f}, {hi:.1f})"
            print(f"{bin_label:>12} {count:>6} {'--':>8} {'--':>8} {'--':>10} {'--':>8} {'--':>8}")


def _print_diagnostics(
    p_raw: "Any",
    p_cal: "Any",
    y: "Any",
    buckets: "Any",
    tte_seconds: "Any",
) -> None:
    """Print comprehensive calibration diagnostics."""
    np, _ = _require_deps()

    from tradebot.models.prob_calibration_bucketed import FINAL_BUCKET_THRESHOLD_SECONDS

    print("\n" + "=" * 60)
    print("CALIBRATION DIAGNOSTICS")
    print("=" * 60)

    # Overall stats
    n = len(p_raw)
    print(f"\nOverall ({n} samples):")
    print(f"  Base rate (mean y_true): {np.mean(y):.4f}")
    print(f"  Mean p_raw:              {np.mean(p_raw):.4f}")
    print(f"  Mean p_cal:              {np.mean(p_cal):.4f}")

    brier_raw = _compute_brier(p_raw, y)
    brier_cal = _compute_brier(p_cal, y)
    print(f"\n  Brier score (raw):       {brier_raw:.6f}")
    print(f"  Brier score (cal):       {brier_cal:.6f}")
    if brier_raw > 0:
        improvement = (brier_raw - brier_cal) / brier_raw * 100
        direction = "improvement" if improvement > 0 else "degradation"
        print(f"  Change:                  {abs(improvement):.2f}% {direction}")

    # By bucket
    mask_early = tte_seconds > FINAL_BUCKET_THRESHOLD_SECONDS
    mask_final60 = ~mask_early

    for name, mask in [("Early (tte > 60s)", mask_early), ("Final60 (tte <= 60s)", mask_final60)]:
        count = int(mask.sum())
        if count == 0:
            print(f"\n{name}: No samples")
            continue

        print(f"\n{name} ({count} samples):")
        print(f"  Base rate: {np.mean(y[mask]):.4f}")
        print(f"  Mean p_raw: {np.mean(p_raw[mask]):.4f}")
        print(f"  Mean p_cal: {np.mean(p_cal[mask]):.4f}")

        b_raw = _compute_brier(p_raw[mask], y[mask])
        b_cal = _compute_brier(p_cal[mask], y[mask])
        print(f"  Brier raw: {b_raw:.6f}")
        print(f"  Brier cal: {b_cal:.6f}")
        if b_raw > 0:
            imp = (b_raw - b_cal) / b_raw * 100
            print(f"  Change: {imp:.2f}%")

    # Reliability tables
    _print_reliability_table("Overall", p_raw, p_cal, y)

    if mask_early.sum() >= 10:
        _print_reliability_table("Early", p_raw[mask_early], p_cal[mask_early], y[mask_early])
    if mask_final60.sum() >= 10:
        _print_reliability_table("Final60", p_raw[mask_final60], p_cal[mask_final60], y[mask_final60])

    print("\n" + "=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply calibrators to CSV and output calibrated probabilities"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input CSV with columns: p_raw_up, tte_seconds, [y_true]",
    )
    parser.add_argument(
        "--calibrator-dir", "-c",
        type=str,
        required=True,
        help="Directory containing fitted calibrator (meta.json, early.json, final60.json)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output CSV path (adds p_cal_up, p_cal_bucket columns)",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Only print diagnostics, don't write output CSV",
    )
    args = parser.parse_args()

    np, pd = _require_deps()

    from tradebot.models.prob_calibration_bucketed import BucketedCalibrator, bucket_fn

    # Load calibrator
    print(f"Loading calibrator from: {args.calibrator_dir}")
    try:
        calibrator = BucketedCalibrator.load(args.calibrator_dir)
    except Exception as e:
        print(f"ERROR: Failed to load calibrator: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Calibrator: {calibrator}")

    # Load data
    print(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input)

    required_cols = ["p_raw_up", "tte_seconds"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    p_raw = df["p_raw_up"].values.astype(float)
    tte_seconds = df["tte_seconds"].values.astype(float)

    print(f"Loaded {len(df)} samples")

    # Apply calibration
    p_cal = calibrator.predict(p_raw, tte_seconds)
    buckets = np.array([bucket_fn(t) for t in tte_seconds])

    # Add to dataframe
    df["p_cal_up"] = p_cal
    df["p_cal_bucket"] = buckets

    # Diagnostics (if y_true is available)
    has_labels = "y_true" in df.columns
    if has_labels:
        y_true = df["y_true"].values.astype(int)
        _print_diagnostics(p_raw, p_cal, y_true, buckets, tte_seconds)
    else:
        print("\nNote: y_true column not found, skipping diagnostics")
        print(f"\nCalibration summary:")
        print(f"  Input samples: {len(df)}")
        print(f"  Mean p_raw: {np.mean(p_raw):.4f}")
        print(f"  Mean p_cal: {np.mean(p_cal):.4f}")
        print(f"  Early samples: {(buckets == 'early').sum()}")
        print(f"  Final60 samples: {(buckets == 'final60').sum()}")

    # Write output
    if not args.diagnostics_only:
        if args.output is None:
            # Default: add _calibrated suffix
            base = args.input.rsplit(".", 1)
            if len(base) == 2:
                output_path = f"{base[0]}_calibrated.{base[1]}"
            else:
                output_path = f"{args.input}_calibrated"
        else:
            output_path = args.output

        df.to_csv(output_path, index=False)
        print(f"\nWrote calibrated output to: {output_path}")
        print(f"  New columns: p_cal_up, p_cal_bucket")
    else:
        print("\n(--diagnostics-only: no output CSV written)")


if __name__ == "__main__":
    main()
