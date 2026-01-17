"""Time-bucketed probability calibrator.

This module provides calibration that varies based on time-to-expiry (TTE).
The intuition is that model behavior may differ significantly when:
- Early: plenty of time for price to move, uncertainty is higher
- Final 60s: price is nearly locked in, model confidence should be different

Two separate calibrators are maintained and applied based on TTE.

Usage:
    from tradebot.models.prob_calibration_bucketed import BucketedCalibrator

    cal = BucketedCalibrator(method="platt")
    cal.fit(p_raw, y, tte_seconds)
    p_calibrated = cal.predict(p_raw_new, tte_seconds_new)
    cal.save("calibration/bucketed")

    # Later:
    cal2 = BucketedCalibrator.load("calibration/bucketed")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np

from tradebot.models.prob_calibration_platt import PlattCalibrator
from tradebot.models.prob_calibration_isotonic import IsotonicCalibrator


def _require_numpy() -> Any:
    try:
        import numpy as np
        return np
    except ImportError as e:
        raise RuntimeError(
            "numpy is required for calibration. Install with: pip install numpy"
        ) from e


CalibrationMethod = Literal["platt", "isotonic"]

# Threshold in seconds: <= this is "final60", > is "early"
FINAL_BUCKET_THRESHOLD_SECONDS = 60


def bucket_fn(tte_seconds: float | int) -> Literal["early", "final60"]:
    """Determine calibration bucket based on time-to-expiry.

    Args:
        tte_seconds: Time to expiry in seconds

    Returns:
        "final60" if tte <= 60 seconds, else "early"
    """
    if tte_seconds <= FINAL_BUCKET_THRESHOLD_SECONDS:
        return "final60"
    return "early"


def _create_calibrator(method: CalibrationMethod) -> PlattCalibrator | IsotonicCalibrator:
    """Factory to create calibrator by method name."""
    if method == "platt":
        return PlattCalibrator()
    elif method == "isotonic":
        return IsotonicCalibrator()
    else:
        raise ValueError(f"Unknown calibration method: {method}")


@dataclass
class BucketedCalibrator:
    """Time-bucketed calibrator with separate models for early vs final60.

    Attributes:
        method: Calibration method ("platt" or "isotonic")
        early_cal: Calibrator for TTE > 60 seconds
        final60_cal: Calibrator for TTE <= 60 seconds
    """

    method: CalibrationMethod = "platt"
    early_cal: PlattCalibrator | IsotonicCalibrator | None = None
    final60_cal: PlattCalibrator | IsotonicCalibrator | None = None

    def __post_init__(self) -> None:
        if self.early_cal is None:
            self.early_cal = _create_calibrator(self.method)
        if self.final60_cal is None:
            self.final60_cal = _create_calibrator(self.method)

    def fit(
        self,
        p_raw: "np.ndarray",
        y: "np.ndarray",
        tte_seconds: "np.ndarray",
        *,
        min_samples_per_bucket: int = 10,
    ) -> dict[str, int]:
        """Fit both calibrators on appropriately bucketed data.

        Args:
            p_raw: Raw probability predictions, shape (n_samples,)
            y: True binary labels (0 or 1), shape (n_samples,)
            tte_seconds: Time to expiry in seconds, shape (n_samples,)
            min_samples_per_bucket: Minimum samples required to fit each bucket

        Returns:
            Dict with counts: {"early": n_early, "final60": n_final60}
        """
        np = _require_numpy()

        p_raw = np.asarray(p_raw).ravel()
        y = np.asarray(y).ravel()
        tte_seconds = np.asarray(tte_seconds).ravel()

        if not (len(p_raw) == len(y) == len(tte_seconds)):
            raise ValueError(
                f"Arrays must have same length: p_raw={len(p_raw)}, "
                f"y={len(y)}, tte_seconds={len(tte_seconds)}"
            )

        # Split into buckets
        mask_early = tte_seconds > FINAL_BUCKET_THRESHOLD_SECONDS
        mask_final60 = ~mask_early

        n_early = int(mask_early.sum())
        n_final60 = int(mask_final60.sum())

        # Fit early calibrator
        if n_early >= min_samples_per_bucket:
            self.early_cal = _create_calibrator(self.method)
            self.early_cal.fit(p_raw[mask_early], y[mask_early])
        else:
            self.early_cal = None

        # Fit final60 calibrator
        if n_final60 >= min_samples_per_bucket:
            self.final60_cal = _create_calibrator(self.method)
            self.final60_cal.fit(p_raw[mask_final60], y[mask_final60])
        else:
            self.final60_cal = None

        return {"early": n_early, "final60": n_final60}

    def predict(
        self,
        p_raw: "np.ndarray",
        tte_seconds: "np.ndarray | float | int",
    ) -> "np.ndarray":
        """Apply calibration based on TTE bucket.

        If a bucket's calibrator is not fitted, returns p_raw unchanged for
        that bucket.

        Args:
            p_raw: Raw probability predictions
            tte_seconds: Time to expiry in seconds (scalar or array)

        Returns:
            Calibrated probabilities
        """
        np = _require_numpy()

        p_raw = np.asarray(p_raw)
        scalar_input = p_raw.ndim == 0
        p_raw = np.atleast_1d(p_raw)

        tte_seconds = np.asarray(tte_seconds)
        tte_seconds = np.atleast_1d(tte_seconds)

        # Broadcast if tte_seconds is scalar
        if len(tte_seconds) == 1 and len(p_raw) > 1:
            tte_seconds = np.full(len(p_raw), tte_seconds[0])

        if len(p_raw) != len(tte_seconds):
            raise ValueError(
                f"p_raw and tte_seconds must have same length or tte_seconds "
                f"must be scalar: {len(p_raw)} vs {len(tte_seconds)}"
            )

        # Start with raw probabilities
        p_cal = p_raw.copy()

        # Apply early calibrator where applicable
        mask_early = tte_seconds > FINAL_BUCKET_THRESHOLD_SECONDS
        if mask_early.any() and self.early_cal is not None and self.early_cal.is_fitted:
            p_cal[mask_early] = self.early_cal.predict(p_raw[mask_early])

        # Apply final60 calibrator where applicable
        mask_final60 = ~mask_early
        if mask_final60.any() and self.final60_cal is not None and self.final60_cal.is_fitted:
            p_cal[mask_final60] = self.final60_cal.predict(p_raw[mask_final60])

        if scalar_input:
            return float(p_cal[0])
        return p_cal

    def predict_single(self, p_raw: float, tte_seconds: float | int) -> float:
        """Convenience method for single prediction."""
        np = _require_numpy()
        result = self.predict(np.array([p_raw]), np.array([tte_seconds]))
        return float(result[0])

    @property
    def is_fitted(self) -> bool:
        """True if at least one bucket calibrator is fitted."""
        early_fitted = self.early_cal is not None and self.early_cal.is_fitted
        final60_fitted = self.final60_cal is not None and self.final60_cal.is_fitted
        return early_fitted or final60_fitted

    def save(self, directory: str) -> None:
        """Save calibrators to a directory.

        Creates:
            directory/meta.json - method and bucket info
            directory/early.json - early calibrator (if fitted)
            directory/final60.json - final60 calibrator (if fitted)
        """
        os.makedirs(directory, exist_ok=True)

        meta = {
            "type": "bucketed",
            "method": self.method,
            "threshold_seconds": FINAL_BUCKET_THRESHOLD_SECONDS,
            "early_fitted": self.early_cal is not None and self.early_cal.is_fitted,
            "final60_fitted": self.final60_cal is not None and self.final60_cal.is_fitted,
        }
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        if self.early_cal is not None and self.early_cal.is_fitted:
            self.early_cal.save_json(os.path.join(directory, "early.json"))

        if self.final60_cal is not None and self.final60_cal.is_fitted:
            self.final60_cal.save_json(os.path.join(directory, "final60.json"))

    @classmethod
    def load(cls, directory: str) -> "BucketedCalibrator":
        """Load calibrators from a directory."""
        meta_path = os.path.join(directory, "meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        if meta.get("type") != "bucketed":
            raise ValueError(f"Expected type 'bucketed', got {meta.get('type')}")

        method = meta.get("method", "platt")
        cal = cls(method=method)

        # Load early calibrator if it exists
        early_path = os.path.join(directory, "early.json")
        if meta.get("early_fitted") and os.path.exists(early_path):
            if method == "platt":
                cal.early_cal = PlattCalibrator.load_json(early_path)
            else:
                cal.early_cal = IsotonicCalibrator.load_json(early_path)
        else:
            cal.early_cal = None

        # Load final60 calibrator if it exists
        final60_path = os.path.join(directory, "final60.json")
        if meta.get("final60_fitted") and os.path.exists(final60_path):
            if method == "platt":
                cal.final60_cal = PlattCalibrator.load_json(final60_path)
            else:
                cal.final60_cal = IsotonicCalibrator.load_json(final60_path)
        else:
            cal.final60_cal = None

        return cal

    def __repr__(self) -> str:
        early_str = repr(self.early_cal) if self.early_cal else "None"
        final60_str = repr(self.final60_cal) if self.final60_cal else "None"
        return (
            f"BucketedCalibrator(method={self.method!r}, "
            f"early={early_str}, final60={final60_str})"
        )
