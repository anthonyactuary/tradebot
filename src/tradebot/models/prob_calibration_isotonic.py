"""Isotonic regression calibrator for probability outputs.

Isotonic calibration fits a monotonically increasing step function to map
raw probabilities to calibrated probabilities. This is more flexible than
Platt scaling and doesn't assume a particular parametric form.

Usage:
    from tradebot.models.prob_calibration_isotonic import IsotonicCalibrator

    cal = IsotonicCalibrator()
    cal.fit(p_raw_train, y_train)
    p_calibrated = cal.predict(p_raw_test)
    cal.save_json("calibration/isotonic.json")

    # Later:
    cal2 = IsotonicCalibrator.load_json("calibration/isotonic.json")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def _require_numpy() -> Any:
    try:
        import numpy as np
        return np
    except ImportError as e:
        raise RuntimeError(
            "numpy is required for calibration. Install with: pip install numpy"
        ) from e


def _require_sklearn_isotonic() -> Any:
    try:
        from sklearn.isotonic import IsotonicRegression
        return IsotonicRegression
    except ImportError as e:
        raise RuntimeError(
            "scikit-learn is required for isotonic calibration. "
            "Install with: pip install scikit-learn"
        ) from e


@dataclass
class IsotonicCalibrator:
    """Isotonic regression calibrator.

    Maps raw probabilities to calibrated probabilities using a monotonically
    increasing piecewise-constant function learned from data.
    """

    # Store knot points for JSON serialization
    _x_knots: list[float] = field(default_factory=list)
    _y_knots: list[float] = field(default_factory=list)
    _fitted: bool = False

    def fit(self, p_raw: "np.ndarray", y: "np.ndarray") -> None:
        """Fit isotonic calibrator on raw probabilities and true labels.

        Args:
            p_raw: Raw probability predictions, shape (n_samples,)
            y: True binary labels (0 or 1), shape (n_samples,)
        """
        np = _require_numpy()
        IsotonicRegression = _require_sklearn_isotonic()

        p_raw = np.asarray(p_raw).ravel()
        y = np.asarray(y).ravel()

        if len(p_raw) != len(y):
            raise ValueError(
                f"p_raw and y must have same length, got {len(p_raw)} vs {len(y)}"
            )
        if len(p_raw) < 2:
            raise ValueError("Need at least 2 samples to fit calibrator")

        # Fit isotonic regression
        ir = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
            increasing=True,
        )
        ir.fit(p_raw, y)

        # Extract knot points for serialization
        # IsotonicRegression stores X_thresholds_ and y_thresholds_ after fit
        if hasattr(ir, "X_thresholds_") and hasattr(ir, "y_thresholds_"):
            self._x_knots = ir.X_thresholds_.tolist()
            self._y_knots = ir.y_thresholds_.tolist()
        else:
            # Fallback: use unique sorted values
            sorted_idx = np.argsort(p_raw)
            self._x_knots = p_raw[sorted_idx].tolist()
            self._y_knots = ir.predict(p_raw[sorted_idx]).tolist()

        self._fitted = True

    def predict(self, p_raw: "np.ndarray") -> "np.ndarray":
        """Apply calibration to raw probabilities.

        Args:
            p_raw: Raw probability predictions, shape (n_samples,) or scalar

        Returns:
            Calibrated probabilities, same shape as input
        """
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        np = _require_numpy()
        p_raw = np.asarray(p_raw)
        scalar_input = p_raw.ndim == 0
        p_raw = np.atleast_1d(p_raw)

        # Use numpy interpolation with clipping at boundaries
        x_knots = np.array(self._x_knots)
        y_knots = np.array(self._y_knots)

        # np.interp handles out-of-bounds by using first/last value
        p_cal = np.interp(p_raw, x_knots, y_knots)

        # Ensure output is in [0, 1]
        p_cal = np.clip(p_cal, 0.0, 1.0)

        if scalar_input:
            return float(p_cal[0])
        return p_cal

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def n_knots(self) -> int:
        """Number of knot points in the isotonic function."""
        return len(self._x_knots)

    def save_json(self, path: str) -> None:
        """Save calibrator parameters to JSON file."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted calibrator")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "type": "isotonic",
            "x_knots": self._x_knots,
            "y_knots": self._y_knots,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "IsotonicCalibrator":
        """Load calibrator from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        if data.get("type") != "isotonic":
            raise ValueError(f"Expected type 'isotonic', got {data.get('type')}")

        cal = cls()
        cal._x_knots = [float(x) for x in data["x_knots"]]
        cal._y_knots = [float(y) for y in data["y_knots"]]
        cal._fitted = True
        return cal

    def __repr__(self) -> str:
        if self._fitted:
            return f"IsotonicCalibrator(n_knots={self.n_knots})"
        return "IsotonicCalibrator(unfitted)"
