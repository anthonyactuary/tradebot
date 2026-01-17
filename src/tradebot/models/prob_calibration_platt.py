"""Platt scaling calibrator for probability outputs.

Platt scaling fits a logistic regression on log-odds to map raw probabilities
to calibrated probabilities. This is effective when the model's probability
estimates are monotonically related to true probabilities but systematically
biased (e.g., overconfident or underconfident).

Usage:
    from tradebot.models.prob_calibration_platt import PlattCalibrator

    cal = PlattCalibrator()
    cal.fit(p_raw_train, y_train)
    p_calibrated = cal.predict(p_raw_test)
    cal.save_json("calibration/platt.json")

    # Later:
    cal2 = PlattCalibrator.load_json("calibration/platt.json")
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
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


def _require_sklearn_logistic() -> Any:
    try:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression
    except ImportError as e:
        raise RuntimeError(
            "scikit-learn is required for Platt calibration. "
            "Install with: pip install scikit-learn"
        ) from e


def _clip_prob(p: "np.ndarray", eps: float = 1e-6) -> "np.ndarray":
    """Clip probabilities to [eps, 1-eps] to avoid log(0)."""
    np = _require_numpy()
    return np.clip(p, eps, 1.0 - eps)


def _logit(p: "np.ndarray") -> "np.ndarray":
    """Compute log-odds: log(p / (1-p))."""
    np = _require_numpy()
    p_clipped = _clip_prob(p)
    return np.log(p_clipped / (1.0 - p_clipped))


def _sigmoid(z: "np.ndarray") -> "np.ndarray":
    """Stable sigmoid function."""
    np = _require_numpy()
    # Use stable computation to avoid overflow
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z))
    )


@dataclass
class PlattCalibrator:
    """Platt scaling calibrator.

    Maps raw probabilities to calibrated probabilities using:
        p_cal = sigmoid(a * logit(p_raw) + b)

    where a and b are fitted via logistic regression.
    """

    a: float | None = None  # slope
    b: float | None = None  # intercept
    _fitted: bool = False

    def fit(self, p_raw: "np.ndarray", y: "np.ndarray") -> None:
        """Fit Platt calibrator on raw probabilities and true labels.

        Args:
            p_raw: Raw probability predictions, shape (n_samples,)
            y: True binary labels (0 or 1), shape (n_samples,)
        """
        np = _require_numpy()
        LogisticRegression = _require_sklearn_logistic()

        p_raw = np.asarray(p_raw).ravel()
        y = np.asarray(y).ravel()

        if len(p_raw) != len(y):
            raise ValueError(
                f"p_raw and y must have same length, got {len(p_raw)} vs {len(y)}"
            )
        if len(p_raw) < 2:
            raise ValueError("Need at least 2 samples to fit calibrator")

        # Compute log-odds (with clipping for numerical stability)
        z = _logit(p_raw)

        # Fit logistic regression: P(y=1|z) = sigmoid(a*z + b)
        # sklearn's LogisticRegression fits: P(y=1) = sigmoid(coef * X + intercept)
        lr = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            fit_intercept=True,
            C=1e10,  # Very weak regularization (essentially none)
        )
        lr.fit(z.reshape(-1, 1), y)

        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.intercept_[0])
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

        z = _logit(p_raw)
        z_cal = self.a * z + self.b
        p_cal = _sigmoid(z_cal)

        if scalar_input:
            return float(p_cal[0])
        return p_cal

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def save_json(self, path: str) -> None:
        """Save calibrator parameters to JSON file."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted calibrator")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "type": "platt",
            "a": self.a,
            "b": self.b,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "PlattCalibrator":
        """Load calibrator from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        if data.get("type") != "platt":
            raise ValueError(f"Expected type 'platt', got {data.get('type')}")

        cal = cls()
        cal.a = float(data["a"])
        cal.b = float(data["b"])
        cal._fitted = True
        return cal

    def __repr__(self) -> str:
        if self._fitted:
            return f"PlattCalibrator(a={self.a:.4f}, b={self.b:.4f})"
        return "PlattCalibrator(unfitted)"
