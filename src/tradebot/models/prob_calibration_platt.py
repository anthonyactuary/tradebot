"""Platt scaling calibrator for probability outputs.

Platt scaling fits a logistic regression on log-odds to map raw probabilities
to calibrated probabilities. This is effective when the model's probability
estimates are monotonically related to true probabilities but systematically
biased (e.g., overconfident or underconfident).

Usage:
    # Single calibrator
    from tradebot.models.prob_calibration_platt import PlattCalibrator
    cal = PlattCalibrator()
    cal.fit(p_raw_train, y_train)
    p_calibrated = cal.predict(p_raw_test)

    # Multi-TTE calibrator (for polling)
    from tradebot.models.prob_calibration_platt import MultiTTECalibrator
    
    # Train from CSV files
    multi_cal = MultiTTECalibrator.train_from_csvs("calibration_training/")
    multi_cal.save("calibration/platt_multi_tte.json")
    
    # Runtime usage
    multi_cal = MultiTTECalibrator.load("calibration/platt_multi_tte.json")
    p_cal = multi_cal.predict(p_raw=0.65, tte=450)  # Uses nearest TTE bucket
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def _require_numpy() -> Any:
    try:
        import numpy as np
        return np
    except ImportError as e:
        raise RuntimeError("numpy required: pip install numpy") from e


def _require_sklearn_logistic() -> Any:
    try:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression
    except ImportError as e:
        raise RuntimeError("scikit-learn required: pip install scikit-learn") from e


def _clip_prob(p: "np.ndarray", eps: float = 1e-6) -> "np.ndarray":
    np = _require_numpy()
    return np.clip(p, eps, 1.0 - eps)


def _logit(p: "np.ndarray") -> "np.ndarray":
    np = _require_numpy()
    p_clipped = _clip_prob(p)
    return np.log(p_clipped / (1.0 - p_clipped))


def _sigmoid(z: "np.ndarray") -> "np.ndarray":
    np = _require_numpy()
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


@dataclass
class PlattCalibrator:
    """Single Platt scaling calibrator: p_cal = sigmoid(a * logit(p_raw) + b)"""

    a: float | None = None
    b: float | None = None
    _fitted: bool = False

    def fit(self, p_raw: "np.ndarray", y: "np.ndarray") -> None:
        np = _require_numpy()
        LogisticRegression = _require_sklearn_logistic()

        p_raw, y = np.asarray(p_raw).ravel(), np.asarray(y).ravel()
        if len(p_raw) != len(y) or len(p_raw) < 2:
            raise ValueError("Need matching arrays with at least 2 samples")

        z = _logit(p_raw)
        lr = LogisticRegression(solver="lbfgs", max_iter=1000, C=1e10)
        lr.fit(z.reshape(-1, 1), y)

        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        self._fitted = True

    def predict(self, p_raw: "np.ndarray") -> "np.ndarray":
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted")
        np = _require_numpy()
        p_raw = np.atleast_1d(np.asarray(p_raw))
        return _sigmoid(self.a * _logit(p_raw) + self.b)

    def predict_single(self, p_raw: float) -> float:
        """Fast path for single probability."""
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted")
        import math
        p = max(1e-6, min(1 - 1e-6, p_raw))
        z = math.log(p / (1 - p))
        z_cal = self.a * z + self.b
        return 1.0 / (1.0 + math.exp(-z_cal)) if z_cal >= 0 else math.exp(z_cal) / (1.0 + math.exp(z_cal))

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def save_json(self, path: str) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted calibrator")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"type": "platt", "a": self.a, "b": self.b}, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "PlattCalibrator":
        with open(path) as f:
            data = json.load(f)
        if data.get("type") != "platt":
            raise ValueError(f"Expected type 'platt', got {data.get('type')}")
        cal = cls(a=float(data["a"]), b=float(data["b"]), _fitted=True)
        return cal

    def __repr__(self) -> str:
        return f"PlattCalibrator(a={self.a:.4f}, b={self.b:.4f})" if self._fitted else "PlattCalibrator(unfitted)"


@dataclass
class MultiTTECalibrator:
    """Multi-TTE Platt calibrator bundle for polling scenarios.
    
    Stores (a, b) parameters for each TTE bucket and provides fast lookup
    to calibrate probabilities at any TTE between 60-720s.
    """
    
    # TTE bucket max -> (a, b) tuple
    calibrators: dict[int, tuple[float, float]] = field(default_factory=dict)
    # Sorted bucket boundaries for fast lookup
    _buckets: list[int] = field(default_factory=list)
    
    @classmethod
    def train_from_csvs(cls, csv_dir: str | Path, verbose: bool = True) -> "MultiTTECalibrator":
        """Train calibrators from all CSV files in directory.
        
        Expects files named: platt_training_data_{max}_{min}_tte.csv
        Each CSV should have columns: ticker, p_yes, y
        """
        import pandas as pd
        
        csv_dir = Path(csv_dir)
        calibrators = {}
        
        for f in sorted(csv_dir.glob("platt_training_data_*_tte.csv")):
            # Parse TTE range from filename
            parts = f.stem.replace("platt_training_data_", "").replace("_tte", "").split("_")
            tte_max = int(parts[0])
            
            # Load and train
            df = pd.read_csv(f)
            cal = PlattCalibrator()
            cal.fit(df["p_yes"].values, df["y"].values)
            calibrators[tte_max] = (cal.a, cal.b)
            
            if verbose:
                print(f"  TTE {tte_max:>3}s: a={cal.a:+.4f}, b={cal.b:+.4f}")
        
        if verbose:
            print(f"\nTrained {len(calibrators)} TTE calibrators")
        
        instance = cls(calibrators=calibrators)
        instance._buckets = sorted(calibrators.keys())
        return instance
    
    def predict(self, p_raw: float, tte: float) -> float:
        """Calibrate probability using nearest TTE bucket.
        
        Args:
            p_raw: Raw probability (0-1)
            tte: Time to expiry in seconds
            
        Returns:
            Calibrated probability
        """
        import math
        
        # Find nearest bucket (buckets are TTE max values: 720, 700, 680, ...)
        # TTE 450 should use bucket 460 (covers 460-440)
        bucket = self._find_bucket(tte)
        a, b = self.calibrators[bucket]
        
        # Fast single-value calibration
        p = max(1e-6, min(1 - 1e-6, p_raw))
        z = math.log(p / (1 - p))
        z_cal = a * z + b
        return 1.0 / (1.0 + math.exp(-z_cal)) if z_cal >= 0 else math.exp(z_cal) / (1.0 + math.exp(z_cal))
    
    def _find_bucket(self, tte: float) -> int:
        """Find the TTE bucket that contains this TTE value."""
        # Buckets are max values covering (max-20, max] ranges
        # e.g., bucket 720 covers (700, 720], bucket 700 covers (680, 700]
        # For TTE=450, we want bucket 460 which covers (440, 460]
        
        if not self._buckets:
            raise RuntimeError("No calibrators loaded")
        
        # Clamp to valid range
        tte = max(self._buckets[0] - 19, min(self._buckets[-1], tte))
        
        # Find smallest bucket >= tte (bucket whose range contains tte)
        # Since bucket N covers (N-20, N], we want smallest bucket where bucket >= tte
        for bucket in self._buckets:
            if tte <= bucket:
                return bucket
        return self._buckets[-1]
    
    def save(self, path: str | Path) -> None:
        """Save all calibrators to single JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "type": "platt_multi_tte",
            "calibrators": {str(k): {"a": v[0], "b": v[1]} for k, v in self.calibrators.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> "MultiTTECalibrator":
        """Load calibrators from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        if data.get("type") != "platt_multi_tte":
            raise ValueError(f"Expected type 'platt_multi_tte', got {data.get('type')}")
        
        calibrators = {int(k): (v["a"], v["b"]) for k, v in data["calibrators"].items()}
        instance = cls(calibrators=calibrators)
        instance._buckets = sorted(calibrators.keys())
        return instance
    
    def __repr__(self) -> str:
        if self._buckets:
            return f"MultiTTECalibrator({len(self.calibrators)} buckets: {self._buckets[0]}-{self._buckets[-1]}s)"
        return "MultiTTECalibrator(empty)"


# ============================================================
# CLI: Train and save multi-TTE calibrator
# ============================================================
if __name__ == "__main__":
    CSV_DIR = Path(__file__).parent / "calibration_training"
    OUTPUT_FILE = Path(__file__).parent / "platt_multi_tte.json"
    
    print("=" * 60)
    print("TRAINING MULTI-TTE PLATT CALIBRATORS")
    print("=" * 60)
    print(f"CSV directory: {CSV_DIR}")
    print()
    
    multi_cal = MultiTTECalibrator.train_from_csvs(CSV_DIR)
    multi_cal.save(OUTPUT_FILE)
    
    print(f"\nSaved to: {OUTPUT_FILE}")
    
    # Quick test
    print("\n" + "-" * 60)
    print("TEST: Calibrating p=0.7 at various TTEs")
    print("-" * 60)
    for tte in [720, 600, 400, 200, 100, 60]:
        p_cal = multi_cal.predict(0.7, tte)
        print(f"  TTE={tte:>3}s: 0.70 → {p_cal:.3f}")
