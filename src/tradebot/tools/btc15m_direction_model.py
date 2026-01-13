"""Training pipeline: predict 15-minute BTC direction (binary).

Goal
- Estimate P(BTC close at end of 15-minute window > close at start).

Dataset construction (from 1-minute candles)
- For each rolling 15-minute window:
  - Strike K = close at window start (index i)
  - Choose a random observation minute m in [1..14]
  - Spot S_t = close at index i+m
  - time_remaining = 15 - m (minutes)
  - Label y = 1 if close at window end (index i+14) > K else 0

Features at observation time (index t = i+m)
- delta = (S_t - K) / K
- abs_delta
- return_1m, return_3m, return_5m (based on close returns)
- vol_5m = std dev of last 5 one-minute returns
- trend_5m = slope of linear regression over last 5 closes
- time_remaining (minutes)

Model
- XGBoost (preferred) or LightGBM binary classifier
- Time-based split (first 80% train, last 20% validation)
- Optimize logloss with early stopping

Outputs
- Metrics: accuracy, ROC-AUC, logloss
- Calibration data: predicted probability bins vs empirical frequency
- Saved model + metadata to disk

Usage
  # Train from CSV (timestamp, open, high, low, close, volume)
    python -m tradebot.tools.btc15m_direction_model \
    --input-csv data/btc_1m.csv --out-dir models/btc15m

    # Optional: download Coinbase 1m data (can be slow/rate-limited; may cap history)
    python -m tradebot.tools.btc15m_direction_model \
    --download-coinbase --days 180 --out-dir models/btc15m
"""

from __future__ import annotations

import asyncio
import argparse
import datetime as dt
import json
import math
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import httpx

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import pandas as pd


COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"


@dataclass(frozen=True)
class DatasetBundle:
    X: "np.ndarray"
    y: "np.ndarray"
    feature_names: list[str]
    sample_timestamps: "np.ndarray"  # window start timestamps (unix seconds)


def _require_numpy_pandas_sklearn() -> tuple[Any, Any, Any]:
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        from sklearn import metrics  # type: ignore

        return np, pd, metrics
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing required packages. Install with: pip install numpy pandas scikit-learn"
        ) from e


def _checkpoint_write_csv(rows: list[dict[str, float]], out_csv: str) -> None:
    """Write an (append-safe) checkpoint CSV from in-memory rows."""
    _np, pd, _metrics = _require_numpy_pandas_sklearn()
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)


def _try_import_booster() -> tuple[Literal["xgboost", "lightgbm"], Any]:
    """Return (backend, module) preferring xgboost."""
    try:
        import xgboost as xgb  # type: ignore

        return "xgboost", xgb
    except Exception:
        try:
            import lightgbm as lgb  # type: ignore

            return "lightgbm", lgb
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Install a boosting library: pip install xgboost  (or: pip install lightgbm)"
            ) from e


def load_candles_csv(path: str) -> "pd.DataFrame":
    """Load 1-minute candles from CSV.

    Expected columns: timestamp, open, high, low, close, volume

    timestamp can be:
    - unix seconds
    - unix milliseconds
    - ISO 8601 string
    """
    np, pd, _metrics = _require_numpy_pandas_sklearn()

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    ts = df["timestamp"]
    if np.issubdtype(ts.dtype, np.number):
        ts_f = ts.astype(float)
        # heuristic: treat large values as ms
        ts_s = np.where(ts_f > 1e12, ts_f / 1000.0, ts_f)
        df["timestamp"] = ts_s.astype(int)
    else:
        parsed = pd.to_datetime(ts, utc=True, errors="coerce")
        if parsed.isna().any():
            bad = int(parsed.isna().sum())
            raise ValueError(f"Failed to parse {bad} timestamps in CSV")
        df["timestamp"] = (parsed.view("int64") // 1_000_000_000).astype(int)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


async def download_coinbase_1m_candles(*, days: int, out_csv: str) -> str:
    """Download Coinbase BTC-USD 1m candles and write CSV.

    Coinbase candles endpoint returns up to 300 candles per request.
    This function paginates backwards from "now".

    Notes
    - This can be slow for 180 days (~864 requests at 300 candles/request).
    - Coinbase rate limits may apply.
    """
    np, pd, _metrics = _require_numpy_pandas_sklearn()

    granularity = 60
    candles_needed = int(days * 24 * 60)
    max_per_request = 300

    end_ts = int(time.time())
    rows: list[dict[str, float]] = []
    requests_made = 0
    backoff = 0.5

    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {"Accept": "application/json"}

        try:
            while len(rows) < candles_needed:
                start_ts = end_ts - max_per_request * granularity
                params = {"granularity": granularity, "start": start_ts, "end": end_ts}

                resp = await client.get(COINBASE_CANDLES_URL, params=params, headers=headers)
                if resp.status_code == 429:
                    await asyncio.sleep(backoff)
                    backoff = min(10.0, backoff * 1.5)
                    continue
                resp.raise_for_status()
                raw = resp.json()
                if not raw:
                    break

                requests_made += 1
                backoff = 0.5

                # Coinbase format: [time, low, high, open, close, volume]
                for c in raw:
                    rows.append(
                        {
                            "timestamp": float(c[0]),
                            "open": float(c[3]),
                            "high": float(c[2]),
                            "low": float(c[1]),
                            "close": float(c[4]),
                            "volume": float(c[5]),
                        }
                    )

                # next page goes earlier
                batch_min_ts = int(min(float(c[0]) for c in raw))
                end_ts = batch_min_ts

                if requests_made % 25 == 0:
                    oldest = int(min(r["timestamp"] for r in rows))
                    newest = int(max(r["timestamp"] for r in rows))
                    covered_days = (newest - oldest) / 86400.0
                    print(
                        f"download_progress: requests={requests_made} candles={len(rows)} "
                        f"covered_days~={covered_days:.1f} oldest={oldest} newest={newest}",
                        flush=True,
                    )
                    _checkpoint_write_csv(rows, out_csv)

                # be polite
                await asyncio.sleep(0.2)
        finally:
            _checkpoint_write_csv(rows, out_csv)

    # Final write already handled by checkpoint in finally.
    return out_csv


def build_training_dataset(
    candles: "pd.DataFrame",
    *,
    seed: int = 7,
    samples_per_window: int = 1,
    window_minutes: int = 15,
) -> DatasetBundle:
    """Build the supervised dataset from 1-minute candles."""
    np, pd, _metrics = _require_numpy_pandas_sklearn()

    if samples_per_window < 1:
        raise ValueError("samples_per_window must be >= 1")
    if window_minutes != 15:
        raise ValueError("Only 15-minute windows are supported right now")

    ts = candles["timestamp"].to_numpy(dtype=np.int64)
    close = candles["close"].to_numpy(dtype=float)

    n = len(candles)
    if n < 60:
        raise ValueError("Not enough candles to build a dataset")

    # Precompute contiguity (1-minute increments)
    diffs = ts[1:] - ts[:-1]
    ok_1m = diffs == 60
    # bad_prefix[k] = number of non-60s diffs in the range [0 .. k-1]
    # We allocate n+1 so callers can use i1 == n safely.
    bad_prefix = np.zeros(n + 1, dtype=int)
    if n >= 2:
        bad_prefix[1:n] = np.cumsum((~ok_1m).astype(int))
        bad_prefix[n] = int(bad_prefix[n - 1])

    def _is_contiguous(i0: int, i1: int) -> bool:
        # requires all diffs between [i0..i1-1] are 60
        return (bad_prefix[i1] - bad_prefix[i0]) == 0

    rng = np.random.default_rng(seed)

    feature_names = [
        "delta",
        "abs_delta",
        "return_1m",
        "return_3m",
        "return_5m",
        "vol_5m",
        "trend_5m",
        "time_remaining",
    ]

    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    t_rows: list[int] = []

    end_offset = window_minutes - 1  # i+14

    # We need lookback for returns/trend at observation time.
    min_obs_lookback = 5

    for start_idx in range(0, n - end_offset):
        end_idx = start_idx + end_offset

        # Window must be fully contiguous.
        # Also ensure observation lookback is contiguous.
        if not _is_contiguous(start_idx, end_idx + 1):
            continue

        K = close[start_idx]
        if not (K and math.isfinite(K) and K > 0):
            continue

        y = 1 if close[end_idx] > K else 0

        # Generate one or more random observations within the window.
        for _ in range(samples_per_window):
            m = int(rng.integers(1, window_minutes))  # 1..14
            obs_idx = start_idx + m
            if obs_idx - min_obs_lookback < 0:
                continue

            # Ensure contiguous lookback up to obs (for returns/trend/vol).
            if not _is_contiguous(obs_idx - min_obs_lookback, end_idx + 1):
                continue

            S = close[obs_idx]
            if not (S and math.isfinite(S) and S > 0):
                continue

            delta = (S - K) / K
            abs_delta = abs(delta)

            def r(k: int) -> float:
                prev = close[obs_idx - k]
                if prev <= 0:
                    return 0.0
                return (S / prev) - 1.0

            return_1m = r(1)
            return_3m = r(3)
            return_5m = r(5)

            # last 5 one-minute returns ending at obs
            rets = []
            for j in range(obs_idx - 4, obs_idx + 1):
                prev = close[j - 1]
                cur = close[j]
                if prev <= 0:
                    rets.append(0.0)
                else:
                    rets.append((cur / prev) - 1.0)
            vol_5m = float(np.std(np.asarray(rets, dtype=float), ddof=0))

            # Linear regression slope over last 5 closes.
            yv = close[obs_idx - 4 : obs_idx + 1]
            xv = np.arange(5, dtype=float)
            x_mean = float(np.mean(xv))
            y_mean = float(np.mean(yv))
            denom = float(np.sum((xv - x_mean) ** 2))
            trend_5m = 0.0
            if denom > 0:
                trend_5m = float(np.sum((xv - x_mean) * (yv - y_mean)) / denom)

            time_remaining = float(window_minutes - m)

            X_rows.append(
                [
                    float(delta),
                    float(abs_delta),
                    float(return_1m),
                    float(return_3m),
                    float(return_5m),
                    float(vol_5m),
                    float(trend_5m),
                    float(time_remaining),
                ]
            )
            y_rows.append(int(y))
            t_rows.append(int(ts[start_idx]))

    if not X_rows:
        raise ValueError("No training samples were produced (check candle continuity/format)")

    X = np.asarray(X_rows, dtype=float)
    y_arr = np.asarray(y_rows, dtype=int)
    t_arr = np.asarray(t_rows, dtype=np.int64)

    # Sort by time (important for time split)
    order = np.argsort(t_arr)
    return DatasetBundle(
        X=X[order],
        y=y_arr[order],
        feature_names=feature_names,
        sample_timestamps=t_arr[order],
    )


def time_split(bundle: DatasetBundle, *, train_frac: float = 0.8) -> tuple[DatasetBundle, DatasetBundle]:
    np, _pd, _metrics = _require_numpy_pandas_sklearn()

    if not (0.5 <= train_frac < 1.0):
        raise ValueError("train_frac must be in [0.5, 1.0)")

    n = bundle.X.shape[0]
    cut = int(round(n * train_frac))
    cut = max(1, min(n - 1, cut))

    train = DatasetBundle(
        X=bundle.X[:cut],
        y=bundle.y[:cut],
        feature_names=bundle.feature_names,
        sample_timestamps=bundle.sample_timestamps[:cut],
    )
    val = DatasetBundle(
        X=bundle.X[cut:],
        y=bundle.y[cut:],
        feature_names=bundle.feature_names,
        sample_timestamps=bundle.sample_timestamps[cut:],
    )
    return train, val


def time_split_purged(
    bundle: DatasetBundle,
    *,
    train_frac: float = 0.8,
    purge_minutes: int = 0,
) -> tuple[DatasetBundle, DatasetBundle, dict[str, Any]]:
    """Time split with an optional purge gap around the split boundary.

    This helps reduce leakage from highly-overlapping rolling windows.

    Purge is applied in timestamp space (window start timestamps).
    """
    np, _pd, _metrics = _require_numpy_pandas_sklearn()

    if purge_minutes < 0:
        raise ValueError("purge_minutes must be >= 0")

    # Base time split index.
    n = bundle.X.shape[0]
    if n < 2:
        raise ValueError("Not enough samples to split")

    cut = int(round(n * train_frac))
    cut = max(1, min(n - 1, cut))
    cut_ts = int(bundle.sample_timestamps[cut])

    purge_seconds = int(purge_minutes) * 60
    meta: dict[str, Any] = {
        "train_frac": float(train_frac),
        "purge_minutes": int(purge_minutes),
        "cut_timestamp": int(cut_ts),
    }

    if purge_seconds <= 0:
        train, val = time_split(bundle, train_frac=train_frac)
        meta["mode"] = "time_split"
        return train, val, meta

    train_mask = bundle.sample_timestamps <= (cut_ts - purge_seconds)
    val_mask = bundle.sample_timestamps >= (cut_ts + purge_seconds)

    n_train = int(np.sum(train_mask))
    n_val = int(np.sum(val_mask))
    meta.update({"n_train": int(n_train), "n_val": int(n_val)})

    # If purge would wipe out a side, fall back to plain time split.
    if n_train < 1 or n_val < 1:
        train, val = time_split(bundle, train_frac=train_frac)
        meta["mode"] = "time_split_fallback"
        meta["warning"] = "purge too large for dataset; fell back to plain time split"
        return train, val, meta

    train = DatasetBundle(
        X=bundle.X[train_mask],
        y=bundle.y[train_mask],
        feature_names=bundle.feature_names,
        sample_timestamps=bundle.sample_timestamps[train_mask],
    )
    val = DatasetBundle(
        X=bundle.X[val_mask],
        y=bundle.y[val_mask],
        feature_names=bundle.feature_names,
        sample_timestamps=bundle.sample_timestamps[val_mask],
    )
    meta["mode"] = "time_split_purged"
    return train, val, meta


def train_model(
    train: DatasetBundle,
    val: DatasetBundle,
    *,
    seed: int = 7,
) -> tuple[Any, dict[str, Any]]:
    """Train an XGBoost/LightGBM binary classifier with early stopping."""
    np, _pd, metrics = _require_numpy_pandas_sklearn()
    backend, booster = _try_import_booster()

    if backend == "xgboost":
        xgb = booster
        model = xgb.XGBClassifier(
            n_estimators=5000,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=50,
            random_state=seed,
            tree_method="hist",
        )
        model.fit(
            train.X,
            train.y,
            eval_set=[(val.X, val.y)],
            verbose=False,
        )
        meta = {
            "backend": "xgboost",
            "best_iteration": int(getattr(model, "best_iteration", -1) or -1),
        }
        return model, meta

    # LightGBM fallback
    lgb = booster
    model = lgb.LGBMClassifier(
        n_estimators=10000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary",
        random_state=seed,
    )

    model.fit(
        train.X,
        train.y,
        eval_set=[(val.X, val.y)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    meta = {
        "backend": "lightgbm",
        "best_iteration": int(getattr(model, "best_iteration_", -1) or -1),
    }
    return model, meta


def calibration_data(
    y_true: "np.ndarray",
    p_pred: "np.ndarray",
    *,
    bins: int = 10,
) -> list[dict[str, float]]:
    np, _pd, _metrics = _require_numpy_pandas_sklearn()

    bins = int(bins)
    if bins < 2:
        raise ValueError("bins must be >= 2")

    edges = np.linspace(0.0, 1.0, bins + 1)
    out: list[dict[str, float]] = []

    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == bins - 1:
            mask = (p_pred >= lo) & (p_pred <= hi)
        else:
            mask = (p_pred >= lo) & (p_pred < hi)

        n = int(np.sum(mask))
        if n == 0:
            continue
        out.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "count": float(n),
                "p_mean": float(np.mean(p_pred[mask])),
                "y_mean": float(np.mean(y_true[mask])),
            }
        )

    return out


def evaluate_model(model: Any, val: DatasetBundle, *, bins: int = 10) -> dict[str, Any]:
    np, _pd, metrics = _require_numpy_pandas_sklearn()

    p = model.predict_proba(val.X)[:, 1]
    y_hat = (p >= 0.5).astype(int)

    out: dict[str, Any] = {
        "n_val": int(val.X.shape[0]),
        "accuracy": float(metrics.accuracy_score(val.y, y_hat)),
        "roc_auc": float(metrics.roc_auc_score(val.y, p)),
        "logloss": float(metrics.log_loss(val.y, p, labels=[0, 1])),
        "calibration": calibration_data(val.y.astype(int), p.astype(float), bins=bins),
    }
    return out


def _evaluate_probabilities(y_true: Any, p_pred: Any) -> dict[str, float]:
    _np, _pd, metrics = _require_numpy_pandas_sklearn()
    y_hat = (p_pred >= 0.5).astype(int)
    return {
        "accuracy": float(metrics.accuracy_score(y_true, y_hat)),
        "roc_auc": float(metrics.roc_auc_score(y_true, p_pred)),
        "logloss": float(metrics.log_loss(y_true, p_pred, labels=[0, 1])),
    }


def baseline_metrics(train: DatasetBundle, val: DatasetBundle) -> dict[str, Any]:
    """Compute simple baselines to contextualize accuracy/AUC/logloss."""
    np, _pd, _metrics = _require_numpy_pandas_sklearn()

    yv = val.y.astype(int)
    base_rate = float(np.mean(yv))
    majority_acc = float(max(base_rate, 1.0 - base_rate))

    # Constant-probability baseline at base rate.
    p_const = np.full_like(yv, fill_value=base_rate, dtype=float)
    const = _evaluate_probabilities(yv, p_const)
    const["base_rate"] = float(base_rate)
    const["majority_accuracy"] = float(majority_acc)

    # Logistic regression on only delta + time_remaining.
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore

        # indices in feature_names
        name_to_idx = {n: i for i, n in enumerate(train.feature_names)}
        idx_delta = int(name_to_idx["delta"])
        idx_tr = int(name_to_idx["time_remaining"])

        Xtr = train.X[:, [idx_delta, idx_tr]]
        Xv = val.X[:, [idx_delta, idx_tr]]

        lr = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=0,
        )
        lr.fit(Xtr, train.y.astype(int))
        p_lr = lr.predict_proba(Xv)[:, 1]
        lr_out = _evaluate_probabilities(yv, p_lr)
    except Exception as e:
        lr_out = {"error": str(e)}

    return {
        "majority": const,
        "logreg_delta_time": lr_out,
    }


def save_artifacts(
    *,
    out_dir: str,
    model: Any,
    bundle: DatasetBundle,
    train_meta: dict[str, Any],
    metrics_out: dict[str, Any],
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "feature_names": bundle.feature_names,
        "train": train_meta,
        "metrics": metrics_out,
        "note": "Tree models do not require feature normalization.",
    }

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, sort_keys=True)

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    backend = str(train_meta.get("backend") or "")
    if backend == "xgboost":
        model.save_model(os.path.join(out_dir, "model.json"))
    elif backend == "lightgbm":
        model.booster_.save_model(os.path.join(out_dir, "model.txt"))
    else:  # pragma: no cover
        # Best-effort generic fallback
        try:
            import joblib  # type: ignore

            joblib.dump(model, os.path.join(out_dir, "model.joblib"))
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 15m BTC direction model")

    parser.add_argument("--input-csv", type=str, default=None, help="Path to 1m BTC candles CSV")
    parser.add_argument(
        "--download-coinbase",
        action="store_true",
        help="Download 1m BTC candles from Coinbase (slow for 180d)",
    )
    parser.add_argument("--days", type=int, default=180, help="Days of history to download")
    parser.add_argument(
        "--download-csv",
        type=str,
        default=None,
        help="Where to write the downloaded Coinbase 1m CSV (defaults under --out-dir)",
    )

    parser.add_argument("--out-dir", type=str, default="models/btc15m", help="Output directory")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples-per-window", type=int, default=1)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument(
        "--purge-minutes",
        type=int,
        default=0,
        help="Optional gap (minutes) to remove around the train/val boundary to reduce overlap leakage",
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run random-search hyperparameter tuning (optimizes validation logloss)",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=40,
        help="Number of random hyperparameter trials to run when --tune is set",
    )
    parser.add_argument(
        "--tune-max-estimators",
        type=int,
        default=8000,
        help="Max estimators during tuning (early stopping usually stops earlier)",
    )
    parser.add_argument(
        "--tune-early-stopping-rounds",
        type=int,
        default=50,
        help="Early stopping rounds used during tuning",
    )

    args = parser.parse_args()

    # Lazy imports to keep module import light.
    np, pd, _metrics = _require_numpy_pandas_sklearn()

    csv_path = args.input_csv
    if args.download_coinbase:
        if csv_path is None:
            csv_path = args.download_csv
        if csv_path is None:
            csv_path = os.path.join(args.out_dir, f"btc_1m_coinbase_{args.days}d.csv")
        asyncio.run(download_coinbase_1m_candles(days=int(args.days), out_csv=str(csv_path)))

    if not csv_path:
        raise SystemExit("Provide --input-csv or use --download-coinbase")

    candles = load_candles_csv(csv_path)
    bundle = build_training_dataset(
        candles,
        seed=int(args.seed),
        samples_per_window=int(args.samples_per_window),
    )
    train, val, split_meta = time_split_purged(
        bundle,
        train_frac=float(args.train_frac),
        purge_minutes=int(args.purge_minutes),
    )

    if bool(args.tune):
        model, train_meta = tune_model_random_search(
            train,
            val,
            seed=int(args.seed),
            trials=int(args.tune_trials),
            early_stopping_rounds=int(args.tune_early_stopping_rounds),
            max_estimators=int(args.tune_max_estimators),
        )
    else:
        model, train_meta = train_model(train, val, seed=int(args.seed))
    metrics_out = evaluate_model(model, val, bins=int(args.calibration_bins))

    baselines = baseline_metrics(train, val)
    # Contextualize accuracy with baseline.
    try:
        majority_acc = float(baselines["majority"]["majority_accuracy"])
        metrics_out["lift_over_majority_accuracy"] = float(metrics_out["accuracy"] - majority_acc)
    except Exception:
        pass
    metrics_out["baselines"] = baselines
    metrics_out["split"] = split_meta

    save_artifacts(
        out_dir=str(args.out_dir),
        model=model,
        bundle=bundle,
        train_meta=train_meta,
        metrics_out=metrics_out,
    )

    print(json.dumps({"train": train_meta, "metrics": metrics_out}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
