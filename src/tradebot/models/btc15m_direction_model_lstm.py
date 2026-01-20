"""LSTM model for 15-minute BTC direction prediction.

Same task as btc15m_direction_model.py but using PyTorch LSTM instead of XGBoost.

Features (same as XGBoost model):
- delta = (S_t - K) / K
- abs_delta
- return_1m, return_3m, return_5m
- vol_5m
- trend_5m
- time_remaining

Usage:
    python -m tradebot.models.btc15m_direction_model_lstm \
        --input-csv data/btc_1m_coinbase_180d.csv \
        --out-dir data/btc15m_model_lstm_180d \
        --purge-minutes 30
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics as sklearn_metrics


@dataclass(frozen=True)
class DatasetBundle:
    X: np.ndarray  # (n_samples, seq_len, n_features) for LSTM
    y: np.ndarray
    feature_names: list[str]
    sample_timestamps: np.ndarray


def load_candles_csv(path: str) -> pd.DataFrame:
    """Load 1-minute candles from CSV."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    ts = df["timestamp"]
    if np.issubdtype(ts.dtype, np.number):
        ts_f = ts.astype(float)
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


def build_training_dataset(
    candles: pd.DataFrame,
    *,
    seed: int = 7,
    samples_per_window: int = 1,
    window_minutes: int = 15,
    seq_len: int = 10,  # Number of historical observations for LSTM sequence
) -> DatasetBundle:
    """Build the supervised dataset from 1-minute candles.
    
    For LSTM, we create sequences of features from multiple observation points.
    """
    ts = candles["timestamp"].to_numpy(dtype=np.int64)
    close = candles["close"].to_numpy(dtype=float)

    n = len(candles)
    if n < 60:
        raise ValueError("Not enough candles to build a dataset")

    # Precompute contiguity
    diffs = ts[1:] - ts[:-1]
    ok_1m = diffs == 60
    bad_prefix = np.zeros(n + 1, dtype=int)
    if n >= 2:
        bad_prefix[1:n] = np.cumsum((~ok_1m).astype(int))
        bad_prefix[n] = int(bad_prefix[n - 1])

    def _is_contiguous(i0: int, i1: int) -> bool:
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
    n_features = len(feature_names)

    X_sequences: list[np.ndarray] = []  # Each is (seq_len, n_features)
    y_rows: list[int] = []
    t_rows: list[int] = []

    end_offset = window_minutes - 1

    def compute_features(obs_idx: int, K: float, m: int) -> list[float] | None:
        """Compute features at a given observation index."""
        if obs_idx - 5 < 0:
            return None
        
        S = close[obs_idx]
        if not (S and math.isfinite(S) and S > 0):
            return None

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

        # vol_5m
        rets = []
        for j in range(obs_idx - 4, obs_idx + 1):
            prev = close[j - 1]
            cur = close[j]
            if prev <= 0:
                rets.append(0.0)
            else:
                rets.append((cur / prev) - 1.0)
        vol_5m = float(np.std(np.asarray(rets, dtype=float), ddof=0))

        # trend_5m
        yv = close[obs_idx - 4 : obs_idx + 1]
        xv = np.arange(5, dtype=float)
        x_mean = float(np.mean(xv))
        y_mean = float(np.mean(yv))
        denom = float(np.sum((xv - x_mean) ** 2))
        trend_5m = 0.0
        if denom > 0:
            trend_5m = float(np.sum((xv - x_mean) * (yv - y_mean)) / denom)

        time_remaining = float(window_minutes - m)

        return [delta, abs_delta, return_1m, return_3m, return_5m, vol_5m, trend_5m, time_remaining]

    # For LSTM, we'll create sequences by looking at multiple observation points
    # within each 15-minute window
    for start_idx in range(seq_len + 5, n - end_offset):
        end_idx = start_idx + end_offset

        if not _is_contiguous(start_idx - seq_len - 5, end_idx + 1):
            continue

        K = close[start_idx]
        if not (K and math.isfinite(K) and K > 0):
            continue

        y = 1 if close[end_idx] > K else 0

        for _ in range(samples_per_window):
            # Pick a random observation minute
            m = int(rng.integers(1, window_minutes))
            obs_idx = start_idx + m

            # Build sequence: look back seq_len minutes from observation
            sequence = []
            valid = True
            for seq_offset in range(seq_len):
                hist_idx = obs_idx - (seq_len - 1 - seq_offset)
                # For historical points, compute features relative to K
                hist_m = m - (seq_len - 1 - seq_offset)
                if hist_m < 1:
                    hist_m = 1  # Clamp time_remaining calculation
                
                feats = compute_features(hist_idx, K, hist_m)
                if feats is None:
                    valid = False
                    break
                sequence.append(feats)

            if not valid or len(sequence) != seq_len:
                continue

            X_sequences.append(np.array(sequence, dtype=np.float32))
            y_rows.append(y)
            t_rows.append(int(ts[start_idx]))

    if not X_sequences:
        raise ValueError("No training samples were produced")

    X = np.stack(X_sequences, axis=0)  # (n_samples, seq_len, n_features)
    y_arr = np.asarray(y_rows, dtype=np.int64)
    t_arr = np.asarray(t_rows, dtype=np.int64)

    # Sort by time
    order = np.argsort(t_arr)
    return DatasetBundle(
        X=X[order],
        y=y_arr[order],
        feature_names=feature_names,
        sample_timestamps=t_arr[order],
    )


def time_split_purged(
    bundle: DatasetBundle,
    *,
    train_frac: float = 0.8,
    purge_minutes: int = 0,
) -> tuple[DatasetBundle, DatasetBundle, dict[str, Any]]:
    """Time split with optional purge gap."""
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
        meta["mode"] = "time_split"
        return train, val, meta

    train_mask = bundle.sample_timestamps <= (cut_ts - purge_seconds)
    val_mask = bundle.sample_timestamps >= (cut_ts + purge_seconds)

    n_train = int(np.sum(train_mask))
    n_val = int(np.sum(val_mask))
    meta.update({"n_train": n_train, "n_val": n_val, "mode": "time_split_purged"})

    if n_train < 1 or n_val < 1:
        raise ValueError("Purge too large for dataset")

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
    return train, val, meta


class LSTMClassifier(nn.Module):
    """LSTM-based binary classifier for BTC direction prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last time step
        last_hidden = lstm_out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits.squeeze(-1)


def train_lstm(
    train: DatasetBundle,
    val: DatasetBundle,
    *,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 0.001,
    batch_size: int = 256,
    epochs: int = 100,
    patience: int = 10,
    device: str | None = None,
) -> tuple[LSTMClassifier, dict[str, Any]]:
    """Train LSTM model with early stopping."""
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Training on device: {device}")
    
    # Normalize features
    X_train = train.X.copy()
    X_val = val.X.copy()
    
    # Compute mean/std from training data (flatten to 2D for stats)
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    feat_mean = X_train_flat.mean(axis=0)
    feat_std = X_train_flat.std(axis=0) + 1e-8
    
    # Normalize
    X_train = (X_train - feat_mean) / feat_std
    X_val = (X_val - feat_mean) / feat_std
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(train.y, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(val.y, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    input_size = X_train.shape[-1]
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0
    history: list[dict[str, float]] = []
    
    X_val_dev = X_val_t.to(device)
    y_val_dev = y_val_t.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        train_loss /= n_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_dev)
            val_loss = criterion(val_logits, y_val_dev).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs >= 0.5).astype(int)
            val_acc = float(np.mean(val_preds == val.y))
        
        scheduler.step(val_loss)
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"  Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.5f} val_loss={val_loss:.5f} val_acc={val_acc:.4f} [BEST]")
        else:
            epochs_no_improve += 1
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.5f} val_loss={val_loss:.5f} val_acc={val_acc:.4f}")
        
        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    meta = {
        "backend": "pytorch_lstm",
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "lr": lr,
        "batch_size": batch_size,
        "best_epoch": len(history) - epochs_no_improve,
        "total_epochs": len(history),
        "best_val_loss": best_val_loss,
        "feat_mean": feat_mean.tolist(),
        "feat_std": feat_std.tolist(),
        "history": history[-20:],  # Keep last 20 epochs
    }
    
    return model, meta


def evaluate_model(
    model: LSTMClassifier,
    val: DatasetBundle,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    *,
    bins: int = 10,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate LSTM model."""
    model = model.to(device)
    model.eval()
    
    # Normalize
    X_val = (val.X - feat_mean) / feat_std
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(X_val_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    y_true = val.y
    y_pred = (probs >= 0.5).astype(int)
    
    # Calibration data
    edges = np.linspace(0.0, 1.0, bins + 1)
    calibration: list[dict[str, float]] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        n = int(np.sum(mask))
        if n == 0:
            continue
        calibration.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "count": float(n),
            "p_mean": float(np.mean(probs[mask])),
            "y_mean": float(np.mean(y_true[mask])),
        })
    
    out = {
        "n_val": int(len(y_true)),
        "accuracy": float(sklearn_metrics.accuracy_score(y_true, y_pred)),
        "roc_auc": float(sklearn_metrics.roc_auc_score(y_true, probs)),
        "logloss": float(sklearn_metrics.log_loss(y_true, probs, labels=[0, 1])),
        "calibration": calibration,
    }
    
    # Lift over majority
    base_rate = float(np.mean(y_true))
    majority_acc = float(max(base_rate, 1.0 - base_rate))
    out["lift_over_majority_accuracy"] = out["accuracy"] - majority_acc
    out["baselines"] = {
        "majority": {
            "base_rate": base_rate,
            "majority_accuracy": majority_acc,
        }
    }
    
    return out


def save_artifacts(
    *,
    out_dir: str,
    model: LSTMClassifier,
    train_meta: dict[str, Any],
    metrics_out: dict[str, Any],
    feature_names: list[str],
) -> None:
    """Save model and metrics."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    
    # Save metrics
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, sort_keys=True)
    
    # Save meta
    meta = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "feature_names": feature_names,
        "train": train_meta,
        "metrics": metrics_out,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM model for 15m BTC direction")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to 1m BTC candles CSV")
    parser.add_argument("--out-dir", type=str, default="models/btc15m_lstm", help="Output directory")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--purge-minutes", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=10, help="LSTM sequence length")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"Loading candles from {args.input_csv}...")
    candles = load_candles_csv(args.input_csv)
    print(f"  Loaded {len(candles):,} candles")
    
    print(f"Building dataset (seq_len={args.seq_len})...")
    bundle = build_training_dataset(
        candles,
        seed=args.seed,
        seq_len=args.seq_len,
    )
    print(f"  Built {bundle.X.shape[0]:,} samples, shape={bundle.X.shape}")
    
    print(f"Splitting (train_frac={args.train_frac}, purge={args.purge_minutes}min)...")
    train, val, split_meta = time_split_purged(
        bundle,
        train_frac=args.train_frac,
        purge_minutes=args.purge_minutes,
    )
    print(f"  Train: {train.X.shape[0]:,}, Val: {val.X.shape[0]:,}")
    
    print("Training LSTM...")
    model, train_meta = train_lstm(
        train,
        val,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )
    train_meta["seq_len"] = int(args.seq_len)
    
    print("Evaluating...")
    feat_mean = np.array(train_meta["feat_mean"])
    feat_std = np.array(train_meta["feat_std"])
    metrics_out = evaluate_model(model, val, feat_mean, feat_std)
    metrics_out["split"] = split_meta
    
    print(f"\nResults:")
    print(f"  Accuracy: {metrics_out['accuracy']:.5f}")
    print(f"  ROC-AUC:  {metrics_out['roc_auc']:.5f}")
    print(f"  Logloss:  {metrics_out['logloss']:.5f}")
    
    print(f"\nSaving to {args.out_dir}...")
    save_artifacts(
        out_dir=args.out_dir,
        model=model,
        train_meta=train_meta,
        metrics_out=metrics_out,
        feature_names=bundle.feature_names,
    )
    
    print("\nDone!")
    print(json.dumps({"train": train_meta, "metrics": metrics_out}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
