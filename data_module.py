# data_module.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader

RAW_TARGETS = ["Traffic_Speed_kmh", "Road_Occupancy_%"]


@dataclass
class PreprocessArtifacts:
    feature_names: List[str]
    target_names: List[str]
    x_scaler_mean: np.ndarray
    x_scaler_scale: np.ndarray
    y_scaler_mean: np.ndarray
    y_scaler_scale: np.ndarray
    idx_speed_in_X: int
    idx_occ_in_X: int


class TimeSeriesWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 3, "X must be [N, lookback, n_features]"
        assert y.ndim == 2 and y.shape[1] == 2, "y must be [N, 2]"
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _mode(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return s.value_counts().index[0]


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    minute_of_day = df["Timestamp"].dt.hour * 60 + df["Timestamp"].dt.minute
    df["tod_sin"] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
    df["tod_cos"] = np.cos(2 * np.pi * minute_of_day / (24 * 60))
    dow = df["Timestamp"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


def _one_hot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = [c for c in ["Traffic_Light_State", "Weather_Condition"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    return df


def _coerce_numeric_and_dropna(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c == "Timestamp":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


def _build_windows_no_gaps(timestamps: np.ndarray, X_scaled: np.ndarray, y_scaled: np.ndarray,
                          lookback: int, freq_minutes: int,
                          train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray):
    # Build windows but do not cross gaps in timestamps.
    freq_ns = int(freq_minutes * 60 * 1_000_000_000)
    ts = timestamps.astype("datetime64[ns]").astype(np.int64)
    deltas = np.diff(ts)
    breaks = np.where(deltas > int(1.5 * freq_ns))[0]
    run_starts = np.r_[0, breaks + 1]
    run_ends = np.r_[breaks + 1, len(ts)]

    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []

    for rs, re in zip(run_starts, run_ends):
        run = np.arange(rs, re, dtype=np.int64)
        if len(run) <= lookback:
            continue
        for j in range(lookback, len(run)):
            t = run[j]
            w = run[j - lookback:j]
            if train_mask[t]:
                X_train.append(X_scaled[w, :]); y_train.append(y_scaled[t, :])
            elif val_mask[t]:
                X_val.append(X_scaled[w, :]); y_val.append(y_scaled[t, :])
            elif test_mask[t]:
                X_test.append(X_scaled[w, :]); y_test.append(y_scaled[t, :])

    def stack(X_list, y_list):
        if len(X_list) == 0:
            return np.zeros((0, lookback, X_scaled.shape[1]), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)

    return (*stack(X_train, y_train), *stack(X_val, y_val), *stack(X_test, y_test))


def load_and_prepare(
    csv_path: str,
    lookback: int = 72,
    freq_minutes: int = 5,
    drop_traffic_condition: bool = True,
    target_smooth_window: int = 1,
) -> Tuple[TimeSeriesWindowDataset, TimeSeriesWindowDataset, TimeSeriesWindowDataset, PreprocessArtifacts]:
    """
    Treat the dataset as ONE time series (ignore lat/lon). There is one row per Timestamp in your CSV.
    Optionally smooth the targets with a trailing moving average.

    target_smooth_window:
      - 1  => use raw targets
      - 3  => 15 min average
      - 6  => 30 min average
      - 12 => 60 min average
    """
    df = pd.read_csv(csv_path)
    if "Timestamp" not in df.columns:
        raise ValueError("Expected a 'Timestamp' column.")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Drop lat/lon explicitly (ignore them)
    for col in ["Latitude", "Longitude"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Optional: drop Traffic_Condition (often derived)
    if drop_traffic_condition and "Traffic_Condition" in df.columns:
        df = df.drop(columns=["Traffic_Condition"])

    # Time features + one-hot
    df = _add_time_features(df)
    df = _one_hot(df)

    # Ensure raw targets exist
    for c in RAW_TARGETS:
        if c not in df.columns:
            raise ValueError(f"Missing raw target column: {c}")

    # Coerce numeric and drop NaN
    df = _coerce_numeric_and_dropna(df)

    # Optional smoothing for targets (create new target columns, keep raw as inputs)
    target_cols = RAW_TARGETS
    if target_smooth_window and int(target_smooth_window) > 1:
        w = int(target_smooth_window)
        df["Traffic_Speed_kmh_target"] = df["Traffic_Speed_kmh"].rolling(w).mean()
        df["Road_Occupancy_%_target"] = df["Road_Occupancy_%"].rolling(w).mean()
        df = df.dropna().reset_index(drop=True)
        target_cols = ["Traffic_Speed_kmh_target", "Road_Occupancy_%_target"]

    # Split (single series)
    n = len(df)
    i_train = int(n * 0.70)
    i_val = int(n * 0.85)

    train_mask = np.zeros(n, dtype=bool); val_mask = np.zeros(n, dtype=bool); test_mask = np.zeros(n, dtype=bool)
    train_mask[:i_train] = True
    val_mask[i_train:i_val] = True
    test_mask[i_val:] = True

    # y and X
    y_df = df[target_cols].copy()
    X_df = df.drop(columns=["Timestamp"]).copy()  # X includes raw targets as features

    feature_names = list(X_df.columns)

    # indices for baseline (use RAW targets in X)
    idx_speed = feature_names.index("Traffic_Speed_kmh")
    idx_occ = feature_names.index("Road_Occupancy_%")

    # Fit scalers on TRAIN only
    x_scaler = StandardScaler().fit(X_df.iloc[:i_train].values)
    y_scaler = StandardScaler().fit(y_df.iloc[:i_train].values)

    X_scaled = x_scaler.transform(X_df.values)
    y_scaled = y_scaler.transform(y_df.values)

    timestamps = df["Timestamp"].values.astype("datetime64[ns]")

    X_train, y_train, X_val, y_val, X_test, y_test = _build_windows_no_gaps(
        timestamps=timestamps,
        X_scaled=X_scaled,
        y_scaled=y_scaled,
        lookback=lookback,
        freq_minutes=freq_minutes,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    train_ds = TimeSeriesWindowDataset(X_train, y_train)
    val_ds = TimeSeriesWindowDataset(X_val, y_val)
    test_ds = TimeSeriesWindowDataset(X_test, y_test)

    art = PreprocessArtifacts(
        feature_names=feature_names,
        target_names=target_cols,
        x_scaler_mean=x_scaler.mean_.astype(np.float64),
        x_scaler_scale=x_scaler.scale_.astype(np.float64),
        y_scaler_mean=y_scaler.mean_.astype(np.float64),
        y_scaler_scale=y_scaler.scale_.astype(np.float64),
        idx_speed_in_X=idx_speed,
        idx_occ_in_X=idx_occ,
    )

    return train_ds, val_ds, test_ds, art


def make_dataloaders(train_ds: Dataset, val_ds: Dataset, test_ds: Dataset,
                     batch_size: int, num_workers: int = 2, pin_memory: bool = True,
                     distributed: bool = False, rank: int = 0, world_size: int = 1):
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle = False
    else:
        train_sampler = val_sampler = test_sampler = None
        shuffle = True

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, sampler=test_sampler,
                             num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler
