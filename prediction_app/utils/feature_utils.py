"""Temporal feature generation for Random Forest window inference."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .constants import ANGLE_COLUMNS, VISIBILITY_COLUMNS
from .model_utils import build_feature_columns


def build_frames_dataframe(frame_records: List[Dict], angle_columns: List[str] | None = None, visibility_columns: List[str] | None = None) -> pd.DataFrame:
    angle_cols = angle_columns or ANGLE_COLUMNS
    vis_cols = visibility_columns or VISIBILITY_COLUMNS
    all_feature_cols = list(angle_cols) + list(vis_cols)
    
    df = pd.DataFrame(frame_records)

    if df.empty:
        return pd.DataFrame(columns=["frame", "timestamp_s", "has_landmarks", *all_feature_cols])

    for col in all_feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    if "has_landmarks" not in df.columns:
        df["has_landmarks"] = False

    keep_columns = ["frame", "timestamp_s", "has_landmarks", *all_feature_cols, "landmarks", "visibility", "presence"]
    for col in keep_columns:
        if col not in df.columns:
            df[col] = None

    return df.sort_values("frame").reset_index(drop=True)


def _fill_missing_values_rowwise(X: pd.DataFrame) -> pd.DataFrame:
    X = X.astype(float)
    row_means = X.mean(axis=1)
    X = X.T.fillna(row_means).T
    X = X.fillna(0.0)
    return X


def create_temporal_features_window(
    df: pd.DataFrame,
    window_size: int,
    angle_columns: List[str] | None = None,
    visibility_columns: List[str] | None = None,
    min_landmark_frames_in_window: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates one row per temporal window using only angle values, matching the training notebook.

    Each output feature column follows the notebook format:
    frame_1_right_cotovelo, frame_1_right_ombro, ..., frame_N_left_quadril.
    
    Total features: window_size × n_angles
    Example: 15 frames × 8 angles = 120 features
    """
    angle_cols = angle_columns or ANGLE_COLUMNS
    vis_cols = visibility_columns or VISIBILITY_COLUMNS
    
    feature_columns = build_feature_columns(window_size, list(angle_cols))

    features_list: List[Dict[str, float]] = []
    metadata_list: List[Dict[str, float]] = []

    if df.empty or len(df) < window_size:
        return pd.DataFrame(columns=feature_columns), pd.DataFrame()

    for i in range(len(df) - window_size + 1):
        window = df.iloc[i : i + window_size]

        frame_diffs = window["frame"].diff().iloc[1:].to_numpy(dtype=float)
        if frame_diffs.size > 0 and not np.all(frame_diffs > 0):
            continue

        landmark_count = int(window["has_landmarks"].sum())
        if landmark_count < min_landmark_frames_in_window:
            continue

        row = {}
        for frame_offset in range(window_size):
            frame_data = window.iloc[frame_offset]
            
            # Adicionar apenas ângulos (matching training: 15 × 8 = 120 features)
            for angle_col in angle_cols:
                row[f"frame_{frame_offset + 1}_{angle_col}"] = frame_data.get(angle_col, np.nan)

        features_list.append(row)
        metadata_list.append(
            {
                "window_index": len(metadata_list),
                "start_frame": int(window.iloc[0]["frame"]),
                "end_frame": int(window.iloc[-1]["frame"]),
                "start_timestamp_s": float(window.iloc[0]["timestamp_s"]),
                "end_timestamp_s": float(window.iloc[-1]["timestamp_s"]),
                "landmark_frames": landmark_count,
            }
        )

    X = pd.DataFrame(features_list, columns=feature_columns)
    metadata = pd.DataFrame(metadata_list)

    if not X.empty:
        X = _fill_missing_values_rowwise(X)

    return X, metadata
