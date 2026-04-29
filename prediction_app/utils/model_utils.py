"""Helpers for loading model artifacts and preparing feature schema."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

from .constants import ANGLE_COLUMNS, ML_MODELS_DIR


def _load_pickle(path: Path):
    with open(path, "rb") as file_obj:
        return pickle.load(file_obj)


def resolve_artifact_paths(models_dir: Path | None = None) -> Dict[str, Path]:
    base_dir = models_dir or ML_MODELS_DIR
    paths = {
        "model": base_dir / "random_forest_4exercises.pkl",
        "scaler": base_dir / "random_forest_4exercises_scaler.pkl",
        "label_map": base_dir / "random_forest_4exercises_label_map.pkl",
    }

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required model artifacts: " + ", ".join(missing)
        )

    return paths


def normalize_label_map(raw_label_map: Dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Normalizes label map to both directions."""
    if not isinstance(raw_label_map, dict) or not raw_label_map:
        raise ValueError("label_map artifact is empty or invalid")

    key_types = {type(key) for key in raw_label_map.keys()}
    value_types = {type(value) for value in raw_label_map.values()}

    if int in key_types and str in value_types:
        id_to_name = {int(key): str(value) for key, value in raw_label_map.items()}
        name_to_id = {name: idx for idx, name in id_to_name.items()}
        return name_to_id, id_to_name

    if str in key_types and int in value_types:
        name_to_id = {str(key): int(value) for key, value in raw_label_map.items()}
        id_to_name = {idx: name for name, idx in name_to_id.items()}
        return name_to_id, id_to_name

    raise ValueError(
        "label_map artifact must be either {name: id} or {id: name}"
    )


def load_model_artifacts(models_dir: Path | None = None):
    paths = resolve_artifact_paths(models_dir)
    model = _load_pickle(paths["model"])
    scaler = _load_pickle(paths["scaler"])
    raw_label_map = _load_pickle(paths["label_map"])
    name_to_id, id_to_name = normalize_label_map(raw_label_map)
    return model, scaler, name_to_id, id_to_name, paths


def infer_window_size_from_scaler(scaler, angle_columns: List[str] | None = None) -> int:
    columns = angle_columns or ANGLE_COLUMNS
    n_features = getattr(scaler, "n_features_in_", None)
    if n_features is None:
        raise ValueError("Scaler does not expose n_features_in_ for window size inference")

    if len(columns) == 0:
        raise ValueError("angle_columns must not be empty")

    if n_features % len(columns) != 0:
        raise ValueError(
            f"Feature count mismatch: scaler expects {n_features} features, "
            f"but {len(columns)} angle columns do not divide this value"
        )

    window_size = n_features // len(columns)
    if window_size <= 0:
        raise ValueError("Invalid inferred window size")

    return int(window_size)


def build_feature_columns(window_size: int, angle_columns: List[str] | None = None) -> List[str]:
    """
    Builds feature column names in the exact order used during training.
    
    IMPORTANT: The order of angle_columns MUST match training (2-random_forest_training.ipynb):
    right_cotovelo, left_cotovelo, right_ombro, left_ombro, right_joelho, left_joelho, right_quadril, left_quadril
    """
    columns = angle_columns or ANGLE_COLUMNS
    feature_columns = []
    for frame_idx in range(1, window_size + 1):
        for angle_col in columns:
            feature_columns.append(f"frame_{frame_idx}_{angle_col}")
    return feature_columns


def validate_feature_columns(X: 'pd.DataFrame', expected_window_size: int, angle_columns: List[str] | None = None) -> Tuple[bool, str]:
    """
    Validates that feature columns match expected training format.
    
    Args:
        X: Feature DataFrame
        expected_window_size: Window size used during training (should be 15)
        angle_columns: Angle columns used (should match ANGLE_COLUMNS)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    columns = angle_columns or ANGLE_COLUMNS
    expected_columns = build_feature_columns(expected_window_size, columns)
    
    if X.empty:
        return True, ""  # Empty dataframe is OK
    
    # Check if columns exist and are in correct order
    missing_cols = [col for col in expected_columns if col not in X.columns]
    if missing_cols:
        return False, f"Missing columns: {missing_cols[:5]}..."
    
    # Check if first few columns match (order validation)
    actual_cols = list(X.columns)
    if actual_cols[:len(expected_columns)] != expected_columns:
        return False, f"Column order mismatch. Expected first cols: {expected_columns[:3]}, got: {actual_cols[:3]}"
    
    return True, ""
