"""Constants used by the Random Forest video prediction pipeline.

CRITICAL: The order of ANGLE_COLUMNS MUST match EXACTLY the order used in training
(2-random_forest_training.ipynb, variable `angle_columns` in cell 9).

Training order:
  angle_columns = ['right_cotovelo', 'left_cotovelo', 'right_ombro', 'left_ombro',
                   'right_joelho', 'left_joelho', 'right_quadril', 'left_quadril']

The StandardScaler is positional: it applies mean/std by column index, not by name.
Wrong order = wrong normalization = garbage predictions.
"""

from pathlib import Path

import os as _os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_MODELS_DIR = PROJECT_ROOT / "ml_models"
POSE_MODELS_DIR = PROJECT_ROOT / "models"
# On Hugging Face Spaces the repo dir is read-only; use /tmp instead.
_running_on_hf = bool(_os.environ.get("SPACE_ID"))
DEFAULT_OUTPUT_DIR = (
    Path("/tmp/race_outputs")
    if _running_on_hf
    else PROJECT_ROOT / "videos_output" / "prediction_app"
)

DEFAULT_PROCESS_FPS = 5
DEFAULT_MIN_POSE_DETECTION_CONFIDENCE = 0.2
DEFAULT_MIN_POSE_PRESENCE_CONFIDENCE = 0.2
DEFAULT_POSE_MODEL_VARIANT = "full"

# IMPORTANT: This order MUST match the training notebook (2-random_forest_training.ipynb)
# Training used: window_size=15, angle_columns only (no visibility)
# Total features: 15 frames × 8 angles = 120 features
# Order from training: right_cotovelo, left_cotovelo, right_ombro, left_ombro,
#                      right_joelho, left_joelho, right_quadril, left_quadril
TRAINING_WINDOW_SIZE = 15
ANGLE_COLUMNS = [
    "right_cotovelo",
    "left_cotovelo",
    "right_ombro",
    "left_ombro",
    "right_joelho",
    "left_joelho",
    "right_quadril",
    "left_quadril",
]

# Visibility weight columns (same order as ANGLE_COLUMNS)
VISIBILITY_COLUMNS = [
    "right_cotovelo_visibility_weight",
    "left_cotovelo_visibility_weight",
    "right_ombro_visibility_weight",
    "left_ombro_visibility_weight",
    "right_joelho_visibility_weight",
    "left_joelho_visibility_weight",
    "right_quadril_visibility_weight",
    "left_quadril_visibility_weight",
]

LANDMARK_INDEX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# Angle definitions (anatomical order: p1 → vertex → p2)
# CRITICAL: Dict order matches ANGLE_COLUMNS above (same as training)
# Cotovelo: flexão/extensão (shoulder → elbow → wrist)
# Ombro: abução/adução (elbow → shoulder → hip)
# Joelho: flexão/extensão (hip → knee → ankle)
# Quadril: flexão/extensão (knee → hip → shoulder)
ANGLE_DEFINITIONS = {
    "right_cotovelo": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_cotovelo":  ("left_shoulder",  "left_elbow",  "left_wrist"),
    "right_ombro":    ("right_elbow",    "right_shoulder", "right_hip"),
    "left_ombro":     ("left_elbow",     "left_shoulder",  "left_hip"),
    "right_joelho":   ("right_hip",      "right_knee",   "right_ankle"),
    "left_joelho":    ("left_hip",       "left_knee",    "left_ankle"),
    "right_quadril":  ("right_knee",     "right_hip",    "right_shoulder"),
    "left_quadril":   ("left_knee",      "left_hip",     "left_shoulder"),
}

LANDMARK_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24),
    (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
    (29, 31), (30, 32),
]
