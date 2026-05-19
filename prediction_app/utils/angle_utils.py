"""Angle extraction utilities for per-frame prediction features.

Usa o calculador_angulos do módulo utils para manter consistência
com o processamento de ângulos em todo o projeto.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import sys
from pathlib import Path

# Importar o calculador_angulos do utils raiz
utils_root = Path(__file__).resolve().parents[2] / "utils"
sys.path.insert(0, str(utils_root))
from calculador_angulos import calculate_angle

from .constants import ANGLE_COLUMNS, ANGLE_DEFINITIONS, LANDMARK_INDEX


def _get_landmark_point(
    landmarks: List[Dict[str, float]],
    visibility: List[float],
    presence: List[float],
    landmark_name: str,
    min_detection_confidence: float,
    min_presence_confidence: float,
) -> Optional[np.ndarray]:
    """Extrai ponto de landmark com verificação de visibilidade e presença.
    
    Retorna apenas X,Y (coordenadas 2D), descartando Z para consistência
    com o calculador_angulos que usa apenas coordenadas de imagem.
    """
    idx = LANDMARK_INDEX.get(landmark_name)
    if idx is None or idx >= len(landmarks):
        return None

    vis = visibility[idx] if idx < len(visibility) else 0.0
    pres = presence[idx] if idx < len(presence) else 0.0

    if vis < min_detection_confidence or pres < min_presence_confidence:
        return None

    lm = landmarks[idx]
    # Retorna apenas X,Y (coordenadas 2D)
    return np.asarray([lm["x"], lm["y"]], dtype=np.float32)


def _get_landmark_visibility(
    visibility: List[float],
    landmark_name: str,
) -> float:
    """Extrai apenas a visibilidade de um landmark.
    
    Retorna 0.0 se o landmark não existir ou índice for inválido.
    """
    idx = LANDMARK_INDEX.get(landmark_name)
    if idx is None or idx >= len(visibility):
        return 0.0
    return float(visibility[idx])


def extract_angles_from_frame(
    landmarks: List[Dict[str, float]],
    visibility: List[float],
    presence: List[float],
    min_detection_confidence: float,
    min_presence_confidence: float,
) -> Dict[str, float]:
    """Extrai os 8 ângulos articulares de um frame usando coordenadas 2D.
    
    Retorna tanto os ângulos quanto a visibility_weight para cada ângulo.
    
    Usa a função calculate_angle do módulo utils/calculador_angulos.py
    que trabalha com ângulos planares (X,Y apenas).
    """
    angles = {name: np.nan for name in ANGLE_COLUMNS}
    visibility_weights = {f"{name}_visibility_weight": np.nan for name in ANGLE_COLUMNS}

    for angle_name, (start_name, vertex_name, end_name) in ANGLE_DEFINITIONS.items():
        p1 = _get_landmark_point(
            landmarks,
            visibility,
            presence,
            start_name,
            min_detection_confidence,
            min_presence_confidence,
        )
        vertex = _get_landmark_point(
            landmarks,
            visibility,
            presence,
            vertex_name,
            min_detection_confidence,
            min_presence_confidence,
        )
        p2 = _get_landmark_point(
            landmarks,
            visibility,
            presence,
            end_name,
            min_detection_confidence,
            min_presence_confidence,
        )
        angle_val = calculate_angle(p1, vertex, p2)
        angles[angle_name] = float(angle_val) if angle_val is not None else np.nan
        
        # Calcular visibility_weight como a média das visibilidades dos 3 pontos
        vis_start = _get_landmark_visibility(visibility, start_name)
        vis_vertex = _get_landmark_visibility(visibility, vertex_name)
        vis_end = _get_landmark_visibility(visibility, end_name)
        
        if all(v > 0 for v in [vis_start, vis_vertex, vis_end]):
            visibility_weights[f"{angle_name}_visibility_weight"] = float(np.mean([vis_start, vis_vertex, vis_end]))

    # Combinar ângulos e visibility_weights no resultado final
    result = {**angles, **visibility_weights}
    return result
