"""Pose detection and drawing helpers for video inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

# Keep runtime consistent with the existing streamlit pipeline.
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("DISPLAY", "")

import cv2
import mediapipe as mp
import numpy as np

from .constants import LANDMARK_CONNECTIONS, POSE_MODELS_DIR


def get_angle_color(angle: float, min_angle: float = 45.0, max_angle: float = 180.0) -> Tuple[int, int, int]:
    """
    Mapeia um ângulo para uma cor BGR: vermelho (45°) -> amarelo -> verde (180°)
    
    Args:
        angle: Ângulo em graus
        min_angle: Ângulo mínimo (vermelho) - padrão 45°
        max_angle: Ângulo máximo (verde) - padrão 180°
    
    Returns:
        Tuple (B, G, R) para OpenCV
    """
    # Normalizar ângulo entre 0 e 1
    normalized = (angle - min_angle) / (max_angle - min_angle)
    normalized = np.clip(normalized, 0.0, 1.0)
    
    # Interpolação: Vermelho (0, 0, 255) -> Amarelo (0, 255, 255) -> Verde (0, 255, 0)
    if normalized < 0.5:
        # Vermelho -> Amarelo
        t = normalized * 2  # 0.0 a 1.0
        r = int(255)
        g = int(255 * t)
        b = int(0)
    else:
        # Amarelo -> Verde
        t = (normalized - 0.5) * 2  # 0.0 a 1.0
        r = int(255 * (1 - t))
        g = int(255)
        b = int(0)
    
    return (b, g, r)  # BGR para OpenCV


class PoseLandmarkerDetector:
    """MediaPipe Pose Landmarker wrapper using IMAGE or VIDEO running mode."""

    def __init__(
        self,
        model_variant: str = "full",
        min_confidence: float = 0.2,
        video_mode: bool = True,
        num_poses: int = 1,
    ) -> None:
        model_map = {
            "lite": POSE_MODELS_DIR / "pose_landmarker_lite.task",
            "full": POSE_MODELS_DIR / "pose_landmarker_full.task",
            "heavy": POSE_MODELS_DIR / "pose_landmarker_heavy.task",
        }

        model_file = model_map.get(model_variant, model_map["full"])
        if not Path(model_file).exists():
            raise FileNotFoundError(f"Pose model not found: {model_file}")

        running_mode = (
            mp.tasks.vision.RunningMode.VIDEO
            if video_mode
            else mp.tasks.vision.RunningMode.IMAGE
        )

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_file)),
            running_mode=running_mode,
            num_poses=num_poses,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )

        self.video_mode = video_mode
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def _to_mp_image(self, frame_bgr: np.ndarray) -> mp.Image:
        rgb_frame = np.ascontiguousarray(frame_bgr[:, :, ::-1]).astype(np.uint8)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    @staticmethod
    def _unpack_result(result) -> Tuple[List[Dict[str, float]], List[float], List[float]]:
        landmarks: List[Dict[str, float]] = []
        visibility: List[float] = []
        presence: List[float] = []

        if result and result.pose_landmarks:
            for lm in result.pose_landmarks[0]:
                landmarks.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)})
                visibility.append(float(getattr(lm, "visibility", 0.0)))
                presence.append(float(getattr(lm, "presence", 0.0)))

        return landmarks, visibility, presence

    def detect_pose(self, frame_bgr: np.ndarray):
        mp_image = self._to_mp_image(frame_bgr)
        result = self.landmarker.detect(mp_image)
        return self._unpack_result(result)

    def detect_for_video(self, frame_bgr: np.ndarray, timestamp_ms: int):
        mp_image = self._to_mp_image(frame_bgr)
        if self.video_mode:
            result = self.landmarker.detect_for_video(mp_image, int(timestamp_ms))
        else:
            result = self.landmarker.detect(mp_image)
        return self._unpack_result(result)


def has_valid_landmarks(
    landmarks: List[Dict[str, float]],
    visibility: List[float],
    presence: List[float],
    min_detection_confidence: float,
    min_presence_confidence: float,
    min_points: int = 8,
) -> bool:
    if not landmarks:
        return False

    valid = 0
    for idx in range(len(landmarks)):
        vis = visibility[idx] if idx < len(visibility) else 0.0
        pres = presence[idx] if idx < len(presence) else 0.0
        if vis >= min_detection_confidence and pres >= min_presence_confidence:
            valid += 1

    return valid >= min_points


def draw_landmarks_on_frame(
    frame_bgr: np.ndarray,
    landmarks: List[Dict[str, float]],
    visibility: List[float],
    presence: List[float],
    min_detection_confidence: float,
    min_presence_confidence: float,
    edge_colors: Dict[Tuple[int, int], Tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """
    Desenha landmarks e conexões no frame.
    
    Args:
        frame_bgr: Frame em BGR
        landmarks: Lista de landmarks
        visibility: Lista de visibilidade
        presence: Lista de presença
        min_detection_confidence: Confiança mínima de detecção
        min_presence_confidence: Confiança mínima de presença
        edge_colors: Dict com cores das arestas {(start_idx, end_idx): (B, G, R)}
    
    Returns:
        Frame com landmarks desenhados
    """
    output = frame_bgr.copy()
    height, width = output.shape[:2]

    valid_idx = set()
    for i in range(len(landmarks)):
        vis = visibility[i] if i < len(visibility) else 0.0
        pres = presence[i] if i < len(presence) else 0.0
        if vis >= min_detection_confidence and pres >= min_presence_confidence:
            valid_idx.add(i)

    for start, end in LANDMARK_CONNECTIONS:
        if start in valid_idx and end in valid_idx:
            start_xy = (
                int(landmarks[start]["x"] * width),
                int(landmarks[start]["y"] * height),
            )
            end_xy = (
                int(landmarks[end]["x"] * width),
                int(landmarks[end]["y"] * height),
            )
            
            # Usar cor específica se fornecida, caso contrário verde padrão
            if edge_colors and (start, end) in edge_colors:
                line_color = edge_colors[(start, end)]
            else:
                line_color = (0, 255, 0)
            
            cv2.line(output, start_xy, end_xy, line_color, 2)

    for idx in valid_idx:
        center = (
            int(landmarks[idx]["x"] * width),
            int(landmarks[idx]["y"] * height),
        )
        cv2.circle(output, center, 4, (0, 255, 0), -1)
        cv2.circle(output, center, 4, (0, 0, 255), 1)

    return output


def draw_angles_on_frame(
    frame_bgr: np.ndarray,
    landmarks: List[Dict[str, float]],
    visibility: List[float],
    presence: List[float],
    min_detection_confidence: float,
    min_presence_confidence: float,
    angle_definitions: Dict[str, Tuple[str, str, str]] | None = None,
    landmark_index: Dict[str, int] | None = None,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], Tuple[int, int, int]]]:
    """
    Desenha os valores dos ângulos articulares no frame com cores dinâmicas.
    
    Args:
        frame_bgr: Frame em BGR
        landmarks: Lista de landmarks
        visibility: Lista de visibilidade
        presence: Lista de presença
        min_detection_confidence: Confiança mínima de detecção
        min_presence_confidence: Confiança mínima de presença
        angle_definitions: Dict com definições de ângulos {nome: (start, vertex, end)}
        landmark_index: Dict com índices dos landmarks {nome: índice}
    
    Returns:
        Tuple[frame_com_angulos, edge_colors_dict]
            - frame_com_angulos: Frame com ângulos desenhados
            - edge_colors_dict: Dict com cores das arestas {(start_idx, end_idx): (B, G, R)}
    """
    from .angle_utils import calculate_angle
    from .constants import ANGLE_DEFINITIONS, LANDMARK_INDEX
    
    if angle_definitions is None:
        angle_definitions = ANGLE_DEFINITIONS
    if landmark_index is None:
        landmark_index = LANDMARK_INDEX
    
    output = frame_bgr.copy()
    height, width = output.shape[:2]
    
    # Dicionário para armazenar cores das arestas
    edge_colors = {}
    
    # Bg para textos
    bg_color = (0, 0, 0)  # Preto para background
    
    # Para cada ângulo definido
    angles_drawn = []
    angles_skipped = {}
    
    # Mapeamento especial: para certos ângulos, desenhar em um landmark diferente
    special_text_positions = {
        "right_ombro": "right_shoulder",   # Desenhar no ombro direito
        "left_ombro": "left_shoulder",     # Desenhar no ombro esquerdo
    }
    
    # Mapeamento de arestas para ângulos: qual ângulo colore qual aresta
    # Estrutura: (start_landmark_idx, end_landmark_idx) -> "angle_name"
    edge_to_angle_mapping = {
        (12, 14): "right_ombro",      # Right shoulder-elbow → right_ombro
        (14, 16): "right_cotovelo",   # Right elbow-wrist → right_cotovelo
        (12, 24): "right_ombro",      # Right shoulder-hip → right_ombro
        (24, 26): "right_quadril",    # Right hip-knee → right_quadril
        (26, 28): "right_joelho",     # Right knee-ankle → right_joelho
        (11, 13): "left_ombro",       # Left shoulder-elbow → left_ombro
        (13, 15): "left_cotovelo",    # Left elbow-wrist → left_cotovelo
        (11, 23): "left_ombro",       # Left shoulder-hip → left_ombro
        (23, 25): "left_quadril",     # Left hip-knee → left_quadril
        (25, 27): "left_joelho",      # Left knee-ankle → left_joelho
    }
    
    # Dicionário temporário para armazenar ângulos calculados para uso na colorização
    calculated_angles = {}
    
    for angle_name, (start_name, vertex_name, end_name) in angle_definitions.items():
        # Obter índices
        start_idx = landmark_index.get(start_name)
        vertex_idx = landmark_index.get(vertex_name)
        end_idx = landmark_index.get(end_name)
        
        # Validar índices
        if (start_idx is None or vertex_idx is None or end_idx is None or
            start_idx >= len(landmarks) or vertex_idx >= len(landmarks) or end_idx >= len(landmarks)):
            angles_skipped[angle_name] = "índice inválido"
            continue
        
        # Verificar visibilidade e presença de todos os 3 pontos
        vis_start = visibility[start_idx] if start_idx < len(visibility) else 0.0
        pres_start = presence[start_idx] if start_idx < len(presence) else 0.0
        vis_v = visibility[vertex_idx] if vertex_idx < len(visibility) else 0.0
        pres_v = presence[vertex_idx] if vertex_idx < len(presence) else 0.0
        vis_end = visibility[end_idx] if end_idx < len(visibility) else 0.0
        pres_end = presence[end_idx] if end_idx < len(presence) else 0.0
        
        # Todos os 3 pontos devem ter confiança mínima
        if (vis_start < min_detection_confidence or pres_start < min_presence_confidence or
            vis_v < min_detection_confidence or pres_v < min_presence_confidence or
            vis_end < min_detection_confidence or pres_end < min_presence_confidence):
            angles_skipped[angle_name] = f"vis/pres baixo: S({vis_start:.2f}/{pres_start:.2f}) V({vis_v:.2f}/{pres_v:.2f}) E({vis_end:.2f}/{pres_end:.2f})"
            continue
        
        # Calcular ângulo
        try:
            p1 = np.array([landmarks[start_idx]["x"], landmarks[start_idx]["y"], landmarks[start_idx]["z"]])
            vertex = np.array([landmarks[vertex_idx]["x"], landmarks[vertex_idx]["y"], landmarks[vertex_idx]["z"]])
            p2 = np.array([landmarks[end_idx]["x"], landmarks[end_idx]["y"], landmarks[end_idx]["z"]])
            
            angle = calculate_angle(p1, vertex, p2)
            
            if np.isnan(angle):
                continue
            
                # Armazenar ângulo calculado para uso na colorização das arestas
                calculated_angles[angle_name] = angle
                
                text_landmark_name = special_text_positions[angle_name]
                text_landmark_idx = landmark_index.get(text_landmark_name)
                if text_landmark_idx is not None and text_landmark_idx < len(landmarks):
                    # Usar o landmark especial para posicionar o texto
                    vertex_xy = np.array([
                        int(landmarks[text_landmark_idx]["x"] * width),
                        int(landmarks[text_landmark_idx]["y"] * height),
                    ])
                else:
                    # Fallback: usar o vértice original
                    vertex_xy = np.array([
                        int(landmarks[vertex_idx]["x"] * width),
                        int(landmarks[vertex_idx]["y"] * height),
                    ])
            else:
                # Usar vértice normal
                vertex_xy = np.array([
                    int(landmarks[vertex_idx]["x"] * width),
                    int(landmarks[vertex_idx]["y"] * height),
                ])
            
            start_xy = np.array([
                int(landmarks[start_idx]["x"] * width),
                int(landmarks[start_idx]["y"] * height),
            ])
            
            end_xy = np.array([
                int(landmarks[end_idx]["x"] * width),
                int(landmarks[end_idx]["y"] * height),
            ])
            
            # Ponto médio entre start e end
            midpoint = (start_xy + end_xy) / 2.0
            
            # Vetor do vértice para o ponto médio (para dentro do corpo)
            to_midpoint = midpoint - vertex_xy
            norm_to_mid = np.linalg.norm(to_midpoint)
            
            if norm_to_mid > 0:
                # Normalizar e reverter (agora aponta para fora do corpo)
                direction_out = -to_midpoint / norm_to_mid
            else:
                direction_out = np.array([1.0, 0.0])
            
            # Distância do texto até a aresta central
            offset_distance = 15
            text_offset = direction_out * offset_distance
            text_pos = vertex_xy + text_offset
            text_pos = (int(text_pos[0]), int(text_pos[1]))
            
            # Desenhar texto com fundo (APENAS NÚMERO INTEIRO)
            angle_int = int(round(angle))
            text = str(angle_int)  # Apenas números, sem símbolo
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Obter cor dinâmica baseada no ângulo
            angle_color = get_angle_color(angle, min_angle=45.0, max_angle=180.0)
            
            # Obter tamanho do texto
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Ajustar posição do texto para não sair da tela
            text_pos = (
                max(5, min(text_pos[0], width - text_size[0] - 5)),
                max(text_size[1] + 5, min(text_pos[1], height - 5))
            )
            
            # Desenhar background
            padding = 3
            cv2.rectangle(
                output,
                (text_pos[0] - padding, text_pos[1] - text_size[1] - padding),
                (text_pos[0] + text_size[0] + padding, text_pos[1] + padding),
                bg_color,
                -1
            )
            
            # Desenhar texto com cor dinâmica
            cv2.putText(
                output,
                text,
                text_pos,
                font,
                font_scale,
                angle_color,
                thickness,
                cv2.LINE_AA
            )
            
            angles_drawn.append(f"{angle_name}: {angle_int}")
        except Exception as e:
            # Ignorar erros ao calcular ângulo
            angles_skipped[angle_name] = str(e)
            continue
    
    # Debug: printar quais ângulos foram desenhados
    if angles_drawn:
        print(f"✓ Ângulos desenhados: {', '.join(angles_drawn)}")
    if angles_skipped:
        print(f"✗ Ângulos pulados: {angles_skipped}")
    
    # Aplicar mapeamento de arestas para ângulos
    # Para cada aresta, verificar qual ângulo deve colorir e usar sua cor
    for edge, angle_name in edge_to_angle_mapping.items():
        if angle_name in calculated_angles:
            angle_value = calculated_angles[angle_name]
            edge_color = get_angle_color(angle_value, min_angle=45.0, max_angle=180.0)
            edge_colors[edge] = edge_color
    
    return output, edge_colors
