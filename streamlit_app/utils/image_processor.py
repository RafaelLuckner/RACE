"""Processamento de imagens com pose detection"""

import cv2
import numpy as np
from utils.mediapipe_utils import PoseLandmarker, filter_landmarks
from config import LANDMARK_COLORS


def draw_landmarks_on_image(image, landmarks, visibility, presence, min_pose_detection_confidence=0.2, min_pose_presence_confidence=0.2):
    """
    Desenha landmarks na imagem
    
    Args:
        image: numpy array BGR
        landmarks: lista de landmarks
        visibility: lista de visibilidade
        presence: lista de presença
        min_pose_detection_confidence: confiança mínima de detecção
        min_pose_presence_confidence: confiança mínima de presença
    """
    output_image = image.copy()
    height, width, _ = image.shape
    
    # Desenhar conexões entre landmarks (skeleton)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24),
        (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
        (29, 31), (30, 32)
    ]
    
    # Desenhar conexões
    filtered_idx = set()
    for i, (lm, vis, pres) in enumerate(zip(landmarks, visibility, presence)):
        if vis >= min_pose_detection_confidence and pres >= min_pose_presence_confidence:
            filtered_idx.add(i)
    
    # Desenhar linhas
    for start, end in connections:
        if start in filtered_idx and end in filtered_idx:
            start_pos = (int(landmarks[start]['x'] * width), int(landmarks[start]['y'] * height))
            end_pos = (int(landmarks[end]['x'] * width), int(landmarks[end]['y'] * height))
            cv2.line(output_image, start_pos, end_pos, (0, 255, 0), 2)
    
    # Desenhar pontos
    for i, (lm, vis, pres) in enumerate(zip(landmarks, visibility, presence)):
        if vis >= min_pose_detection_confidence and pres >= min_pose_presence_confidence:
            x = int(lm['x'] * width)
            y = int(lm['y'] * height)
            cv2.circle(output_image, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(output_image, (x, y), 5, (0, 0, 255), 2)
    
    return output_image


def process_image(image_path, pose_detector, min_pose_detection_confidence=0.2, min_pose_presence_confidence=0.2):
    """
    Processa uma imagem e detecta pose
    
    Args:
        image_path: caminho da imagem
        pose_detector: instância de PoseLandmarker
        min_pose_detection_confidence: confiança mínima de detecção
        min_pose_presence_confidence: confiança mínima de presença
        
    Returns:
        processed_image: imagem com landmarks
        landmarks_data: dicionário com dados dos landmarks
    """
    # Ler imagem
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Não foi possível ler a imagem: {image_path}")
    
    # Detectar pose
    landmarks, visibility, presence = pose_detector.detect_pose(image)
    
    # Desenhar landmarks
    processed_image = draw_landmarks_on_image(
        image, landmarks, visibility, presence, min_pose_detection_confidence, min_pose_presence_confidence
    )
    
    # Preparar dados dos landmarks
    filtered = filter_landmarks(landmarks, visibility, presence, min_pose_detection_confidence, min_pose_presence_confidence)
    
    landmarks_data = {
        'total_landmarks': len(landmarks),
        'detected_landmarks': len(filtered),
        'landmarks': filtered
    }
    
    return processed_image, landmarks_data


def resize_image(image, max_width=800, max_height=600):
    """Redimensiona imagem mantendo proporção"""
    height, width = image.shape[:2]
    
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image
