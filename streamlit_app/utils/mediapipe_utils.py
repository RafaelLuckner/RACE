"""Utilitários para MediaPipe Pose"""

import os
# Desabilitar GPU e renderização gráfica para compatibilidade com Streamlit Cloud
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['DISPLAY'] = ''

import mediapipe as mp
import numpy as np
from pathlib import Path


class PoseLandmarker:
    """Classe para detecção de pose com MediaPipe"""

    def __init__(self, model_path="lite", min_confidence=0.5, video_mode=False, num_poses=1):
        """
        Inicializa o MediaPipe Pose Landmarker
        
        Args:
            model_path: "lite", "heavy" ou "full"
            min_confidence: Confiança mínima para detecção
            video_mode: Se True, usa RunningMode.VIDEO para melhor rastreamento
            num_poses: quantidade máxima de poses detectadas por frame
        """
        self.min_confidence = min_confidence
        self.model_path = model_path
        self.video_mode = video_mode
        self.num_poses = num_poses
        self.frame_count = 0
        
        # Obter o caminho absoluto da pasta raiz do projeto
        # mediapipe_utils.py está em: streamlit_app/utils/mediapipe_utils.py
        # Logo: parent.parent.parent = raiz do projeto
        script_dir = Path(__file__).resolve()  # Caminho absoluto do arquivo
        project_root = script_dir.parent.parent.parent  # Sobe 3 níveis até a raiz
        models_dir = project_root / "models"
        
        # Encontrar o caminho correto do modelo
        if model_path == "heavy":
            model_file = models_dir / "pose_landmarker_heavy.task"
        elif model_path == "full":
            model_file = models_dir / "pose_landmarker_full.task"
        else:
            model_file = models_dir / "pose_landmarker_lite.task"
        
        # Verificar se arquivo existe
        if not model_file.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {model_file}\n"
                f"Procurado em: {models_dir}\n"
                f"Arquivos disponíveis: {list(models_dir.glob('*.task')) if models_dir.exists() else 'Diretório não existe'}"
            )
        
        # Definir modo de execução
        running_mode = mp.tasks.vision.RunningMode.VIDEO if video_mode else mp.tasks.vision.RunningMode.IMAGE
        
        # Criar opções do detector
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_file)),
            running_mode=running_mode,
            num_poses=self.num_poses,
            min_pose_detection_confidence=self.min_confidence,
            min_pose_presence_confidence=self.min_confidence,
        )
        
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def detect_pose(self, image):
        """
        Detecta pose em uma imagem
        
        Args:
            image: numpy array (BGR)
            
        Returns:
            landmarks: lista de landmarks
            visibility: lista de visibilidade
            presence: lista de presença
        """
        # Converter BGR para RGB
        rgb_image = image[:, :, ::-1]
        
        # Criar imagem do MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detectar pose
        detection_result = self.landmarker.detect(mp_image)
        
        landmarks = []
        visibility = []
        presence = []
        
        if detection_result and detection_result.pose_landmarks:
            for landmark in detection_result.pose_landmarks[0]:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })
                visibility.append(landmark.visibility)
                presence.append(landmark.presence)
        
        return landmarks, visibility, presence

    def detect_for_video(self, image, timestamp_ms):
        """
        Detecta pose em um frame de vídeo (usa VIDEO mode para melhor rastreamento)
        
        Args:
            image: numpy array (BGR)
            timestamp_ms: timestamp em milissegundos
            
        Returns:
            landmarks: lista de landmarks
            visibility: lista de visibilidade
            presence: lista de presença
        """
        # Converter BGR para RGB
        rgb_image = image[:, :, ::-1]
        
        # Criar imagem do MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detectar pose com timestamp (necessário para VIDEO mode)
        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        landmarks = []
        visibility = []
        presence = []
        
        if detection_result and detection_result.pose_landmarks:
            for landmark in detection_result.pose_landmarks[0]:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })
                visibility.append(landmark.visibility)
                presence.append(landmark.presence)
        
        return landmarks, visibility, presence

    def __del__(self):
        """Limpar recursos"""
        if hasattr(self, 'landmarker'):
            del self.landmarker


def get_landmark_info(landmark_idx):
    """Retorna informações sobre um landmark específico"""
    from config import BODY_LANDMARKS
    return BODY_LANDMARKS.get(landmark_idx, f"Unknown ({landmark_idx})")


def is_landmark_visible(visibility, presence, min_pose_detection_confidence=0.2, min_pose_presence_confidence=0.2):
    """Verifica se um landmark está visível com base nos limites"""
    return visibility >= min_pose_detection_confidence and presence >= min_pose_presence_confidence


def filter_landmarks(landmarks, visibility, presence, min_pose_detection_confidence=0.2, min_pose_presence_confidence=0.2):
    """Filtra landmarks com base em confiança de detecção e presença"""
    filtered = []
    for i, (lm, vis, pres) in enumerate(zip(landmarks, visibility, presence)):
        if is_landmark_visible(vis, pres, min_pose_detection_confidence, min_pose_presence_confidence):
            filtered.append({
                'index': i,
                'landmark': lm,
                'visibility': vis,
                'presence': pres,
                'name': get_landmark_info(i)
            })
    return filtered
