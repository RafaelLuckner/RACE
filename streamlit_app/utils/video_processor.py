"""Processamento de vídeos com pose detection"""

import cv2
import time
from utils.mediapipe_utils import filter_landmarks
from utils.image_processor import draw_landmarks_on_image


def process_video(
    video_path,
    pose_detector,
    fps_process=15,
    min_pose_detection_confidence=0.2,
    min_pose_presence_confidence=0.2,
    inference_scale=1.0,
    progress_callback=None,
):
    """
    Processa vídeo e detecta pose em cada frame
    
    Args:
        video_path: caminho do vídeo
        pose_detector: instância de PoseLandmarker
        fps_process: frames por segundo a processar
        min_pose_detection_confidence: confiança mínima de detecção
        min_pose_presence_confidence: confiança mínima de presença
        inference_scale: escala do frame para inferência (0.3-1.0)
        progress_callback: função para callback de progresso
        
    Returns:
        frames_data: lista com dados de cada frame processado
        video_info: informações do vídeo
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
    
    # Obter informações do vídeo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calcular intervalo de frames a processar
    interval = max(1, int(original_fps / fps_process))
    
    video_info = {
        'total_frames': total_frames,
        'original_fps': original_fps,
        'width': width,
        'height': height,
        'fps_process': fps_process,
        'interval': interval
    }
    
    frames_data = []
    frame_idx = 0
    processed_frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int((frame_idx / max(original_fps, 1)) * 1000)
        frame_for_inference = frame
        if inference_scale < 0.999:
            frame_for_inference = cv2.resize(
                frame,
                None,
                fx=inference_scale,
                fy=inference_scale,
                interpolation=cv2.INTER_AREA,
            )

        start_time = time.time()
        if getattr(pose_detector, "video_mode", False):
            landmarks, visibility, presence = pose_detector.detect_for_video(frame_for_inference, timestamp_ms)
        else:
            landmarks, visibility, presence = pose_detector.detect_pose(frame_for_inference)
        processing_time = time.time() - start_time

        filtered = filter_landmarks(
            landmarks,
            visibility,
            presence,
            min_pose_detection_confidence,
            min_pose_presence_confidence,
        )

        frame_data = {
            'frame_idx': frame_idx,
            'processed_frame_idx': processed_frame_idx,
            'total_landmarks': len(landmarks),
            'detected_landmarks': len(filtered),
            'landmarks': landmarks,
            'visibility': visibility,
            'presence': presence,
            'processing_time': processing_time,
            'filtered_landmarks': filtered,
            'timestamp': frame_idx / max(original_fps, 1),
        }

        frames_data.append(frame_data)
        processed_frame_idx += 1

        if progress_callback:
            progress_callback(processed_frame_idx, max(total_frames // interval, 1))

        # Pula frames intermediários sem decodificar imagem completa para ganhar desempenho.
        skipped = 0
        while skipped < interval - 1:
            grabbed = cap.grab()
            if not grabbed:
                break
            skipped += 1

        frame_idx += (skipped + 1)

        if skipped < interval - 1:
            break
    
    cap.release()
    return frames_data, video_info


def create_output_video(
    video_path,
    frames_data,
    video_info,
    output_path,
    pose_detector,
    min_pose_detection_confidence=0.2,
    min_pose_presence_confidence=0.2,
    progress_callback=None,
):
    """
    Cria vídeo de saída com landmarks desenhados
    
    Args:
        video_path: caminho do vídeo original
        frames_data: dados dos frames processados
        video_info: informações do vídeo
        output_path: caminho do vídeo de saída
        pose_detector: instância de PoseLandmarker (para reprocessar frames)
        min_pose_detection_confidence: confiança mínima de detecção
        min_pose_presence_confidence: confiança mínima de presença
        progress_callback: função para callback de progresso
    """
    cap = cv2.VideoCapture(video_path)
    
    # Criar writer de vídeo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        video_info['original_fps'],
        (video_info['width'], video_info['height'])
    )
    
    frame_idx = 0
    data_idx = 0
    interval = video_info['interval']
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Se este frame foi processado, desenhar landmarks
        if frame_idx % interval == 0 and data_idx < len(frames_data):
            data = frames_data[data_idx]
            frame = draw_landmarks_on_image(
                frame,
                data['landmarks'],
                data['visibility'],
                data['presence'],
                min_pose_detection_confidence,
                min_pose_presence_confidence,
            )
            data_idx += 1
        
        # Escrever frame
        out.write(frame)
        frame_idx += 1
        
        if progress_callback:
            progress_callback(frame_idx, video_info['total_frames'])
    
    cap.release()
    out.release()


def get_frame_from_video(video_path, frame_idx):
    """Retorna um frame específico do vídeo"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    return None


def get_frames_by_indices(video_path, frame_indices):
    """Retorna múltiplos frames abrindo o vídeo apenas uma vez."""
    if len(frame_indices) == 0:
        return {}

    indices = sorted(set(int(i) for i in frame_indices))
    cap = cv2.VideoCapture(video_path)
    frames = {}

    if not cap.isOpened():
        return frames

    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames[idx] = frame
    finally:
        cap.release()

    return frames
