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
    only_with_landmarks=False,
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
        only_with_landmarks: se True, inclui apenas frames com landmarks detectados
    """
    cap = cv2.VideoCapture(video_path)
    
    # Calcular FPS ajustado se filtrar apenas frames com landmarks
    output_fps = float(video_info['original_fps'])
    if only_with_landmarks:
        # Contar frames com landmarks
        frames_with_landmarks = sum(1 for f in frames_data if len(f['filtered_landmarks']) > 0)
        # Ajustar FPS proporcionalmente
        if frames_with_landmarks > 0 and video_info['total_frames'] > 0:
            output_fps = float(video_info['original_fps'] * (frames_with_landmarks / video_info['total_frames']))
    
    # Garantir FPS válido
    output_fps = max(1.0, min(output_fps, 120.0))
    
    # Criar writer de vídeo com codec compatível
    # Tentar diferentes codecs em ordem de preferência
    fourcc_options = [
        ('H264', cv2.VideoWriter_fourcc(*'H264')),  # H.264
        ('avc1', cv2.VideoWriter_fourcc(*'avc1')),  # MPEG-4 Part 10
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # MPEG-4 Part 2 (XVID)
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # MPEG-4 Part 2
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),  # Motion JPEG (fallback)
    ]
    
    out = None
    for codec_name, fourcc in fourcc_options:
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            output_fps,
            (video_info['width'], video_info['height'])
        )
        if out.isOpened():
            break
        out.release()
    
    if not out or not out.isOpened():
        raise RuntimeError(f"Não foi possível criar o VideoWriter com nenhum codec disponível")
    
    frame_idx = 0
    data_idx = 0
    interval = video_info['interval']
    frames_written = 0
    
    # Contar frames com landmarks para callback
    frames_with_landmarks_count = sum(1 for f in frames_data if len(f['filtered_landmarks']) > 0) if only_with_landmarks else video_info['total_frames']
    
    # Manter última detecção válida para desenhar em frames intermediários
    last_landmarks = None
    last_visibility = None
    last_presence = None
    last_had_landmarks = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flag para indicar se este frame foi processado
        frame_processed = frame_idx % interval == 0 and data_idx < len(frames_data)
        
        # Se este frame foi processado, atualizar a última detecção
        if frame_processed:
            data = frames_data[data_idx]
            last_landmarks = data['landmarks']
            last_visibility = data['visibility']
            last_presence = data['presence']
            # Verificar se tem landmarks com confiança suficiente
            last_had_landmarks = len(data['filtered_landmarks']) > 0
            data_idx += 1
        else:
            # Para frames intermediários, resetar a flag se filtrar apenas com landmarks
            if only_with_landmarks:
                last_had_landmarks = False
        
        # Se filtrar apenas frames com landmarks, pular frames sem landmarks
        if only_with_landmarks and not last_had_landmarks:
            frame_idx += 1
            continue
        
        # Desenhar landmarks usando a última detecção válida
        if last_landmarks is not None:
            frame = draw_landmarks_on_image(
                frame,
                last_landmarks,
                last_visibility,
                last_presence,
                min_pose_detection_confidence,
                min_pose_presence_confidence,
            )
        
        # Escrever frame
        out.write(frame)
        frames_written += 1
        frame_idx += 1
        
        if progress_callback:
            progress_callback(frames_written, frames_with_landmarks_count)
    
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
