"""Video validation and normalization utilities for handling WhatsApp and other video formats."""

import tempfile
from pathlib import Path
from typing import Tuple

import cv2


def validate_video(video_path: Path) -> Tuple[bool, str]:
    """
    Valida se um vídeo pode ser aberto e processado.
    
    Returns:
        (is_valid, message): Tupla com status e mensagem descritiva
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "❌ Não foi possível abrir o vídeo. Tente reconverter em outro formato."
        
        # Tentar ler pelo menos 1 frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return False, "❌ Vídeo corrompido ou sem frames válidos."
        
        return True, "✅ Vídeo válido"
    except Exception as e:
        return False, f"❌ Erro ao validar vídeo: {str(e)}"


def normalize_video_for_preview(video_path: Path, max_duration_seconds: int = 10) -> Path:
    """
    Reconverte um vídeo para um formato compatível com Streamlit.
    Ideal para vídeos do WhatsApp com codecs problemáticos.
    
    Args:
        video_path: Caminho do vídeo original
        max_duration_seconds: Duração máxima em segundos (para reduzir tamanho)
    
    Returns:
        Path: Caminho do vídeo reconvertido
        
    Raises:
        ValueError: Se o vídeo não puder ser processado
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
    
    # Obter propriedades do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if width == 0 or height == 0:
        cap.release()
        raise ValueError("Vídeo tem dimensões inválidas")
    
    # Calcular quantos frames processar (limitar à duração máxima)
    max_frames = int(fps * max_duration_seconds)
    frames_to_process = min(total_frames, max_frames)
    
    # Criar arquivo temporário para o vídeo reconvertido
    tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_output_path = Path(tmp_output.name)
    tmp_output.close()
    
    try:
        # Usar codec H.264 com qualidade boa
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(tmp_output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            # Fallback para codec alternativo
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(str(tmp_output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError("Não foi possível criar vídeo de saída com nenhum codec disponível")
        
        # Processar frames
        frame_count = 0
        while frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Garantir que o frame tem as dimensões corretas
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            
            out.write(frame)
            frame_count += 1
        
        out.release()
        cap.release()
        
        # Validar arquivo de saída
        if not tmp_output_path.exists() or tmp_output_path.stat().st_size == 0:
            raise ValueError("Falha ao reconverter vídeo - arquivo de saída vazio")
        
        return tmp_output_path
        
    except Exception as e:
        cap.release()
        # Limpar arquivo temporário em caso de erro
        if tmp_output_path.exists():
            tmp_output_path.unlink()
        raise ValueError(f"Erro ao reconverter vídeo: {str(e)}")


def get_compatible_preview_video(uploaded_file) -> Tuple[Path, bool]:
    """
    Prepara um arquivo de vídeo upload para preview no Streamlit.
    Reconverte se necessário.
    
    Args:
        uploaded_file: UploadedFile do Streamlit
    
    Returns:
        (video_path, is_converted): Caminho do arquivo e flag indicando se foi reconvertido
    """
    # Salvar arquivo original em temp
    tmp_original = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}")
    tmp_original.write(uploaded_file.getbuffer())
    tmp_original_path = Path(tmp_original.name)
    tmp_original.close()
    
    # Validar se pode abrir normalmente
    is_valid, _ = validate_video(tmp_original_path)
    
    if is_valid:
        return tmp_original_path, False
    
    # Se não é válido, tentar reconverter
    try:
        converted_path = normalize_video_for_preview(tmp_original_path, max_duration_seconds=5)
        # Limpar arquivo original
        tmp_original_path.unlink()
        return converted_path, True
    except Exception as e:
        # Limpar arquivo original
        tmp_original_path.unlink()
        raise ValueError(f"Não foi possível processar o vídeo: {str(e)}")
