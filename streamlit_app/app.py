"""
Aplicativo Streamlit para Análise de Pose com MediaPipe
"""

import os
# Desabilitar GPU e renderização gráfica para compatibilidade com Streamlit Cloud
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['DISPLAY'] = ''

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from datetime import datetime
import time

from utils.mediapipe_utils import PoseLandmarker, filter_landmarks
from utils.image_processor import process_image, resize_image
from utils.video_processor import process_video, get_frame_from_video, get_frames_by_indices, create_output_video
from utils.export_utils import (
    export_landmarks_to_csv, 
    export_landmarks_to_json,
    create_landmarks_table
)
from config import DEFAULT_CONFIDENCE, DEFAULT_POSE_DETECTION_CONFIDENCE, DEFAULT_POSE_PRESENCE_CONFIDENCE, DEFAULT_FPS


# Configurar página Streamlit
st.set_page_config(
    page_title="Pose Analysis with MediaPipe",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS customizado
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    </style>
    """, unsafe_allow_html=True)


def init_session_state():
    """Inicializa variáveis de sessão"""
    if 'pose_detector' not in st.session_state:
        st.session_state.pose_detector = None
    if 'video_detector' not in st.session_state:
        st.session_state.video_detector = None
    if 'model_type' not in st.session_state:
        st.session_state.model_type = 'heavy'
    if 'video_frames_data' not in st.session_state:
        st.session_state.video_frames_data = None
    if 'video_info' not in st.session_state:
        st.session_state.video_info = None
    if 'last_video_path' not in st.session_state:
        st.session_state.last_video_path = None
    if 'last_fps_process' not in st.session_state:
        st.session_state.last_fps_process = None
    if 'last_visibility' not in st.session_state:
        st.session_state.last_visibility = None
    if 'last_presence' not in st.session_state:
        st.session_state.last_presence = None
    if 'frame_navigation_idx' not in st.session_state:
        st.session_state.frame_navigation_idx = 0
    if 'video_temp_path' not in st.session_state:
        st.session_state.video_temp_path = None
    if 'video_all_frames_processed' not in st.session_state:
        st.session_state.video_all_frames_processed = False
    if 'processed_frame_indices' not in st.session_state:
        st.session_state.processed_frame_indices = set()
    if 'processed_frame_counter' not in st.session_state:
        st.session_state.processed_frame_counter = 0
    if 'current_model_type' not in st.session_state:
        st.session_state.current_model_type = 'heavy'
    if 'current_pose_detection_confidence' not in st.session_state:
        st.session_state.current_pose_detection_confidence = DEFAULT_POSE_DETECTION_CONFIDENCE
    if 'current_pose_presence_confidence' not in st.session_state:
        st.session_state.current_pose_presence_confidence = DEFAULT_POSE_PRESENCE_CONFIDENCE
    if 'exercise_label' not in st.session_state:
        st.session_state.exercise_label = 'descanso'
    if 'cached_video_id' not in st.session_state:
        st.session_state.cached_video_id = None
    if 'cached_video_temp_path' not in st.session_state:
        st.session_state.cached_video_temp_path = None


def load_pose_detector(model_type='lite', video_mode=False):
    """Carrega o detector de pose"""
    try:
        detector = PoseLandmarker(
            model_path=model_type,
            min_confidence=DEFAULT_CONFIDENCE,
            video_mode=video_mode,
            num_poses=1,
        )
        return detector
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {str(e)}")
        st.info("Certifique-se de que os modelos estão em: ../models/")
        return None


def main():
    """Função principal"""
    init_session_state()
    
    # Header
    st.title("🏃 Pose Analysis with MediaPipe")
    st.markdown("Análise de pose em imagens e vídeos usando MediaPipe")
    st.divider()
    
    # Sidebar com configurações gerais
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Seleção do modelo
        model_type = st.radio(
            "Selecione o modelo:",
            options=['lite', 'full', 'heavy'],
            index=2,
            help="lite: mais rápido, full: intermediário, heavy: mais preciso"
        )
        
        if model_type != st.session_state.model_type:
            st.session_state.model_type = model_type
            st.session_state.pose_detector = None
            st.session_state.video_detector = None
        
        # Carregar detector se não estiver carregado
        if st.session_state.pose_detector is None:
            with st.spinner(f"Carregando modelo {model_type} (Imagens)..."):
                st.session_state.pose_detector = load_pose_detector(model_type, video_mode=False)
        
        if st.session_state.video_detector is None:
            with st.spinner(f"Carregando modelo {model_type} (Vídeos)..."):
                st.session_state.video_detector = load_pose_detector(model_type, video_mode=True)
        
        if st.session_state.pose_detector is None or st.session_state.video_detector is None:
            st.error("Não foi possível carregar o modelo. Encerrando.")
            return
        
        st.success(f"✓ Modelo {model_type} carregado")
        
        st.divider()
        
        # Hiperparâmetros
        st.subheader("Hiperparâmetros MediaPipe")
        
        min_pose_detection_confidence = st.slider(
            "Confiança de Detecção de Pose",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_POSE_DETECTION_CONFIDENCE,
            step=0.05,
            help="Confiança mínima para detectar pose"
        )
        
        min_pose_presence_confidence = st.slider(
            "Confiança de Presença de Pose",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_POSE_PRESENCE_CONFIDENCE,
            step=0.05,
            help="Confiança mínima de presença da pose"
        )

        inference_scale = st.select_slider(
            "Escala de Processamento (velocidade)",
            options=[0.5, 0.75, 1.0],
            value=0.75,
            format_func=lambda x: f"{int(x * 100)}%",
            help="Reduz apenas a resolução da inferência para acelerar. A exportação mantém resolução original.",
        )
        
        # Detectar mudanças nos parâmetros
        params_changed = False
        if (model_type != st.session_state.current_model_type or 
            min_pose_detection_confidence != st.session_state.current_pose_detection_confidence or 
            min_pose_presence_confidence != st.session_state.current_pose_presence_confidence):
            params_changed = True
            st.session_state.current_model_type = model_type
            st.session_state.current_pose_detection_confidence = min_pose_detection_confidence
            st.session_state.current_pose_presence_confidence = min_pose_presence_confidence
            
            # Limpar dados processados
            st.session_state.video_frames_data = {}
            st.session_state.processed_frame_indices = set()
            st.session_state.processed_frame_counter = 0
            st.session_state.video_all_frames_processed = False
            st.session_state.frame_navigation_idx = 0
            
            if st.session_state.video_info is not None:
                st.warning("⚠️ Parâmetros alterados! Os frames processados foram limpos. Selecione um novo frame para continuar.")
        
        st.divider()
        
        # Informações
        st.subheader("ℹ️ Informações")
        st.info(
            "Esta aplicação detecta landmarks de pose humana em imagens e vídeos "
            "usando MediaPipe. Você pode carregar mídia, visualizar resultados "
            "e exportar dados dos landmarks."
        )
    
    # Abas principais
    tab1, tab2 = st.tabs(["📷 Imagens", "🎬 Vídeos"])
    
    # ============ ABA 1: IMAGENS ============
    with tab1:
        st.header("Análise de Imagens e Frames")
        
        # Sub-abas dentro de Imagens
        sub_tab1, sub_tab2 = st.tabs(["Imagem Estática", "Frame a Frame"])
        
        # ===== SUB-ABA 1: IMAGEM ESTÁTICA =====
        with sub_tab1:
            st.subheader("Análise de Imagem Estática")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Upload da Imagem")
                uploaded_image = st.file_uploader(
                    "Selecione uma imagem",
                    type=['jpg', 'jpeg', 'png', 'bmp'],
                    key='image_uploader'
                )
                
                if uploaded_image:
                    # Salvar imagem temporária
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(uploaded_image.getbuffer())
                        temp_path = tmp_file.name
                    
                    # Processar imagem
                    try:
                        with st.spinner("Processando imagem..."):
                            processed_img, landmarks_data = process_image(
                                temp_path,
                                st.session_state.pose_detector,
                                min_pose_detection_confidence=min_pose_detection_confidence,
                                min_pose_presence_confidence=min_pose_presence_confidence
                            )
                        
                        # Limpar arquivo temporário
                        os.unlink(temp_path)
                        
                        # Mostrar resultado
                        with col2:
                            st.subheader("Resultado")
                            
                            # Redimensionar para visualização
                            display_img = resize_image(processed_img)
                            st.image(display_img, channels="BGR", width='stretch')
                            
                            # Estatísticas
                            st.metric(
                                "Total de Landmarks Detectados",
                                f"{landmarks_data['detected_landmarks']}/{landmarks_data['total_landmarks']}"
                            )
                        
                        # Seção de dados
                        st.subheader("📊 Dados dos Landmarks")
                        
                        if landmarks_data['landmarks']:
                            # Tabela de landmarks
                            landmarks_list = []
                            for lm in landmarks_data['landmarks']:
                                landmarks_list.append({
                                    'Landmark': lm['name'],
                                    'X': f"{lm['landmark']['x']:.4f}",
                                    'Y': f"{lm['landmark']['y']:.4f}",
                                    'Z': f"{lm['landmark']['z']:.4f}",
                                    'Visibility': f"{lm['visibility']:.4f}",
                                    'Presence': f"{lm['presence']:.4f}"
                                })
                            
                            st.dataframe(landmarks_list, width='stretch')
                            
                            # Download dos dados
                            col_csv, col_json = st.columns(2)
                            
                            with col_csv:
                                # Exportar para CSV
                                csv_data = pd.DataFrame(landmarks_list)
                                st.download_button(
                                    label="📥 Baixar dados (CSV)",
                                    data=csv_data.to_csv(index=False),
                                    file_name=f"landmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            
                            with col_json:
                                # Exportar para JSON
                                import json
                                json_data = json.dumps(
                                    {'landmarks': landmarks_list},
                                    indent=2
                                )
                                st.download_button(
                                    label="📥 Baixar dados (JSON)",
                                    data=json_data,
                                    file_name=f"landmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                        else:
                            st.warning("Nenhum landmark detectado com os parâmetros atuais.")
                        
                        # Download da imagem processada
                        st.divider()
                        _, ret = cv2.imencode('.jpg', processed_img)
                        st.download_button(
                            label="📥 Baixar imagem processada",
                            data=ret.tobytes(),
                            file_name=f"processed_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                            mime="image/jpeg"
                        )
                        
                    except Exception as e:
                        st.error(f"Erro ao processar imagem: {str(e)}")
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
        
        # ===== SUB-ABA 2: ANÁLISE FRAME A FRAME =====
        with sub_tab2:
            st.subheader("Análise Frame a Frame")
            st.write("Carregar um vídeo para análise frame a frame com navegação manual")
            
            uploaded_video_frames = st.file_uploader(
                "Selecione um vídeo",
                type=['mp4', 'avi', 'mov', 'mkv'],
                key='video_uploader_frames',
                max_upload_size = 1000

            )
            
            if uploaded_video_frames:
                # Salvar vídeo temporário
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_video_frames.getbuffer())
                    temp_frames_path = tmp_file.name
                
                try:
                    # Obter informações do vídeo
                    cap = cv2.VideoCapture(temp_frames_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    original_fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    
                    st.info(f"📹 Vídeo: {total_frames} frames @ {original_fps:.2f} FPS")
                    
                    # Slider para seleção de frame
                    frame_num = st.slider(
                        "Selecione um frame",
                        min_value=0,
                        max_value=total_frames - 1,
                        value=0,
                        step=1
                    )
                    
                    # Processamento do frame
                    col_reprocess, col_space = st.columns([1, 5])
                    with col_reprocess:
                        if st.button("🔄 Processar Frame", key='btn_process_frame_analysis'):
                            pass
                    
                    frame = get_frame_from_video(temp_frames_path, frame_num)
                    
                    if frame is not None:
                        # Processar com IMAGE mode
                        with st.spinner("Processando frame..."):
                            landmarks, visibility, presence = st.session_state.pose_detector.detect_pose(frame)
                            filtered = filter_landmarks(landmarks, visibility, presence, min_pose_detection_confidence, min_pose_presence_confidence)
                        
                        # Desenhar e exibir
                        from utils.image_processor import draw_landmarks_on_image
                        processed_frame = draw_landmarks_on_image(
                            frame, landmarks, visibility, presence,
                            min_pose_detection_confidence, min_pose_presence_confidence
                        )
                        
                        display_frame = resize_image(processed_frame)
                        st.image(display_frame, channels="BGR")
                        
                        # Métricas
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Frame", f"{frame_num + 1}/{total_frames}")
                        with col2:
                            st.metric("Timestamp", f"{frame_num / original_fps:.2f}s")
                        with col3:
                            st.metric("Landmarks", len(filtered))
                        
                        # Tabela de landmarks
                        if filtered:
                            st.write("**Landmarks Detectados:**")
                            landmarks_table = []
                            for lm in filtered:
                                landmarks_table.append({
                                    'Landmark': lm['name'],
                                    'X': f"{lm['landmark']['x']:.4f}",
                                    'Y': f"{lm['landmark']['y']:.4f}",
                                    'Z': f"{lm['landmark']['z']:.4f}",
                                    'Visibility': f"{lm['visibility']:.4f}",
                                    'Presence': f"{lm['presence']:.4f}"
                                })
                            st.dataframe(landmarks_table, width='stretch')
                        else:
                            st.warning("Nenhum landmark detectado neste frame")
                
                except Exception as e:
                    st.error(f"Erro: {str(e)}")
                finally:
                    if os.path.exists(temp_frames_path):
                        os.unlink(temp_frames_path)
    
    # ============ ABA 2: VÍDEOS ============
    with tab2:
        st.header("Análise Completa de Vídeos")
        st.write("Processamento automático de todo o vídeo com VIDEO mode (rastreamento contínuo)")
        
        st.divider()
        
        # Upload de vídeo
        uploaded_video = st.file_uploader(
            "Selecione um vídeo para análise completa",
            type=['mp4', 'avi', 'mov', 'mkv'],
            key='video_uploader_full'
        )
        
        if uploaded_video:
            # Identificar vídeo único
            video_id = f"{uploaded_video.name}_{uploaded_video.size}"

            # Reaproveitar arquivo temporário entre reruns para evitar custo de escrita repetida.
            cached_path = st.session_state.cached_video_temp_path
            if st.session_state.cached_video_id != video_id or not cached_path or not os.path.exists(cached_path):
                if cached_path and os.path.exists(cached_path):
                    try:
                        os.unlink(cached_path)
                    except Exception:
                        pass
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_video.getbuffer())
                    st.session_state.cached_video_temp_path = tmp_file.name
                    st.session_state.cached_video_id = video_id

            temp_video_path = st.session_state.cached_video_temp_path
            
            try:
                # Obter informações do vídeo
                cap = cv2.VideoCapture(temp_video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                original_fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                # Informações do vídeo
                st.subheader("📹 Informações do Vídeo")
                
                info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                with info_col1:
                    st.metric("Total de Frames", total_frames)
                with info_col2:
                    st.metric("FPS Original", f"{original_fps:.2f}")
                with info_col3:
                    st.metric("Resolução", f"{width}x{height}")
                with info_col4:
                    st.metric("Duração", f"{total_frames / original_fps:.1f}s")
                
                st.divider()
                
                # Controle de FPS para processamento
                st.subheader("⚙️ Configuração de Processamento")
                
                fps_process = st.slider(
                    "Frames Por Segundo (FPS) para Processar",
                    min_value=1,
                    max_value=int(original_fps) if original_fps > 0 else 30,
                    value=min(15, int(original_fps) if original_fps > 0 else 30),
                    step=1,
                    help="Quantos frames por segundo serão processados. Valores maiores = mais frames processados"
                )
                
                # Calcular frames que serão processados
                frame_interval = max(1, int(original_fps / fps_process)) if original_fps > 0 else 1
                expected_processed_frames = total_frames // frame_interval
                
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.metric("Intervalo de Frames", f"A cada {frame_interval} frame(s)")
                with info_col2:
                    st.metric("Frames a Processar", expected_processed_frames)
                with info_col3:
                    st.metric("Tempo de Processamento Est.", f"~{(expected_processed_frames * 0.1):.1f}s" if expected_processed_frames > 0 else "N/A")

                params_key = f"{video_id}_{model_type}_{min_pose_detection_confidence}_{min_pose_presence_confidence}_{fps_process}_{inference_scale}"
                
                st.divider()
                
                # Verificar se já foi processado com esses parâmetros
                if params_key not in st.session_state:
                    st.session_state[params_key] = None
                
                # Botão para processar ou reprocessar
                col_process, col_space = st.columns([1, 5])
                with col_process:
                    if st.button("▶️ Processar Vídeo", key='btn_process_full_video'):
                        # Forçar recriação do detector para garantir que o estado anterior é limpo
                        st.session_state.video_detector = load_pose_detector(model_type, video_mode=True)
                        # Forçar reprocessamento
                        st.session_state[params_key] = None
                
                # Verificar cache
                video_data_cached = st.session_state[params_key]
                
                if video_data_cached is None:
                    # SEMPRE recriar o detector para limpar estado anterior dos timestamps
                    # Isso evita erros de "timestamp not monotonically increasing"
                    st.session_state.video_detector = load_pose_detector(model_type, video_mode=True)
                    
                    # Processar vídeo
                    st.info(f"🔄 Processando vídeo com VIDEO mode (rastreamento contínuo) a {fps_process} FPS...")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def _progress(current, total):
                        ratio = min(current / max(total, 1), 1.0)
                        progress_bar.progress(ratio)
                        status_text.text(f"Processados: {current}/{max(total, 1)} frames")

                    start_process = time.time()
                    frames_list, processed_video_info = process_video(
                        temp_video_path,
                        st.session_state.video_detector,
                        fps_process=fps_process,
                        min_pose_detection_confidence=min_pose_detection_confidence,
                        min_pose_presence_confidence=min_pose_presence_confidence,
                        inference_scale=inference_scale,
                        progress_callback=_progress,
                    )
                    elapsed = time.time() - start_process
                    progress_bar.progress(1.0)
                    status_text.text(f"✅ Vídeo processado! {len(frames_list)} frames em {elapsed:.1f}s")
                    
                    # Cachear resultados
                    st.session_state[params_key] = {
                        'frames_data': frames_list,
                        'video_info': {
                            'total_frames': processed_video_info['total_frames'],
                            'original_fps': processed_video_info['original_fps'],
                            'fps_process': fps_process,
                            'width': width,
                            'height': height,
                            'interval': processed_video_info['interval'],
                            'inference_scale': inference_scale,
                            'model': model_type,
                            'min_pose_detection_confidence': min_pose_detection_confidence,
                            'min_pose_presence_confidence': min_pose_presence_confidence
                        }
                    }
                    
                    video_data_cached = st.session_state[params_key]
                else:
                    st.success("✅ Usando dados em cache")
                
                # Exibir resultados
                if video_data_cached:
                    frames_data = video_data_cached['frames_data']
                    video_info = video_data_cached['video_info']
                    frames_data_map = {f['frame_idx']: f for f in frames_data}
                    
                    st.divider()
                    st.subheader("📊 Resultados")
                    
                    # Estatísticas gerais
                    total_landmarks_detected = sum(
                        len(f['filtered_landmarks']) for f in frames_data
                    )
                    avg_processing_time = sum(
                        f.get('processing_time', 0.0) for f in frames_data
                    ) / len(frames_data) if frames_data else 0
                    
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    with stat_col1:
                        st.metric("Frames Processados", len(frames_data))
                    with stat_col2:
                        st.metric("Total de Landmarks", total_landmarks_detected)
                    with stat_col3:
                        st.metric("Tempo Médio/Frame (s)", f"{avg_processing_time:.3f}")
                    
                    st.divider()
                    
                    # Prévia de frames selecionados
                    st.subheader("🎬 Prévia de Frames")
                    
                    num_preview = min(4, len(frames_data))
                    if num_preview > 0:
                        preview_indices = np.linspace(0, total_frames - 1, num_preview, dtype=int)
                        preview_frames = get_frames_by_indices(temp_video_path, preview_indices)
                        preview_cols = st.columns(num_preview)
                        
                        for col_idx, frame_num in enumerate(preview_indices):
                            with preview_cols[col_idx]:
                                frame = preview_frames.get(int(frame_num))
                                if frame is not None and int(frame_num) in frames_data_map:
                                    frame_data = frames_data_map[int(frame_num)]
                                    from utils.image_processor import draw_landmarks_on_image
                                    processed = draw_landmarks_on_image(
                                        frame,
                                        frame_data['landmarks'],
                                        frame_data['visibility'],
                                        frame_data['presence'],
                                        min_pose_detection_confidence,
                                        min_pose_presence_confidence
                                    )
                                    display = resize_image(processed, max_width=250, max_height=250)
                                    st.image(display, channels="BGR", use_container_width=True)
                                    st.caption(f"Frame {int(frame_num)}")
                    
                    st.divider()
                    
                    # Exportação
                    st.subheader("💾 Exportar Dados")

                    exercise_label = st.selectbox(
                        "Exercício para adicionar no CSV",
                        options=["flexao", "agachamento", "rosca_biceps", "descanso"],
                        index=["flexao", "agachamento", "rosca_biceps", "descanso"].index(st.session_state.exercise_label),
                        key="exercise_label_select",
                    )
                    st.session_state.exercise_label = exercise_label
                    
                    col_csv, col_json, col_video = st.columns(3)
                    
                    with col_csv:
                        if st.button("📊 Exportar CSV", key='btn_export_csv_full'):
                            csv_path = f"pose_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            export_landmarks_to_csv(frames_data, video_info, csv_path, exercise=exercise_label)
                            
                            with open(csv_path, 'rb') as f:
                                st.download_button(
                                    label="📥 Baixar CSV",
                                    data=f.read(),
                                    file_name=csv_path,
                                    mime="text/csv",
                                    key='download_csv_full'
                                )
                            os.remove(csv_path)
                    
                    with col_json:
                        if st.button("📋 Exportar JSON", key='btn_export_json_full'):
                            json_path = f"pose_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            export_landmarks_to_json(frames_data, video_info, json_path)
                            
                            with open(json_path, 'rb') as f:
                                st.download_button(
                                    label="📥 Baixar JSON",
                                    data=f.read(),
                                    file_name=json_path,
                                    mime="application/json",
                                    key='download_json_full'
                                )
                            os.remove(json_path)
                    
                    with col_video:
                        if st.button("🎬 Exportar Vídeo", key='btn_export_video_full'):
                            st.info("⏳ Gerando vídeo com landmarks...")
                            
                            output_video_path = f"pose_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            def _export_progress(current, total):
                                ratio = min(current / max(total, 1), 1.0)
                                progress_bar.progress(ratio)
                                status_text.text(f"Gerando: {current}/{max(total, 1)} frames")

                            create_output_video(
                                temp_video_path,
                                frames_data,
                                video_info,
                                output_video_path,
                                st.session_state.video_detector,
                                min_pose_detection_confidence=min_pose_detection_confidence,
                                min_pose_presence_confidence=min_pose_presence_confidence,
                                progress_callback=_export_progress,
                            )
                            status_text.text(f"✅ Vídeo gerado com {len(frames_data)} frames processados a {fps_process} FPS!")
                            
                            with open(output_video_path, 'rb') as f:
                                st.download_button(
                                    label="📥 Baixar Vídeo",
                                    data=f.read(),
                                    file_name=output_video_path,
                                    mime="video/mp4",
                                    key='download_video_full'
                                )
                            os.remove(output_video_path)
                
            except Exception as e:
                st.error(f"Erro ao processar vídeo: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
            finally:
                pass


if __name__ == "__main__":
    main()
