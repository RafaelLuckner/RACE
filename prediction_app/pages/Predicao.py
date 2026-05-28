"""RACE — Predição de Exercícios."""

from __future__ import annotations

import tempfile
import time as _time
from pathlib import Path

import streamlit as st

from utils.constants import (
    ANGLE_COLUMNS,
    VISIBILITY_COLUMNS,
    TRAINING_WINDOW_SIZE,
    DEFAULT_MIN_POSE_DETECTION_CONFIDENCE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POSE_MODEL_VARIANT,
    DEFAULT_PROCESS_FPS,
    ML_MODELS_DIR,
)
from utils.model_utils import load_model_artifacts
from utils.pose_utils import PoseLandmarkerDetector
from utils.video_pipeline import RandomForestVideoPredictor
from utils.video_validation import get_compatible_preview_video, validate_video

CLASS_LABELS: dict = {
    "flexao":       "Flexão",
    "agachamento":  "Agachamento",
    "rosca_biceps": "Rosca Biceps",
    "descanso":     "Descanso",
}
EX_ICONS: dict = {
    "flexao":       "💪",
    "agachamento":  "🏋️",
    "rosca_biceps": "🦾",
}

_EXERCISES_INFO = [
    ("💪", "Flexão de Braço"),
    ("🏋️", "Agachamento"),
    ("🦾", "Rosca Bíceps"),
]


def _label(class_name: str) -> str:
    return CLASS_LABELS.get(class_name, class_name.replace("_", " ").title())


@st.cache_resource(show_spinner="⏳ Carregando modelo RF...")
def _load_model_cached():
    return load_model_artifacts(ML_MODELS_DIR)


def _create_pose_detector(model_variant: str, min_confidence: float):
    return PoseLandmarkerDetector(
        model_variant=model_variant,
        min_confidence=min_confidence,
        video_mode=True,
        num_poses=1,
    )


# Inicializar session state
if "processing_result" not in st.session_state:
    st.session_state.processing_result = None
if "uploaded_file_key" not in st.session_state:
    st.session_state.uploaded_file_key = None
if "preview_video_path" not in st.session_state:
    st.session_state.preview_video_path = None

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏋️ RACE")
    st.divider()
    st.markdown("#### Exercícios suportados")
    for icon, name in _EXERCISES_INFO:
        st.markdown(f"{icon} {name}")

    st.divider()
    st.markdown("#### ⚙️ Configurações de processamento")

    min_pose_confidence = st.slider(
        "Confiança na detecção",
        min_value=0.05,
        max_value=0.95,
        value=float(DEFAULT_MIN_POSE_DETECTION_CONFIDENCE),
        step=0.05,
        help="Quanto maior, mais rigoroso será na detecção de pose",
    )

    max_seconds = st.number_input(
        "Segundos a processar",
        min_value=1,
        max_value=600,
        value=30,
        step=1,
        help="Quantos segundos do vídeo você quer analisar",
    )

    st.divider()

    visualization_options = st.multiselect(
        "Visualização do vídeo",
        options=["classification", "angles", "landmarks"],
        default=["classification", "angles", "landmarks"],
        format_func=lambda x: {
            "classification": "🎯 Classificação",
            "angles": "📐 Ângulos",
            "landmarks": "👥 Landmarks",
        }[x],
        help="Escolha o que visualizar no vídeo de output",
    )

    if not visualization_options:
        visualization_options = ["classification", "angles", "landmarks"]

# ── CONTEÚDO PRINCIPAL ────────────────────────────────────────────────────────
st.title("🎯 Predição de Exercícios")
st.caption("Faça upload de um vídeo para classificar o exercício e contar repetições.")

uploaded_video = st.file_uploader(
    "📹 Upload seu vídeo",
    type=["mp4", "avi", "mov", "mkv"],
    accept_multiple_files=False,
)

if uploaded_video is not None:
    if st.session_state.uploaded_file_key != uploaded_video.file_id:
        st.session_state.processing_result = None
        st.session_state.uploaded_file_key = uploaded_video.file_id

    st.subheader("📺 Prévia do vídeo")
    preview_status = st.empty()

    try:
        with preview_status.container():
            st.info("🔄 Processando vídeo para prévia...")

        preview_video_path, was_converted = get_compatible_preview_video(uploaded_video)
        preview_status.empty()

        if was_converted:
            st.info("ℹ️ Vídeo foi reconvertido para formato compatível (comum em vídeos do WhatsApp)")

        with open(preview_video_path, "rb") as video_file:
            st.video(video_file, format="video/mp4", width=300)

        st.session_state.preview_video_path = preview_video_path

    except Exception as e:
        preview_status.empty()
        st.error(f"❌ Erro ao processar vídeo para prévia: {str(e)}")
        st.info(
            "**Dicas:** use MP4 com codec H.264. "
            "Se necessário, reconverta com Windows Media Player, QuickTime ou CloudConvert."
        )
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        process_button = st.button("▶️ Processar vídeo", type="primary", key="process_btn")
    with col2:
        if st.session_state.processing_result is not None:
            st.button(
                "🗑️ Limpar resultado",
                key="clear_btn",
                on_click=lambda: st.session_state.update({"processing_result": None}),
            )

    if process_button:
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        timer_placeholder = st.empty()
        _start_time = _time.time()

        try:
            with status_placeholder.container():
                st.info("⏳ Carregando modelos...")

            model, scaler, class_name_to_id, class_id_to_name, _ = _load_model_cached()

            expected_feature_count = TRAINING_WINDOW_SIZE * len(ANGLE_COLUMNS)
            actual_feature_count = getattr(scaler, "n_features_in_", None)
            if actual_feature_count is not None and actual_feature_count != expected_feature_count:
                st.error(
                    f"❌ Scaler espera {actual_feature_count} features, "
                    f"calculamos {expected_feature_count} "
                    f"(window={TRAINING_WINDOW_SIZE} × {len(ANGLE_COLUMNS)} ângulos)"
                )
                st.stop()

            window_size = TRAINING_WINDOW_SIZE

            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_video.name}") as tmp_file:
                tmp_file.write(uploaded_video.getbuffer())
                tmp_video_path = Path(tmp_file.name)

            is_valid, validation_msg = validate_video(tmp_video_path)
            if not is_valid:
                st.error(f"❌ Vídeo inválido: {validation_msg}")
                st.stop()

            pose_detector = _create_pose_detector(DEFAULT_POSE_MODEL_VARIANT, min_pose_confidence)

            def progress_callback(current_frame, total_frames, stage):
                progress = current_frame / max(total_frames, 1)
                progress_placeholder.progress(progress)
                elapsed = _time.time() - _start_time
                timer_placeholder.caption(f"⏱️ Tempo decorrido: {elapsed:.1f}s")
                status_placeholder.info(f"🔄 {stage}: Frame {current_frame}/{total_frames}")

            predictor = RandomForestVideoPredictor(
                model=model,
                scaler=scaler,
                class_name_to_id=class_name_to_id,
                class_id_to_name=class_id_to_name,
                pose_detector=pose_detector,
                window_size=window_size,
                process_fps=DEFAULT_PROCESS_FPS,
                angle_columns=ANGLE_COLUMNS,
                visibility_columns=VISIBILITY_COLUMNS,
                min_pose_detection_confidence=min_pose_confidence,
                min_pose_presence_confidence=min_pose_confidence,
                max_seconds=max_seconds,
                progress_callback=progress_callback,
                visualization_options=visualization_options,
            )

            result = predictor.process_video(str(tmp_video_path), output_dir=DEFAULT_OUTPUT_DIR)

            st.session_state.processing_result = result

            _elapsed = _time.time() - _start_time
            progress_placeholder.empty()
            status_placeholder.empty()
            timer_placeholder.empty()
            st.session_state.processing_elapsed = _elapsed

            st.success(f"✅ Processamento concluído em {_elapsed:.1f}s!")
            st.rerun()

        except Exception as exc:
            status_placeholder.empty()
            progress_placeholder.empty()
            timer_placeholder.empty()

            error_msg = str(exc)
            st.error(f"❌ Erro durante processamento: {error_msg}")

            if "Could not open video" in error_msg or "not open" in error_msg.lower():
                st.warning("O vídeo não pôde ser aberto. Tente reconverter para MP4 H.264.")
            elif "scaler" in error_msg.lower() or "feature" in error_msg.lower():
                st.warning("Erro no processamento de features. Verifique se o modelo foi treinado corretamente.")

            import traceback
            with st.expander("📋 Detalhes técnicos"):
                st.code(traceback.format_exc(), language="python")

    # ── Resultados ────────────────────────────────────────────────────────────
    if st.session_state.processing_result is not None:
        result = st.session_state.processing_result
        rep_counts = result.get("rep_counts", {})
        time_per_exercise = result.get("time_per_exercise", {})

        st.divider()

        elapsed = st.session_state.get("processing_elapsed")
        if elapsed is not None:
            mins = int(elapsed // 60)
            secs = elapsed % 60
            tempo_proc = f"{mins}m {secs:.1f}s" if mins > 0 else f"{secs:.1f}s"
            st.caption(f"⏱️ Tempo de processamento: **{tempo_proc}**")

        exercises = [
            ex for ex in ["agachamento", "flexao", "rosca_biceps", "descanso"]
            if ex in rep_counts or ex in time_per_exercise
        ]
        if exercises:
            st.subheader("🏅 Detalhes por exercício")
            cols = st.columns(len(exercises))
            for i, ex in enumerate(exercises):
                with cols[i]:
                    reps = rep_counts.get(ex, 0)
                    secs_ex = time_per_exercise.get(ex, 0.0)
                    mins_ex = int(secs_ex // 60)
                    s_ex = int(secs_ex % 60)
                    tempo_str = f"{mins_ex}m {s_ex:02d}s" if mins_ex > 0 else f"{s_ex}s"
                    icon = EX_ICONS.get(ex, "")
                    st.metric(
                        label=f"{icon} {_label(ex)}",
                        value=f"{reps} reps" if ex != "descanso" else tempo_str,
                        delta=tempo_str if ex != "descanso" else None,
                    )

        st.subheader("🎬 Vídeo com anotações")
        video_path = Path(result["output_video_path"])

        if video_path.exists() and video_path.stat().st_size > 0:
            with open(video_path, "rb") as video_file:
                video_bytes = video_file.read()

            st.video(video_bytes, format="video/mp4", width=300)

            st.download_button(
                label="💾 Baixar vídeo anotado",
                data=video_bytes,
                file_name=video_path.name,
                mime="video/mp4",
                key="download_btn",
            )
        else:
            st.error(f"❌ Arquivo de vídeo não encontrado ou está vazio: {video_path}")
