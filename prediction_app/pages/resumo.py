"""RACE — Resumo do Projeto."""

from __future__ import annotations

import streamlit as st

_EXERCISES_INFO = [
    ("💪", "Flexão de Braço"),
    ("🏋️", "Agachamento"),
    ("🦾", "Rosca Bíceps"),
]

with st.sidebar:
    st.markdown("## 🏋️ RACE")
    st.divider()
    st.markdown("#### Exercícios suportados")
    for icon, name in _EXERCISES_INFO:
        st.markdown(f"{icon} {name}")

# ── Conteúdo principal ────────────────────────────────────────────────────────
st.title("📋 Resumo do Projeto — RACE")
st.markdown(
    "**Reconhecimento e Avaliação Computacional de Exercícios**  \n"
    "Projeto de Visão Computacional — Universidade de Sorocaba (UNISO)"
)

st.divider()

col_desc, col_tech = st.columns(2, gap="large")

with col_desc:
    st.subheader("🎯 Objetivo")
    st.markdown(
        "Desenvolver um sistema capaz de **classificar automaticamente exercícios físicos** "
        "e **contar repetições** a partir de vídeos, sem necessidade de sensores ou "
        "equipamentos especiais — apenas a câmera do celular ou computador.\n\n"
        "O sistema é projetado para uso assistivo no acompanhamento de treinos, "
        "podendo servir de base para aplicações de personal trainer virtual, "
        "fisioterapia remota e análise de desempenho esportivo."
    )

    st.subheader("📦 Dados coletados")
    st.markdown(
        "- **4 participantes** (dados anonimizados)\n"
        "- Vídeos gravados com câmera comum (smartphone)\n"
        "- Processados a **5 FPS** para extração de landmarks\n"
        "- Classes balanceadas: flexão, agachamento, rosca bíceps e descanso\n"
        "- Janelas de **15 frames** com deslizamento de 1 frame (stride 1)"
    )

with col_tech:
    st.subheader("🔬 Pipeline técnico")
    st.markdown(
        "**1. Detecção de pose — MediaPipe Pose (Full)**\n"
        "- 33 landmarks corporais por frame\n"
        "- Coordenadas 2D (x, y) + visibilidade\n\n"
        "**2. Extração de ângulos articulares**\n"
        "- 8 ângulos bilaterais por frame:\n"
        "  cotovelo D/E · ombro D/E · joelho D/E · quadril D/E\n\n"
        "**3. Janela deslizante**\n"
        "- 15 frames × 8 ângulos = **120 features** por amostra\n\n"
        "**4. Classificador — Random Forest**\n"
        "- Treinado com StandardScaler\n"
        "- Saída: classe por frame\n\n"
        "**5. Contagem de repetições**\n"
        "- Detecção de transições de estado (fase descendente → ascendente)\n"
        "- Filtro de janela para evitar falsos positivos"
    )

st.divider()

st.subheader("Exercícios reconhecidos")
ex_cols = st.columns(4)
ex_details = [
    ("💪", "Flexão de Braço", "Posição prone. Articulações monitoradas: **cotovelo** e **ombro** (bilateral)."),
    ("🏋️", "Agachamento",     "Em pé. Articulações monitoradas: **joelho** e **quadril** (bilateral)."),
    ("🦾", "Rosca Bíceps",    "Em pé. Articulação monitorada: **cotovelo** (bilateral)."),
]
for col, (icon, name, desc) in zip(ex_cols, ex_details):
    with col:
        st.markdown(f"### {icon} {name}")
        st.markdown(desc)

st.divider()

st.subheader("🛠️ Tecnologias utilizadas")
tech_cols = st.columns(3)
with tech_cols[0]:
    st.markdown("**Visão Computacional**\n- MediaPipe Pose\n- OpenCV")
with tech_cols[1]:
    st.markdown("**Machine Learning**\n- scikit-learn (Random Forest)\n- NumPy / Pandas")
with tech_cols[2]:
    st.markdown("**Interface & Deploy**\n- Streamlit\n- Python 3.10+")
