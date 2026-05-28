"""RACE — entrada principal da aplicação."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="RACE - Análise de Exercícios",
    page_icon="🏋️",
    layout="wide",
)

pg = st.navigation(
    [
        st.Page("pages/resumo.py",   title="Resumo do Projeto",       icon="📋", default=True),
        st.Page("pages/Predicao.py", title="Predição de Exercícios",  icon="🎯"),
    ]
)
pg.run()

