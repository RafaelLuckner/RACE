---
title: RACE - Análise de Exercícios com IA
emoji: 🏋️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.55.0"
python_version: "3.11"
app_file: prediction_app/app.py
pinned: false
---

<!-- markdownlint-disable MD012 MD025 -->

# RACE - Reconhecimento de Atividades e Análise de Comportamento Corporal

Projeto de TCC focado em análise de pose corporal e classificação automática de exercícios físicos utilizando visão computacional e machine learning.

## 📋 Sobre o Projeto

O RACE é um sistema que detecta e classifica atividades físicas em vídeos. Ele usa **MediaPipe** para extrair landmarks corporais, calcula ângulos articulares e permite escolher entre os modelos **Random Forest** e **CNN + BiLSTM** para classificar janelas temporais.

### Exercícios Suportados

- 🏋️ **Rosca Direta** (Bicep Curl)
- 💪 **Flexão** (Push-ups)
- 🤸 **Agachamento** (Squats)
- 😴 **Descanso** (Rest)

## 🚀 Como Usar

### Instalação

1. **Clonar o repositório**

```bash
git clone https://github.com/RafaelLuckner/RACE.git
cd "RACE"
```

1. **Criar ambiente virtual**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# ou
source .venv/bin/activate  # Linux/Mac
```

1. **Instalar dependências**

```bash
pip install -r requirements.txt
```

### Executar Aplicações

#### 🎯 App de Predição

```bash
python -m streamlit run prediction_app/app.py
```

No menu lateral, selecione o modelo desejado antes de processar o vídeo:

- **Random Forest**: usa 15 frames e os 8 ângulos por frame, totalizando 120 features achatadas.
- **LSTM**: usa 15 frames com 22 features por timestep: 8 ângulos, 8 velocidades angulares, 4 assimetrias direita-esquerda e 2 atributos temporais.

No Windows, prefira `python -m streamlit` em vez de `streamlit`, pois políticas de Controle de Aplicativo podem bloquear o executável `streamlit.exe`.

#### 🎨 App Streamlit Completo (streamlit_app)

```bash
cd streamlit_app
python -m streamlit run streamlit_app/app.py
```

## 📊 Fluxo de Processamento do prediction_app

```text
[Vídeo de Entrada]
        ↓
[MediaPipe - Detecção de Pose]
        ↓
[Extração de Landmarks - 33 pontos corporais]
        ↓
[Cálculo de Ângulos Articulares - 8 ângulos]
        ↓
[Construção de janela temporal de 15 frames]
        ↓
[Extração de features conforme o modelo selecionado]
        ↓
[Normalização com StandardScaler salvo]
        ↓
[Modelo ML - Random Forest ou CNN + BiLSTM]
        ↓
[Classificação do Exercício]
        ↓
[Visualização + Anotações]
        ↓
[Vídeo de Saída + Estatísticas]
```

## 🎯 Ângulos Articulares Calculados

O sistema extrai e normaliza **8 ângulos principais**:

1. **Ombro Esquerdo** - Ângulo entre quadril, ombro e cotovelo
2. **Ombro Direito** - Ângulo entre quadril, ombro e cotovelo
3. **Cotovelo Esquerdo** - Ângulo entre ombro, cotovelo e pulso
4. **Cotovelo Direito** - Ângulo entre ombro, cotovelo e pulso
5. **Quadril Esquerdo** - Ângulo entre ombro, quadril e joelho
6. **Quadril Direito** - Ângulo entre ombro, quadril e joelho
7. **Joelho Esquerdo** - Ângulo entre quadril, joelho e tornozelo
8. **Joelho Direito** - Ângulo entre quadril, joelho e tornozelo

## 🧠 Modelos e Artefatos

Os artefatos de produção ficam em `ml_models/`. Cada modelo deve ser usado junto com o seu scaler e mapa de classes correspondentes.

| Modelo | Artefatos | Entrada de inferência |
| --- | --- | --- |
| Random Forest | `random_forest_4exercises.pkl`, scaler e mapa de classes | 15 frames x 8 ângulos = 120 features |
| CNN + BiLSTM | `lstm_4exercises_model.keras`, scaler e mapa de classes | tensor com forma `(amostras, 15, 22)` |

A ordem dos oito ângulos é parte do contrato de inferência. Ao mudar ordem, tamanho de janela, cálculo de atributos ou FPS, retreine o modelo e gere novos artefatos compatíveis.

## 📈 Preparação e Treinamento

O dataset canônico de treinamento é produzido antes dos experimentos:

1. `0-preprocessamento_dataset.ipynb` lê os CSVs de `files_brutos/`, identifica `video_id` e `participant_id`, calcula ângulos e gera os datasets em `files_processados/`.
2. `2-random_forest_training.ipynb` e `3-random_forest_velocity_features.ipynb` consomem os dados processados para os experimentos com Random Forest.
3. `4 - lstm_training.ipynb` consome `frames_processados_v1.csv`, monta sequências de 15 frames e avalia o LSTM com Leave-One-Subject-Out (LOSO).

O documento [METODOLOGIA_PREPROCESSAMENTO_E_LSTM.md](METODOLOGIA_PREPROCESSAMENTO_E_LSTM.md) detalha a rastreabilidade, as 22 features da LSTM, o protocolo LOSO e as limitações de janelas sobrepostas.

### Notebooks Disponíveis

#### `1-Análise_detalhada_landmarks.ipynb`

- Tratamento dos dados para geração de ângulos articulares
- Visualização dos ângulos durante a execução de diversos exercícios
- Contagem e visualização de repetições utilizando bibliotecas estatísticas

#### `2-random_forest_training.ipynb`

- Treinamento de Random Forest a partir das janelas processadas
- Avaliação por participante e exportação dos artefatos

#### `3-random_forest_velocity_features.ipynb`

- Experimento de Random Forest com velocidades angulares
- Avaliação comparativa usando o dataset processado

#### `4 - lstm_training.ipynb`

- Construção de tensores temporais com 15 frames e 22 features por timestep
- Treinamento de CNN + BiLSTM
- Avaliação Leave-One-Subject-Out e exportação do modelo, scaler e mapa de classes

