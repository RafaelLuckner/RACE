# RACE - Reconhecimento de Atividades e Análise de Comportamento Corporal

Projeto de TCC focado em análise de pose corporal e classificação automática de exercícios físicos utilizando visão computacional e machine learning.

## 📋 Sobre o Projeto

O RACE é um sistema inteligente que detecta e classifica atividades físicas a partir de vídeos. Utiliza **MediaPipe** para extração de landmarks da pose corporal e modelos de **machine learning** (Random Forest) para classificação de exercícios em tempo real.

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

2. **Criar ambiente virtual**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/Mac
```

3. **Instalar dependências**
```bash
pip install -r requirements.txt
```

### Executar Aplicações

#### 🎯 App com Random Forest (prediction_app)
```bash
python -m streamlit run prediction_app/app.py
```

#### 🎨 App Streamlit Completo (streamlit_app)
```bash
cd streamlit_app
python -m streamlit run streamlit_app/app.py
```


## 📊 Fluxo de Processamento do prediction_app

```
[Vídeo de Entrada]
        ↓
[MediaPipe - Detecção de Pose]
        ↓
[Extração de Landmarks - 33 pontos corporais]
        ↓
[Cálculo de Ângulos Articulares - 8 ângulos]
        ↓
[Normalização MinMaxScaler]
        ↓
[Modelo ML - Random Forest ou LSTM]
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

## 📈 Análise Detalhada

### Notebooks Disponíveis

#### `1-Análise_detalhada_landmarks.ipynb`
- Tratamento dos dados para geração de ângulos articulares
- Visualização dos ângulos durante a execução de diversos exercícios
- Contagem e visualização de repetições utilizando bibliotecas estatísticas

#### `2-random_forest_training.ipynb`
- Preparação de dados
- Normalização de features
- Treinamento do Random Forest
- Avaliação de métricas
- Salvamento do modelo

