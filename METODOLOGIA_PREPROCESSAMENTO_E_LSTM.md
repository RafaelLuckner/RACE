# Estratégia de Preparação e Treinamento Temporal

Este documento descreve a estratégia adotada nos notebooks `0-preprocessamento_dataset.ipynb` e `4 - lstm_training.ipynb`. O objetivo é manter uma fonte de dados rastreável, evitar vazamento entre participantes e avaliar a capacidade de generalização do modelo para pessoas não vistas no treinamento.

## Visão Geral do Fluxo

```text
CSVs de landmarks brutos
        |
        v
Notebook 0: frames, ângulos, metadados e janelas
        |
        +--> files_processados/frames_processados_v1.csv
        +--> files_processados/janelas_w15_s1_v1.csv
        +--> files_processados/manifesto_dataset_v1.json
        |
        v
Notebook 4: sequências temporais, LOSO e CNN + BiLSTM
        |
        v
Artefatos LSTM em ml_models/
```

`files_brutos/` é a fonte imutável dos landmarks extraídos dos vídeos. Os notebooks de treinamento não devem reconstruir dados diretamente desses arquivos: a fonte canônica para os experimentos é `files_processados/`.

## Notebook 0: Pré-processamento do Dataset

### Objetivo

O notebook `0-preprocessamento_dataset.ipynb` transforma os CSVs brutos, em formato longo de landmarks do MediaPipe, em duas representações rastreáveis: frames com atributos calculados e janelas temporais achatadas. Essa separação permite que modelos com formatos de entrada diferentes compartilhem o mesmo conjunto de origem.

### Identificação e rastreabilidade

Para cada arquivo em `files_brutos/`, o notebook preserva metadados de origem e cria dois identificadores:

- `video_id`: nome completo do arquivo sem extensão. Identifica unicamente a gravação e impede que uma janela atravesse a fronteira entre vídeos.
- `participant_id`: prefixo do nome do arquivo antes do primeiro caractere `_`, normalizado para minúsculas. Identifica a pessoa que executou a atividade e é usado nos particionamentos de avaliação.

O notebook valida que cada `video_id` é único e mantém, entre outros campos, o nome e o caminho relativo do arquivo de origem, o rótulo da atividade e a regra de extração do participante. Assim, uma amostra processada pode ser rastreada até o CSV original.

### Atributos por frame

Para cada frame, são calculados oito ângulos articulares a partir dos landmarks de pose:

1. cotovelo direito e esquerdo;
2. ombro direito e esquerdo;
3. joelho direito e esquerdo;
4. quadril direito e esquerdo.

Também é armazenado um peso de visibilidade para cada ângulo, derivado da visibilidade dos landmarks que o compõem. Esses pesos não substituem os ângulos: eles são metadados de qualidade que permitem inspecionar a confiança da pose detectada. O dataset de frames ainda registra `valid_angle_count`, `mean_visibility`, `is_valid_frame`, timestamp e versão do dataset.

O resultado é `files_processados/frames_processados_v1.csv`. Ele é a entrada direta do notebook LSTM, pois conserva a ordem temporal e todos os identificadores necessários para formar sequências sem misturar gravações.

### Janelas temporais

Além dos frames, o notebook gera `files_processados/janelas_w15_s1_v1.csv` com janelas deslizantes de:

- tamanho: 15 frames;
- stride: 1 frame;
- frequência de referência: 5 FPS.

Cada janela contém os oito ângulos dos 15 frames, totalizando $15 \times 8 = 120$ valores, além de metadados como `window_id`, `video_id`, `participant_id`, rótulo, frames inicial e final e estatísticas de visibilidade. As janelas são criadas separadamente para cada vídeo; portanto, não há sequência que combine o fim de uma gravação com o início de outra.

O stride unitário aumenta a quantidade de exemplos e preserva transições curtas de movimento. Como consequência, janelas consecutivas do mesmo vídeo compartilham 14 dos 15 frames. Elas não devem ser distribuídas aleatoriamente entre treino e teste, pois isso produziria amostras quase idênticas nos dois conjuntos.

### Manifesto

O arquivo `files_processados/manifesto_dataset_v1.json` registra parâmetros e estatísticas da geração, como versão, colunas, janela, stride e fontes processadas. Ele funciona como referência de reprodutibilidade: qualquer alteração relevante em regras de cálculo, ordem de atributos ou configuração de janela deve gerar uma nova versão do dataset e do manifesto.

## Notebook 4: Treinamento CNN + BiLSTM

### Fonte e construção das sequências

O notebook `4 - lstm_training.ipynb` lê exclusivamente `frames_processados_v1.csv`. Para cada `video_id`, ordena os frames e monta novamente janelas de 15 frames com stride 1. A construção interna permite usar atributos temporais que não cabem no CSV achatado de janelas e garante que os limites dos vídeos sejam respeitados.

Cada timestep possui 22 atributos:

| Grupo | Quantidade | Descrição |
| --- | ---: | --- |
| Ângulos | 8 | Valores articulares calculados no notebook 0. |
| Velocidades | 8 | Diferença de cada ângulo em relação ao frame anterior. |
| Assimetrias | 4 | Diferenças direita-esquerda para cotovelos, ombros, joelhos e quadris. |
| Temporais | 2 | Posição normalizada do frame no vídeo e tempo em segundos. |

Logo, o tensor de entrada tem a forma $(n\_janelas, 15, 22)$. Os identificadores `video_id` e `participant_id` permanecem associados a cada janela para controle do particionamento e análise posterior.

### Particionamento por participante

A avaliação emprega Leave-One-Subject-Out (LOSO). Em cada fold, todas as janelas de um participante são reservadas para teste, enquanto as janelas dos demais participantes formam o treino. O grupo utilizado pelo `LeaveOneGroupOut` é `participant_id`.

Esse desenho responde à pergunta metodologicamente relevante: como o classificador se comporta para uma pessoa que não participou do treinamento? Ele também evita o vazamento causado por janelas sobrepostas do mesmo vídeo, pois um participante inteiro permanece em apenas um lado do split.

O notebook executa os folds LOSO para todos os participantes disponíveis. Além disso, apresenta um fold detalhado com `TEST_PARTICIPANT = "rodrigo"`: Rafael, Paulo e Letícia compõem o treino, enquanto Rodrigo é o conjunto de teste externo daquele experimento. Esse teste não é usado para selecionar época, ajustar scaler ou treinar pesos.

### Normalização e validação interna

O `StandardScaler` é ajustado exclusivamente nos frames das janelas de treino. Para isso, o tensor de treino é achatado de $(n, 15, 22)$ para duas dimensões, normalizado e restaurado à forma original. O mesmo scaler apenas transforma, sem novo ajuste, os dados de teste.

Durante o treinamento do fold detalhado, `validation_split=0.15` retira uma fração apenas do conjunto de treino para acompanhar `val_loss`, reduzir a taxa de aprendizado e aplicar early stopping. Essa validação interna não é a avaliação de generalização entre participantes e não deve ser apresentada como teste externo. Como há janelas sobrepostas dentro do treino, ela também pode conter sequências parecidas; sua função é regular o treinamento, não substituir o resultado LOSO.

### Arquitetura e treinamento

O classificador possui as seguintes etapas:

1. `Conv1D(64, kernel_size=3)`, para extrair padrões locais em frames consecutivos;
2. normalização em lote, max pooling e dropout;
3. `Bidirectional(LSTM(64, return_sequences=True))`, para modelar dependências temporais nos dois sentidos da janela;
4. `LSTM(32)`, para consolidar a sequência em uma representação;
5. camada densa com softmax para as quatro classes: `agachamento`, `descanso`, `flexao` e `rosca_biceps`.

O treinamento usa Adam, perda `categorical_crossentropy`, até 100 épocas, batch size 32, `EarlyStopping` monitorando `val_loss` com restauração dos melhores pesos e `ReduceLROnPlateau` quando a validação deixa de melhorar.

### Avaliação e artefatos

Para cada fold LOSO, o notebook mede acurácia e F1 ponderado. No fold detalhado, também apresenta relatório de classificação, matriz de confusão, acurácia por classe e uma análise adicional que exclui `descanso`, avaliando somente a discriminação entre exercícios ativos.

Após o treinamento do fold detalhado, são exportados em `ml_models/`:

- `lstm_4exercises_model.keras`: arquitetura e pesos;
- `lstm_4exercises_scaler.pkl`: scaler ajustado somente no treino daquele fold;
- `lstm_4exercises_label_map.pkl`: mapeamento entre índices e rótulos de classe.

Os três artefatos formam um contrato único. A inferência deve fornecer exatamente 15 frames com os mesmos 22 atributos na mesma ordem e aplicar o scaler salvo antes de chamar o modelo.

## Limitações e Cuidados de Interpretação

- O conjunto possui poucos participantes. O LOSO é mais adequado que um split aleatório, mas a variabilidade estimada ainda depende fortemente das pessoas e gravações disponíveis.
- Janelas de stride 1 são altamente correlacionadas. Métricas por janela não equivalem a métricas por vídeo ou por repetição de exercício; análises futuras podem agregar previsões temporalmente para essas unidades.
- Pesos de visibilidade são preservados como indicadores de qualidade no pré-processamento, mas não integram as 22 features atuais da LSTM. Eles podem fundamentar filtros ou ponderações em experimentos posteriores.
- Ao alterar a ordem dos ângulos, as regras de cálculo, a taxa de amostragem, o tamanho da janela ou as 22 features, é necessário regenerar o dataset, repetir a avaliação e exportar novos artefatos compatíveis.
