
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
import numpy as np

def media_movel(df, janela):
    colunas_angulos = [
    'right_cotovelo', 'left_cotovelo',
    'right_ombro', 'left_ombro',
    'right_joelho', 'left_joelho',
    'right_quadril', 'left_quadril'
    ]

    df_mm3 = df.copy()

    df_mm3[colunas_angulos] = (
        df[colunas_angulos]
        .rolling(window=janela, min_periods=1)
        .mean()
    )
    return df_mm3


def plotar_grafico_angulos(df_angulos, titulo,
                          articulacoes=('joelho', 'quadril', 'ombro', 'cotovelo'), fonte=12):

    sns.set_theme(style="whitegrid", context="paper")

    # 🔹 ordem fixa desejada
    ordem = ['joelho', 'quadril', 'ombro', 'cotovelo']
    articulacoes = [a for a in ordem if a in articulacoes]

    n = len(articulacoes)

    # 🔹 definir layout dinamicamente
    if n == 1:
        nrows, ncols = 1, 1
    elif n == 2:
        nrows, ncols = 1, 2
    elif n == 3:
        nrows, ncols = 3, 1  # melhor visual
    else:
        nrows, ncols = 2, 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6 if n <= 2 else 8),
                             sharex=True, sharey=True)

    # garantir que axes seja iterável
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = ['#1f77b4', '#d62728']
    y_min, y_max = 0, 180

    for i, art in enumerate(articulacoes):
        ax = axes[i]

        col_r = f'right_{art}'
        col_l = f'left_{art}'

        if col_r not in df_angulos.columns or col_l not in df_angulos.columns:
            ax.axis('off')
            continue

        sns.lineplot(ax=ax, data=df_angulos,
                     x='timestamp_s', y=col_r, color=colors[0])
        sns.lineplot(ax=ax, data=df_angulos,
                     x='timestamp_s', y=col_l, color=colors[1])

        ax.set_title(art.capitalize(), fontsize=fonte)
        ax.set_ylim(y_min, y_max)

        # labels inteligentes
        if i % ncols == 0:
            ax.set_ylabel('Ângulo (°)', fontsize=fonte)
        else:
            ax.set_ylabel('')

        if i >= (n - ncols):
            ax.set_xlabel('Tempo (s)', fontsize=fonte)
        else:
            ax.set_xlabel('')

        ax.grid(True, alpha=0.25, linestyle='--')
        sns.despine(ax=ax)

    # 🔹 remover eixos extras (caso n=3)
    for j in range(n, len(axes)):
        axes[j].axis('off')

    # legenda global
    handles = [
        plt.Line2D([0], [0], color=colors[0]),
        plt.Line2D([0], [0], color=colors[1])
    ]

    fig.legend(handles, ['Direito', 'Esquerdo'],
               loc = 'upper right',
                bbox_to_anchor=(0.98, 0.96),
               fontsize=fonte)

    fig.suptitle(titulo, fontsize=fonte * 1.3, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def detectar_repeticoes_exercicio(df_angles, articulacoes='cotovelo',
                                   tipo_deteccao='ambos',
                                   prominence_picos=4, distance_picos=3, 
                                   prominence_vales=3, distance_vales=3, fontsize=12):
    """
    Detecção e Análise de Ciclos de Movimento em Exercícios Físicos
    
    Permite análise de uma ou múltiplas articulações com visualização comparativa
    e detecção automática de picos/vales em séries temporais de ângulos.
    
    Parâmetros:
    -----------
    df_angles : pd.DataFrame
        Dataframe com ângulos articulares extraídos via MediaPipe
    
    articulacoes : str ou list
        Articulações a analisar. Exemplos:
        - 'cotovelo' (string simples)
        - ['cotovelo', 'joelho', 'ombro'] (lista)
    
    tipo_deteccao : str, default='ambos'
        Opções: 'picos', 'vales', 'ambos'
    
    prominence_picos : float ou list, default=4
        Proeminência mínima para picos. Se lista, deve ter mesmo tamanho que articulacoes
    
    distance_picos : int ou list, default=3
        Distância mínima entre picos. Se lista, deve corresponder a articulacoes
    
    prominence_vales : float ou list, default=3
        Proeminência mínima para vales. Se lista, deve corresponder a articulacoes
    
    distance_vales : int ou list, default=3
        Distância mínima entre vales. Se lista, deve corresponder a articulacoes
    
    fontsize : int, default=12
        Tamanho da fonte para os rótulos dos eixos e títulos dos subplots
    
    Retorno:
    --------
    dict : Dicionário com resultados para cada articulação
    """
    
    # Normalizar articulacoes para lista
    if isinstance(articulacoes, str):
        articulacoes = [articulacoes]
    
    # Normalizar hiperparâmetros para listas
    if not isinstance(prominence_picos, (list, tuple)):
        prominence_picos = [prominence_picos] * len(articulacoes)
    if not isinstance(distance_picos, (list, tuple)):
        distance_picos = [distance_picos] * len(articulacoes)
    if not isinstance(prominence_vales, (list, tuple)):
        prominence_vales = [prominence_vales] * len(articulacoes)
    if not isinstance(distance_vales, (list, tuple)):
        distance_vales = [distance_vales] * len(articulacoes)
    
    # Validar comprimentos
    if len(prominence_picos) != len(articulacoes):
        raise ValueError("prominence_picos deve ter mesmo tamanho de articulacoes")
    if len(distance_picos) != len(articulacoes):
        raise ValueError("distance_picos deve ter mesmo tamanho de articulacoes")
    if len(prominence_vales) != len(articulacoes):
        raise ValueError("prominence_vales deve ter mesmo tamanho de articulacoes")
    if len(distance_vales) != len(articulacoes):
        raise ValueError("distance_vales deve ter mesmo tamanho de articulacoes")
    
    # Criar coluna agregada para cada articulação
    df = df_angles.copy()
    for art in articulacoes:
        col_name = art.lower()
        col_left = f'left_{art.lower()}'
        col_right = f'right_{art.lower()}'
        
        if col_left not in df.columns or col_right not in df.columns:
            raise ValueError(f"Colunas não encontradas: {col_left}, {col_right}")
        
        df[col_name] = df[[col_left, col_right]].mean(axis=1)
    
    # Definir layout dinâmico
    n = len(articulacoes)
    if n == 1:
        nrows, ncols = 1, 1
    elif n == 2:
        nrows, ncols = 1, 2
    elif n == 3:
        nrows, ncols = 3, 1
    else:
        nrows, ncols = 2, 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6 if n <= 2 else 10), sharex=True, sharey=True)
    
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Armazenar resultados
    resultados = {}
    tem_picos = False
    tem_vales = False
    y_min = df[[f'left_{art.lower()}' for art in articulacoes] + [f'right_{art.lower()}' for art in articulacoes]].min().min() - 20
    y_max = df[[f'left_{art.lower()}' for art in articulacoes] + [f'right_{art.lower()}' for art in articulacoes]].max().max() + 20

    # Processar cada articulação
    for idx, art in enumerate(articulacoes):
        ax = axes[idx]
        col_name = art.lower()
        
        serie = df.set_index('timestamp_s')[col_name]
        
        # Detectar picos/vales com hiperparâmetros específicos
        picos = np.array([], dtype=int)
        vales = np.array([], dtype=int)
        
        if tipo_deteccao in ['picos', 'ambos']:
            picos, _ = find_peaks(serie, prominence=prominence_picos[idx], 
                                 distance=distance_picos[idx])
        
        if tipo_deteccao in ['vales', 'ambos']:
            vales, _ = find_peaks(-serie, prominence=prominence_vales[idx], 
                                 distance=distance_vales[idx])
        
        # Atualizar flags
        if len(picos) > 0:
            tem_picos = True
        if len(vales) > 0:
            tem_vales = True
        
        # Plotar série temporal
        ax.plot(serie.index, serie, linewidth=2.5, color='#2C3E50', label='Série Temporal')
        
        # Plotar picos
        if tipo_deteccao in ['picos', 'ambos'] and len(picos) > 0:
            ax.scatter(serie.index[picos], serie.iloc[picos], 
                      color='#E74C3C', s=fontsize*10, zorder=5, marker='^', 
                      edgecolors='darkred', linewidth=1.5)
            
            for i, p in enumerate(picos):
                ax.annotate(str(i), xy=(serie.index[p], serie.iloc[p]),
                           xytext=(0, 12), textcoords='offset points', 
                           ha='center', va='bottom', fontsize=fontsize*0.8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='#FADBD8', alpha=0.9),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5, color='#E74C3C'))

        
        # Plotar vales
        if tipo_deteccao in ['vales', 'ambos'] and len(vales) > 0:
            ax.scatter(serie.index[vales], serie.iloc[vales], 
                      color='#3498DB', s=fontsize*10, zorder=5, marker='v', 
                      edgecolors='darkblue', linewidth=1.5)
            
            for i, v in enumerate(vales):
                ax.annotate(str(i), xy=(serie.index[v], serie.iloc[v]),
                           xytext=(0, -12), textcoords='offset points', 
                           ha='center', va='top', fontsize=fontsize*0.8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='#D6EAF8', alpha=0.9),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5, color='#3498DB'))
        
        
        # Formatação do subplot
        ax.set_title(art.capitalize(), fontsize=fontsize, fontweight='bold')


        ax.set_ylim(y_min, y_max)
        
        # Labels inteligentes
        if idx % ncols == 0:
            ax.set_ylabel('Ângulo (°)', fontsize=fontsize)
        else:
            ax.set_ylabel('')
        
        if idx >= (n - ncols):
            ax.set_xlabel('Tempo (s)', fontsize=fontsize)
        else:
            ax.set_xlabel('')
        
        ax.grid(True, alpha=0.4, linestyle='--')
        sns.despine(ax=ax)
        
    # Remover axes extras
    for j in range(n, len(axes)):
        axes[j].axis('off')
    
    # Criar legenda global DINÂMICA baseada no que foi detectado
    handles = [
        plt.Line2D([0], [0], color='#2C3E50', linewidth=2.5)
    ]
    labels = ['Série Temporal']
    
    if tem_picos:
        handles.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#E74C3C', 
                       markersize=fontsize, markeredgecolor='darkred', markeredgewidth=1.5))
        labels.append('Máximos')
    
    if tem_vales:
        handles.append(plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#3498DB', 
                       markersize=fontsize, markeredgecolor='darkblue', markeredgewidth=1.5))
        labels.append('Mínimos')
    
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
               fontsize=fontsize, framealpha=0.95)
    
    # Título geral
    titulo = f'Detecção de Ciclos de Movimento - {", ".join([a.capitalize() for a in articulacoes])}'
    fig.suptitle(titulo, fontsize=fontsize*1.3, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    