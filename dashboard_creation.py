# %% Bibliotecas
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta

from bcb import sgs
import sidrapy as sidra

# import statsmodels.api as sm
# from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
# from sktime.utils.plotting import plot_series

import streamlit as st

import pickle
import gzip

import warnings
warnings.filterwarnings("ignore")

# %% Defnindo Funções

@st.cache_data
def get_bacen(series, start=None, end=None, max_tent=10):
    """
    Coleta série do SGS
    """
    tent = 1
    while tent <= max_tent:
        try:
            print('Tentativa', tent, ':', end='')
            data = sgs.get(series, start=start, end=end)
            print('✅')
            return data
        except Exception as e:
            print('❌', e)
            tent += 1
            continue
    raise Exception("Falha ao coletar dados do SGS após várias tentativas.") 

@st.cache_data
def load_dataset():
    with gzip.open("initial_dataset.pkl.gz", "rb") as f:
        return  pickle.load(f)

@st.cache_data
def load_model():
    with gzip.open("initial_model.pkl.gz", "rb") as f:
        return pickle.load(f)
# %% Importando dataset e modelo treinado

df_completo = load_dataset()
dfmq = load_model()

pib = df_completo[['pib']].resample('QE').last()
mensais = df_completo.drop(columns=['pib'])

pib_indice = get_bacen({'pib_indice_nsa': 22099},
                       start='1998-04-01'
                       ).assign(pib_yoy = lambda df: df.pct_change(4).multiply(100),
                                pib_acum4q = lambda df: df['pib_indice_nsa'].rolling(4).sum().divide(
                                    df['pib_indice_nsa'].shift(4).rolling(4).sum()).sub(1).multiply(100)
                                ).dropna()

# Ajusta datas para último mês do trimestre
pib_indice.index = pib_indice.index + pd.offsets.QuarterEnd(1)
pib_indice = pib_indice.asfreq('QE')

# %% Nowcasts

# Última data com dados
last_month_available = pib_indice.index.max()

# Estabelece data final de forecast, final do ano seguinte
future_month = dt.date(last_month_available.year + 1, 12, 31)

# Cria objeto com previsões médias e intervalos de confiança, insample e out-of-sample até @future_month
nowcast_obj = dfmq.get_prediction( end=future_month, information_set='smoothed')

# Cria dataframe com previsões YoY%
nowcast = nowcast_obj.predicted_mean['pib'].rename('pib_nowcast_yoy').to_frame()

# Adiciona intervalos de confiança para estimativa YoY%
# nowcast = nowcast.assign(
#     lower_pib=nowcast_obj.conf_int().loc[:, 'lower pib'],
#     upper_pib=nowcast_obj.conf_int().loc[:, 'upper pib']
# )

# Ajusta datas e dados para trimestrais
nowcast.index = nowcast.index.to_timestamp()
nowcast = nowcast.resample('QE').last()

# nowcast_confint = nowcast_obj.conf_int().loc[:, ['lower pib', 'upper pib']]

# Cria um dataframe com dados observados e as previsões YoY%
df_graficos = pib_indice.merge(nowcast, 
                        how='outer', 
                        left_index=True, 
                        right_index=True)

# Lógica para estender o índice até @future_month aplicando as taxas YoY%


pib_indice_nsa_nowcast =  (df_graficos['pib_indice_nsa'] # Índice do PIB
                           * (1 + df_graficos['pib_nowcast_yoy'].shift(-4)/100) # Taxa de YoY% 4 trimestres a frente
                           ).shift(4).rename('pib_indice_nsa_nowcast').to_frame() # Joga resultado 4 linhas para baixo

# O processo acima gera NaN já que os índices observados são limitados
# Todos os índices calculados para períodos onde existe dado real são substituídos por eles 

pib_indice_nsa_nowcast.loc[:last_month_available] = df_graficos[['pib_indice_nsa']].loc[:last_month_available].copy()

# O processo é feito novamente e agora se estende até o último período futuro
pib_indice_nsa_nowcast =  (pib_indice_nsa_nowcast['pib_indice_nsa_nowcast'] # Índice do PIB 
                           * (1 + df_graficos['pib_nowcast_yoy'].shift(-4)/100) # Taxa de YoY% 4 trimestres a frente
                           ).shift(4).rename('pib_indice_nsa_nowcast').to_frame() # Joga resultado 4 linhas para baixo


df_graficos = df_graficos.assign(
    # Adiciona índices futuros ao df
    pib_indice_nsa_nowcast = pib_indice_nsa_nowcast[['pib_indice_nsa_nowcast']],
    
    # Com os índices futuros disponíveis, agora é possível calcular a taxa acumulada em 4 trimestres
    pib_acum4q_nowcast = lambda df: (
        df['pib_indice_nsa_nowcast'].rolling(4).sum() 
        
        .divide(
                df['pib_indice_nsa_nowcast'].shift(4).rolling(4).sum()
                ).sub(1).multiply(100)
                                    )
    
).round(1)

# Previsões e cálculos insample são substituídos por NaN
df_graficos.loc[:last_month_available, 'pib_nowcast_yoy':] = np.nan

# %% Criação dash streamlit

st.title('Nowcast do PIB Real')
st.write('Este aplicativo apresenta o nowcast do PIB Real brasileiro, calculado com base em um modelo de fatores dinâmicos mensais (DFMQ).')

# st.sidebar.header('Ajustes do Gráfico')

# Filtro de Data
# data_inicio = (df_graficos.index.max() - pd.offsets.QuarterEnd(60)).to_pydatetime()
data_inicio = dt.datetime(df_graficos.index.max().year -15, 3, 31)
data_fim = df_graficos.index.max().to_pydatetime()

# intervalo_data = st.sidebar.slider(
#                             "Filtre o período",
#                             min_value=data_inicio,
#                             max_value=data_fim,
#                             value=(data_inicio, data_fim),
#                             # step=timedelta(days=30 )
#                             )

# df_graficos = df_graficos.loc[intervalo_data[0]:intervalo_data[1], :]
df_graficos = df_graficos.loc[data_inicio : data_fim, :]

# Filtro de colunas

indice_escolhido = st.selectbox(label="Tipo de índice", 
                                options=['PIB YoY%', 'Taxa Acumulada em 4 Trimestres', 'Índice'], )

indice_map = {'PIB YoY%': dict(base='pib_yoy', nowcast='pib_nowcast_yoy', alias_base='PIB YoY%', ),
               
              'Taxa Acumulada em 4 Trimestres': dict(base='pib_acum4q', nowcast='pib_acum4q_nowcast', 
                                                     alias_base='Taxa Acumulada em 4 Trimestres'), 

              'Índice': dict(base='pib_indice_nsa', nowcast='pib_indice_nsa_nowcast', alias_base='Índice')
              }

indice_base_escolhido = indice_map[indice_escolhido]['base']
nowcast_escolhido = indice_map[indice_escolhido]['nowcast']

# %% Criação dos Gráficos
st.plotly_chart(
    
px.line(
            df_graficos,
            x=df_graficos.index,
            # y=['pib_yoy', 'pib_nowcast_yoy', ],
            y = [
                indice_base_escolhido,
                nowcast_escolhido
            ],
            title='Nowcast do PIB Real',
            labels={'value': '%' if indice_escolhido != 'Índice' else 'Índice',
                     'index':'',
                     'variable': ''},
            markers=True,
            color_discrete_sequence=["#288BC5", 'orange'],
                
        ) \
        # Renomeia séries na legenda
        .update_traces(name=indice_escolhido, selector=dict(name=indice_base_escolhido)) \
        
        .update_traces(name='Nowcast', selector=dict(name=nowcast_escolhido)) \
        
        # Slider
        .update_layout(
        xaxis=dict(
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)
        # .add_scatter(
        #                 x=df_graficos.index,
        #                 y=df_graficos['lower_pib'],
        #                 mode='lines',
        #                 line=dict(color='lightgray', width=1, dash='dash'),
        #                 # fill='tonexty',
        #                 # fillcolor='rgba(211, 211, 211, 0.4)',
        #                 showlegend=False
        #             ) \
        # .add_scatter(
        #                 x=df_graficos.index,
        #                 y=df_graficos['upper_pib'],
        #                 mode='lines',
        #                 line=dict(color='lightgray', width=1, dash='dash'),
        #                 fill='tonexty',
        #                 # fillcolor='rgba(211, 211, 211, 0.4)',
        #                 showlegend=False
        #             )   
)
# %% Gráfico de Barras - Coeficiente de Determinação
justify_html_start = '<div style="text-align: justify">'
justify_html_end = '<div>'

st.write('## Coeficiente de Determinação (R²) dos Fatores em Relação ao PIB')

st.write(f'{justify_html_start}O R², também conhecido como coeficiente de determinação, é uma medida estatística que '
        f'avalia a qualidade do ajuste de um modelo de regressão.{justify_html_end}', unsafe_allow_html=True)
st.write('')
st.write(f'{justify_html_start}Ele indica a proporção da variação na variável dependente '
         f'(PIB em nosso caso) que pode ser prevista a partir das variáveis independentes (fatores latentes).{justify_html_end}',
          unsafe_allow_html=True)

st.plotly_chart(
px.bar(
    dfmq.coefficients_of_determination.loc[['pib']].round(2), 
    labels={'value':'R²', 'index':'', 'variable':'Fatores'},
    range_y=[0, 1],
    barmode='relative',
    facet_col='variable'

        )
        .update_xaxes(showticklabels=False)
        .add_annotation(
                        text="PIB",
                        xref="paper", yref="paper",
                        x=0.5, y=-0.15,
                        showarrow=False,
                        font=dict(size=14)
                        )
)

# %% News 

@st.cache_data
def load_news():
    with gzip.open("news.pkl.gz", "rb") as f:
        return pickle.load(f)
    
news = load_news()
st.write('### Impacto dos novos dados')
st.write(news)

# %% Tabela-Resumo do Modelo
st.write('### Resumo do Modelo')
st.write(dfmq.summary())

# %%
r2_df  = dfmq.coefficients_of_determination.loc[['pib']].round(4)
st.plotly_chart(
go.Figure(
            go.Waterfall(
                measure=['relative', 'relative', 'relative', 'relative'],
                x=[fator for fator in r2_df.columns],
                y = r2_df.loc['pib'],
                text = r2_df.loc['pib'],
                textposition = "outside",
                

            )
        )
)