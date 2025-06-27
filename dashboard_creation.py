# %% Bibliotecas
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta

from bcb import sgs
import sidrapy as sidra
# from model_creation import get_bacen

# import statsmodels.api as sm
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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

pib_indice = get_bacen({'pib_indice_nsa': 22099}
                       ).assign(pib_yoy = lambda df: df.pct_change(4).multiply(100),
                                pib_acum4q = lambda df: df['pib_indice_nsa'].rolling(4).sum().divide(
                                    df['pib_indice_nsa'].shift(4).rolling(4).sum()).sub(1).multiply(100)
                                )#.round(1)

# Ajusta datas para último mês do trimestre
pib_indice.index = pib_indice.index + pd.offsets.QuarterEnd(1)
pib_indice = pib_indice.asfreq('QE')

# %% Nowcasts

last_month_available = df_completo.index.max()
steps = 15 - df_completo.index.max().month
future_month = last_month_available + pd.offsets.MonthEnd(steps)
nowcast_obj = dfmq.get_prediction( end=future_month)

nowcast = nowcast_obj.predicted_mean['pib'].rename('pib_nowcast_yoy').to_frame()

nowcast = nowcast.assign(
    lower_pib=nowcast_obj.conf_int().loc[:, 'lower pib'],
    upper_pib=nowcast_obj.conf_int().loc[:, 'upper pib']
)
nowcast.index = nowcast.index.to_timestamp()
nowcast = nowcast.resample('QE').last()
# nowcast_confint = nowcast_obj.conf_int().loc[:, ['lower pib', 'upper pib']]

df_graficos = pib_indice.merge(nowcast, 
                        how='outer', 
                        left_index=True, 
                        right_index=True)
# df_graficos

# df_graficos['pib_indice_nsa'].applymap(lambda x: x*(1+x['pib_nowcast_yoy'].shift(-4)/100))

pib_indice_nsa_nowcast =  (df_graficos['pib_indice_nsa'] 
                           * (1 + df_graficos['pib_nowcast_yoy'].shift(-4)/100)
                           ).shift(4).rename('pib_indice_nsa_nowcast').to_frame()
# pib_indice_nsa_nowcast

# pib_indice_nsa_nowcast['pib_indice_nsa_nowcast'] =  [observed if observed != np.nan 
#                         else nowcast
#                         for observed, nowcast 
#                         in zip(df_graficos['pib_indice_nsa'].values, 
#                                pib_indice_nsa_nowcast['pib_indice_nsa_nowcast'].values)]

# pib_indice_nsa_nowcast['pib_indice_nsa_nowcast'] = (pib_indice_nsa_nowcast['pib_indice_nsa_nowcast'] 
#                                                     if df_graficos['pib_indice_nsa'].isnull()
#                                                     else df_graficos['pib_indice_nsa']
# )
pib_indice_nsa_nowcast.loc[:last_month_available] = df_graficos[['pib_indice_nsa']].loc[:last_month_available].copy()

df_graficos = df_graficos.assign(

    pib_indice_nsa_nowcast = pib_indice_nsa_nowcast[['pib_indice_nsa_nowcast']],
    # (df_graficos['pib_indice_nsa'] * (1 + df_graficos['pib_nowcast_yoy'].shift(-4)/100)).shift(4),
    
    pib_acum4_nowcast = lambda df: (
        df['pib_indice_nsa_nowcast'].rolling(4).sum()
        
        .divide(
                df['pib_indice_nsa_nowcast'].shift(4).rolling(4).sum()
                ).sub(1).multiply(100)
                                    )
    
).round(1)

# df_graficos.tail(10)

# %% Criação dash streamlit

st.title('Nowcast do PIB Real')
st.write('Este aplicativo apresenta o nowcast do PIB Real brasileiro, calculado com base em um modelo de fatores dinâmicos mensais (DFMQ).')

st.sidebar.header('Ajustes do Gráfico')

data_inicio = (df_graficos.index.max() - pd.offsets.QuarterEnd(40)).to_pydatetime()
data_fim = df_graficos.index.max().to_pydatetime()#.strftime('%Y-%m-%d')
intervalo_data = st.sidebar.slider(
                            "Filtre o período",
                            min_value=data_inicio,
                            max_value=data_fim,
                            value=(data_inicio, data_fim),
                            # step=timedelta(days=30 )
                            )

df_graficos = df_graficos.loc[intervalo_data[0]:intervalo_data[1], :]
# %% Criação dos Gráficos
st.plotly_chart(
    
px.line(
            df_graficos,
            x=df_graficos.index,
            y=['pib_yoy', 'pib_nowcast_yoy', ],
            title='Nowcast do PIB Real',
            labels={'value': 'YoY%', 'index':''},
            markers=True,
            color_discrete_sequence=["#288BC5", 'orange'],
                
        ) \
        .update_traces(name='PIB Real YoY%', selector=dict(name='pib_yoy')) \
        .update_traces(name='Nowcast PIB YoY%', selector=dict(name='pib_nowcast_yoy')) \
        .add_scatter(
                        x=df_graficos.index,
                        y=df_graficos['lower_pib'],
                        mode='lines',
                        line=dict(color='lightgray', width=1, dash='dash'),
                        # fill='tonexty',
                        # fillcolor='rgba(211, 211, 211, 0.4)',
                        showlegend=False
                    ) \
        .add_scatter(
                        x=df_graficos.index,
                        y=df_graficos['upper_pib'],
                        mode='lines',
                        line=dict(color='lightgray', width=1, dash='dash'),
                        fill='tonexty',
                        # fillcolor='rgba(211, 211, 211, 0.4)',
                        showlegend=False
                    )   
)

st.write('### Resumo do Modelo')
st.write(dfmq.summary())

# %%
