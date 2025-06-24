# %% Bibliotecas
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta

from bcb import sgs
import sidrapy as sidra

import statsmodels.api as sm
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sktime.utils.plotting import plot_series

import streamlit as st

import warnings
warnings.filterwarnings("ignore")

# %% Coleta
# %%% Especificando séries

dados_mensais = {
                  # Output
                  'ind_extrativa':28504,
                  'ind_transformacao':28505,
                  'ind_bens_capital':28506,
                  'ind_bens_intermediarios':28507,
                  'ind_bens_consumo':28508,
                  'ind_construcao':28511,

                  'cons_energ_comercial':1402,
                  'cons_energ_indusrial':1404,

                  'pmc_ampliada': 28485,
                  'ibcbr_agro':29602,
                  'ibcbr_ind':29604,
                  'ibcbr_servicos':29606,

                  # Sentiment
                  'icc_fecomercio':4393,
                  'icea_fecomercio':4394,
                  'ief_fecomercio':4395,

                  'ics_fgv':20339,
                  'isas_fgv':20340,
                  'ies_fgv':20341,

                  # Employment
                  'caged_estoque':28784,
                  'renda_media':24399,
                  'tx_desemprego':24369,

                  # Prices
                  'ipca_12m':13522,
                  'ipca15': 7478,

                  'igpm':189,
                  'igpdi':190,
                  'incc':192,
                  'ipam':7450
              }

# %%% Especificando Fatores
fatores = {

            # Output
            'ind_extrativa':['Global', 'Output'],
            'ind_transformacao':['Global', 'Output'],
            'ind_bens_capital':['Global', 'Output'],
            'ind_bens_intermediarios':['Global', 'Output'],
            'ind_bens_consumo':['Global', 'Output'],
            'ind_construcao':['Global', 'Output'],

            'cons_energ_comercial':['Global', 'Output'],
            'cons_energ_indusrial':['Global', 'Output'],

            'pmc_ampliada': ['Global', 'Output'],
            'ibcbr_agro':['Global', 'Output'],
            'ibcbr_ind':['Global', 'Output'],
            'ibcbr_servicos':['Global', 'Output'],

            # Sentiment
            'icc_fecomercio':['Global', 'Sentiment'],
            'icea_fecomercio':['Global', 'Sentiment'],
            'ief_fecomercio':['Global', 'Sentiment'],

            'ics_fgv':['Global', 'Sentiment'],
            'isas_fgv':['Global', 'Sentiment'],
            'ies_fgv':['Global', 'Sentiment'],

            # Employment
            'caged_estoque':['Global', 'Employment'],
            'renda_media':['Global', 'Employment'],
            'tx_desemprego':['Global', 'Employment'],

            # Prices
            'ipca_12m':['Global', 'Prices'],
            'ipca15': ['Global', 'Prices'],

            'igpm':['Global', 'Prices'],
            'igpdi':['Global', 'Prices'],
            'incc':['Global', 'Prices'],
            'ipam':['Global', 'Prices'],

            # PIB ~ Output
            'pib': ['Global', 'Output']

        }

# %%% Coletando séries
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


pib = get_bacen({'pib':22099})
mensais = get_bacen(dados_mensais, start='1999-01-01')

# %% Tratando dados
# %%% PIB

# Ajusta datas para último mês do trimestre
pib.index = pib.index + pd.offsets.QuarterEnd(1)

# A variável está como índice, vamos calcular a taxa de crescimento real
# PIB Acumulado em 4 trimestres / PIB Acumulado em 4 trimestres no mesmo trimestre do ano anterior
# pib = pib.rolling(4).sum().divide(pib.rolling(4).sum().shift(4)).sub(1).multiply(100)

# A variável está como índice, vamos calcular a taxa de crescimento YoY%
# pib = pib.pipe(np.log).diff(4).multiply(100)
pib = pib.pct_change(4).multiply(100)


# %%% Mensais

# Ajusta datas para último dia do mês
mensais = mensais.resample('ME').last()

# Especifica quais séries vão sofre alteração
ln_diff_list = list()
for k, v in fatores.items():
  if v != ['Global', 'Prices']:
    if k == 'pib':
      continue
    ln_diff_list.append(k)

# Calcula log-diferença de 12 meses para as séries mensais especificadas, YoY%
mensais[ln_diff_list] = mensais[ln_diff_list].pipe(np.log).diff(12).multiply(100)

# %%% Juntando dados
cutoff = '2000-01-01'
df_completo = mensais.merge(pib, how='left', left_index=True, right_index=True).loc[cutoff:]


# %% Criando modelo

@st.cache_resource
def create_model(endog, k_endog_monthly, factors, factor_orders, idiosyncratic_ar1=True):
    """
    Cria o modelo de fatores dinâmicos mensais (DFMQ) com as especificações dadas.
    """
    dfmq = DynamicFactorMQ(
        endog=endog,
        k_endog_monthly=k_endog_monthly,  # Número de séries mensais
        factors=factors,  # Número de fatores
        factor_orders=factor_orders,  # Ordem dos fatores
        idiosyncratic_ar1=idiosyncratic_ar1  # AR(1) para os termos idiossincráticos
    ).fit()
    return dfmq
# Cria o modelo com as especificações
# Aqui, k_endog_monthly é o número de séries mensais, fatores é o número de fatores, 
# factor_orders é a ordem dos fatores, e 
# idiosyncratic_ar1 indica se os termos idiossincráticos seguem um modelo AR(1).  
# dfmq = DynamicFactorMQ(
#                         endog=df_completo,
#                         k_endog_monthly= 27,
#                         factors=4,
#                         factor_orders=3,
#                         idiosyncratic_ar1=True
#                     ).fit()

dfmq = create_model(
                    endog=df_completo,
                    k_endog_monthly= 27,
                    factors=4,
                    factor_orders=3,
                    idiosyncratic_ar1=True
                    )

# %% Nowcasts

steps = 15 - df_completo.index.max().month
nowcast_obj = dfmq.get_forecast(steps=steps)

nowcast = nowcast_obj.predicted_mean['pib'].rename('pib_nowcast').to_frame()

nowcast = nowcast.assign(
    lower_pib=nowcast_obj.conf_int().loc[:, 'lower pib'],
    upper_pib=nowcast_obj.conf_int().loc[:, 'upper pib']
)
nowcast = nowcast.resample('QE').last().round(1)
# nowcast_confint = nowcast_obj.conf_int().loc[:, ['lower pib', 'upper pib']]

df_graficos = pib.merge(nowcast, how='outer', left_index=True, right_index=True).loc[cutoff:]

# %% Criação dash streamlit

st.title('Nowcast do PIB Real')
st.write('Este aplicativo apresenta o nowcast do PIB Real brasileiro, calculado com base em um modelo de fatores dinâmicos mensais (DFMQ).')

st.sidebar.header('Ajustes do Gráfico')

# st.sidebar.slider

st.plotly_chart(px.line(
    nowcast,
    x=nowcast.index,
    y='pib_nowcast',
    title='Nowcast do PIB Real',
    labels={'pib_nowcast': 'YoY%', 'index':''},
    # name='Nowcast PIB',
    # range_y=[-10, 10]
).update_traces(
    mode='lines+markers',
    line=dict(color='orange', width=2),
    marker=dict(size=5)
).add_scatter(
    x=nowcast.index,
    y=nowcast['lower_pib'],
    mode='lines',
    line=dict(color='lightgray', width=1, dash='dash'),
    # name='Limite Inferior',
    # name='',
    # fill='tonexty',
    # fillcolor='rgba(211, 211, 211, 0.5)'  # Cor cinza claro com transparência
).add_scatter(
    x=nowcast.index,
    y=nowcast['upper_pib'],
    mode='lines',
    line=dict(color='lightgray', width=1, dash='dash'),
    # name='Limite Superior',
    name='Intervalo de Confiança',
    fill='tonexty',
    fillcolor='rgba(211, 211, 211, 0.5)'  # Cor cinza claro com transparência
   
).add_scatter(
    x=pib.index,
    y=pib['pib'].round(1),
    mode='lines+markers',
    line=dict(color='lightblue', width=1),
    marker=dict(size=5),
    name='PIB YoY%',
)
)

data_inicio = (pib.index.max() - pd.offsets.QuarterEnd(40)).to_pydatetime()
data_fim = df_graficos.index.max().to_pydatetime()#.strftime('%Y-%m-%d')
intervalo_data = st.sidebar.slider(
                            "Filtre o período",
                            min_value=data_inicio,
                            max_value=data_fim,
                            value=(data_inicio, data_fim),
                            # step=timedelta(days=30 )
                            )

df_graficos = df_graficos.loc[intervalo_data[0]:intervalo_data[1], :]

st.plotly_chart(px.line(
                            df_graficos,
                            x=df_graficos.index,
                            y=['pib', 'pib_nowcast', ],
                            title='Nowcast do PIB Real',
                            labels={'value': 'YoY%', 'index':''},
                            markers=True,
                            color_discrete_sequence=["#288BC5", 'orange'],
                                
                        )
                        .update_traces(name='PIB Real YoY%', selector=dict(name='pib'))
                        .update_traces(name='Nowcast PIB YoY%', selector=dict(name='pib_nowcast'))
                        .add_scatter(
                                        x=df_graficos.index,
                                        y=df_graficos['lower_pib'],
                                        mode='lines',
                                        line=dict(color='lightgray', width=1, dash='dash'),
                                        fill='tonexty',
                                        fillcolor='rgba(211, 211, 211, 0.5)',
                                        showlegend=False
                                    )
                        .add_scatter(
                                        x=df_graficos.index,
                                        y=df_graficos['upper_pib'],
                                        mode='lines',
                                        line=dict(color='lightgray', width=1, dash='dash'),
                                        fill='tonexty',
                                        fillcolor='rgba(211, 211, 211, 0.5)',
                                        showlegend=False
                                    )   
)

st.write('### Resumo do Modelo')
st.write(dfmq.summary())