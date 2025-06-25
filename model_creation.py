# %% Bibliotecas
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta

from bcb import sgs
import sidrapy as sidra

# import statsmodels.api as sm
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# from sktime.utils.plotting import plot_series

# import streamlit as st

import pickle

import warnings
warnings.filterwarnings("ignore")

# %% Funções
# %%% get_bacen
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

# %%% create_model

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

# %%% create_model_quarterly

def create_model_quarterly(endog, endog_quarterly, factors, factor_orders, idiosyncratic_ar1=True):
    """
    Cria o modelo de fatores dinâmicos mensais (DFMQ) com séries trimestrais.
    """
    dfmq = DynamicFactorMQ(
                            endog=endog,
                            endog_quarterly=endog_quarterly,  # Séries trimestrais
                            factors=factors,  # Número de fatores
                            factor_orders=factor_orders,  # Ordem dos fatores
                            idiosyncratic_ar1=idiosyncratic_ar1  # AR(1) para os termos idiossincráticos
                        ).fit()
    return dfmq

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
# %% importando .pkl's

with open('initial_model.pkl', 'rb') as file:
    initial_model = pickle.load(file)

with open('initial_dataset.pkl', 'rb') as file:
    initial_dataset = pickle.load(file)

# %%% Coletando séries

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
pib = pib.asfreq('QE')

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

pib = pib.loc[cutoff:]
mensais = mensais.loc[cutoff:]

df_completo = mensais.merge(pib, how='left', left_index=True, right_index=True)

# %% Verificando se o dataset é igual ao inicial
if df_completo.equals(initial_dataset):
   pass

else:
    print('Dataset atualizado. Salvando novo dataset...')
    df_completo.to_pickle('initial_dataset.pkl')
    print('Novo dataset salvo como initial_dataset.pkl')

# %% Criando modelo


# Cria o modelo com as especificações
# Aqui, k_endog_monthly é o número de séries mensais, fatores é o número de fatores, 
# factor_orders é a ordem dos fatores, e 
# idiosyncratic_ar1 indica se os termos idiossincráticos seguem um modelo AR(1).  
dfmq = create_model_quarterly(
                    endog=mensais.loc[:'2025-04', :],
                    endog_quarterly= pib,
                    factors=4,
                    factor_orders=3,
                    idiosyncratic_ar1=True
                    )


dfmq.predict()['pib']
# %%% Atualizando modelo

new_obs = df_completo.iloc[-1:, :].copy()
dist_next_quarter = (new_obs.index[0] + pd.offsets.QuarterEnd(1)).month - new_obs.index[0].month

extra_line = pd.DataFrame(columns=df_completo.columns, 
                          index=pd.date_range(start=new_obs.index[0], 
                                              freq='ME', 
                                              periods=dist_next_quarter+1,  
                                              inclusive='right')
                    )


new_obs = pd.concat([new_obs, extra_line], axis=0)
new_obs.asfreq(None )
# print(new_obs)

new_model = dfmq.append(endog = new_obs.loc[:, mensais.columns.to_list()],
            endog_quarterly = new_obs.loc[:, ['pib']].resample('Q').last(),
            )
print("Modelo atualizado!")
new_obs.info()
# new_model.save('updated_model.pkl')
# dfmq.save('initial_model.pkl')
# df_completo.to_pickle('initial_dataset.pkl')
# %% News


if new_obs[['pib']].iloc[-1].isna().values[0]: 
    news = dfmq.news(comparison=new_model, 
                            impacted_variable='pib', 
                            impact_date = new_obs.index[-1].strftime('%Y-%m'),
                            #  comparison_type='updated', 
                            # start='2025-05',
                            # periods=1
                            )

print(news.summary())
# %%

news.post_impacted_forecasts[['pib']]


new_model.predict(start='2025-05', end='2025-07')[['pib']]

new_model.forecast(3)['pib']








TESTES
# %%
dfmq.forecast(3)['pib']
dfmq2.predict(end='2025-07')['pib']
# %%

news_results.post_impacted_forecasts
# %%
pib = pib.asfreq('QE')
dfmq3 = DynamicFactorMQ(
                    endog=mensais.loc[cutoff:'2025-04', :],
                    endog_quarterly=pib.loc[cutoff:'2025-04', :],
                    factors=4,
                    factor_orders=3,
                    idiosyncratic_ar1=True
                    ).fit()

# %%
new_obs = df_completo.loc['2025-05':, :].copy()
extra_line = pd.DataFrame(columns=df_completo.columns, 
                          index=pd.date_range(start=new_obs.index[0], 
                                              freq='ME', 
                                              periods=5, inclusive='right'))
new_obs = pd.concat([new_obs, extra_line], axis=0)
# print(new_obs)

new_model = dfmq.append(endog = new_obs.loc[:, :'ipam'],
            endog_quarterly = new_obs.loc[:, ['pib']].resample('Q').last(),
            )
# %%
dfmq3.forecast(3)['pib']
# %%
new_model.forecast(3)['pib']
# %%
new_model.predict(start='2025-05', end='2025-09')['pib']
# %%
# new_model.summary()

print(dfmq3.news(comparison=new_model, 
            impacted_variable='pib',
            impact_date='2025-06',
            # comparison_type='updated',
            ).summary_details())
# %%
