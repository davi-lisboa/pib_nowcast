# 🇧🇷 PIB Nowcasting - Brasil

> Nowcasting do PIB trimestral brasileiro com Modelo de Fatores Dinâmicos e visualização interativa via Streamlit.

---

## 📌 Visão Geral

Este projeto implementa um processo de **Nowcasting do Produto Interno Bruto (PIB)** brasileiro utilizando um **Modelo de Fatores Dinâmicos Mensais (Dynamic Factor Model - DFM)**. O objetivo é antecipar as variações do PIB antes da sua divulgação oficial, que ocorre com cerca de dois meses de atraso em relação ao trimestre de referência.

---

## 🧠 Motivação

O PIB é um dos principais indicadores macroeconômicos de um país, medindo a produção total de bens e serviços finais em um período. Porém, sua **divulgação tardia** dificulta decisões oportunas por parte de formuladores de políticas, analistas e investidores.

Para contornar esse atraso, este projeto utiliza **indicadores antecedentes** — divulgados com maior frequência e menor defasagem — como base para prever, em tempo real, o valor mais provável do PIB atual (ou iminente).

---

## 🔧 Metodologia

- Utiliza um **Modelo de Fatores Dinâmicos Mensais com estatísticas de estado** (`DynamicFactorMQ` via `statsmodels`)
- Integra séries mensais e trimestrais via abordagem "Mixed Frequency"
- Reduz dimensionalidade de (atualmente) 27 variáveis mensais a 4 fatores latentes
- Realiza previsões condicionais para o PIB trimestral

---

## 📦 Estrutura do Repositório
```
📁 pib_nowcasting/
├── model_creation.py # Coleta dados, compara versões e reestima o modelo se necessário
├── dashboard_creation.py # Script do dashboard Streamlit que consome o modelo e o dataset
├── initial_model.pkl.gz # Modelo estimado salvo (atualizado via GitHub Actions)
├── initial_dataset.pkl.gz # Dataset base salvo (idem)
├── requirements.txt # Dependências do projeto
├── runtime.txt # Versão do Python (para Streamlit Cloud)
└── .github/
└── workflows/
└── run_nowcast.yml # Workflow agendado que atualiza os arquivos automaticamente
```
---

## 🚀 Automação via GitHub Actions

O repositório conta com um workflow agendado para:
- Rodar o script `model_creation.py` nos dias úteis (segunda a sexta)
- Detectar mudanças no dataset ou revisões
- Atualizar e salvar os arquivos `.pkl.gz` (modelo e dados)
- Commitar automaticamente os novos arquivos no repositório
- Garantir que o app Streamlit consuma sempre as versões atualizadas

⏱️ Horários de execução:
- Segunda a sexta às 18:30 (horário de Brasília)

---

## 📊 Dashboard Interativo

O dashboard é desenvolvido em **Streamlit** e mostra:
- PIB observado e nowcastado
- Intervalos de confiança para o nowcast
- Resumo técnico do modelo estimado

⚙️ O app pode ser executado localmente com:
```
streamlit run dashboard_creation.py
```


Ou diretamente no Streamlit Cloud em [PIB Nowcast](https://pib-nowcast.streamlit.app/).

---

## 📈 Próximos passos

- Testes com diferentes configurações de fatores

- Inclusão de novas variáveis antecedentes

---

## Tecnologias Utilizadas

* ```statsmodels```

* ```pandas``` & ```numpy```

* ```streamlit```

* ```plotly```, ```matplotlib```, ```seaborn```

* ```python-bcb``` & ```sidrapy``` (coleta automática de dados)

---
## 📄 Licença
Este projeto é distribuído sob a Licença MIT.

---
## 🙋‍♂️ Autor
Davi Lisboa • [LinkedIn](https://www.linkedin.com/in/lisboadavi/) • [GitHub](https://github.com/davi-lisboa)
