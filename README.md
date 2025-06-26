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
- Reduz dimensionalidade de 27 variáveis mensais a 4 fatores latentes
- Realiza previsões condicionais para o PIB trimestral

---

## 📦 Estrutura do Repositório

