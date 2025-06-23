## Projeto de nowcasting do PIB do Brasil.

O PIB é um dos principais indicadores macroeconomicos de um país, medindo tudo o que a economia produziu, em termos de bens e serviços finais, em um dado período no tempo.

A problemática se vale do fato do PIB fazer referência a todo um trimestre, tendo sua divulgação pouco mais de dois meses após o período de referência.

Com tamanho atraso precisamos recorrer aos sinais que indicadores antecedentes - de divulgação mais acelerada - podem nos oferecer.

A partir disso, elaborei um processo de Nowcasting por meio de um Modelo de Fatores Dinâmicos (DFM) para o PIB trimestral.

Atualmente o modelo conta com 27 variáveis mensais, reduzidas a 4 fatores, mais testes serão feitos.

Eventualmente será criado um processo de atualização automática dos dados e reestimação do modelo e das projeções. Ao final, será feito o deploy de um Dashboard de acompanhamento do modelo via Streamlit.
