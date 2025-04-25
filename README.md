# Experimento: Classificação de Discurso de Ódio em Português com LSTM e XGBoost usando Partições Salvas

Este repositório contém a implementação do experimento já reailizado no https://github.com/Carlosbera7/ExperimentoBaseOriginal, utilizando as partições pré-processadas e divididas treino e teste, seguindo as diretrizes do experimento original, disponibilizadas em https://github.com/Carlosbera7/SalvarParticoes. 

## Descrição do Experimento
O experimento segue as etapas descritas no artigo:

1. **Extração de Features**:
   - Uso de embeddings pré-treinados GloVe com 300 dimensões para português.
   - Tokenização e padding das sequências de entrada.

2. **Classificação**:
   - Construção de um modelo LSTM com:
     - Camada de embedding usando os pesos dos embeddings GloVe.
     - Uma camada LSTM com 50 unidades e dropout.
     - Uma camada densa final com ativação sigmoidal para classificação binária.
   - Extração das representações da penúltima camada da LSTM como entrada para o XGBoost.

3. **XGBoost**:
   - Treinamento do XGBoost com os seguintes parâmetros ajustáveis via grid search:
     - `eta`: [0, 0.3, 1]
     - `gamma`: [0.1, 1, 10]

## Implementação
O experimento foi implementado em Python 3.6 utilizando as bibliotecas:
- TensorFlow
- NLTK
- Gensim
- Scikit-learn
- XGBoost

O script principal executa as seguintes etapas:
1. Carregamento das partições salvas.
2. Tokenização e padding das sequências de texto.
3. Carregamento dos embeddings GloVe.
4. Construção e treinamento do modelo LSTM.
5. Extração das representações intermediárias.
6. Treinamento e avaliação do XGBoost.
7. Busca de hiperparâmetros com validação cruzada.

## Resultados
Os resultados incluem:
- **Relatórios de métricas**: Precision, recall, f1-score e accuracy.
- **Melhores parâmetros do XGBoost** obtidos via grid search.

Exemplo de saída:
```

Melhores parâmetros: {'eta': 0.3, 'gamma': 0.1}
Melhor f1-score: 0.907074
              precision    recall  f1-score   support

           0       0.88      0.91      0.89       431
           1       0.68      0.60      0.64       136

    accuracy                           0.84       567
   macro avg       0.78      0.75      0.76       567
weighted avg       0.83      0.84      0.83       567
```
![GraficoOriginalParticoes](https://github.com/user-attachments/assets/22af5312-9a81-40e7-8300-febb130496a4)

## Estrutura do Repositório
- [`Scripts/ClassificadorOriginalParticoes.py`](https://github.com/Carlosbera7/ExperimentoOriginalParticoes/blob/main/Script/ClassificadorOriginalParticoes.py): Script principal para executar o experimento.
- [`Data/`](https://github.com/Carlosbera7/ExperimentoOriginalParticoes/tree/main/Data): Pasta contendo o conjunto de dados e o Embeddings GloVe pré-treinados (necessário para execução).
- [`Execução`](https://organic-broccoli-rqj9p9696wfwwqx.github.dev/): O código pode ser executado diretamente no ambiente virtual.


