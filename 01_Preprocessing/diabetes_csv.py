#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
"""
import sys
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, minmax_scale, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import cross_val_score

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests

print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')

# Criando par X (colunas independentes) e y (resultado) para algorítmo de aprendizagem de máquina.
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')

# Caso queira modificar as colunas consideradas basta alterar o array a seguir.
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols]
y = data.Outcome

# Criando imputadores de valores
print(' - Criando imputador')
imputador_zeros = SimpleImputer(strategy="constant", fill_value=0)
imputador_media = SimpleImputer(add_indicator=True)
imp_knn = KNNImputer()

# Criando escaladores
print(' - Criando escalador (normalizacao)')
min_max_scaler = MinMaxScaler()
statistics_scaler = RobustScaler()

# Criando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3) # classificador
# neigh.fit(X_scaled, y)

# Pipeline de pre processamento
pipe = make_pipeline(imp_knn, statistics_scaler, neigh)

# Testando com validacao cruzada na base de testes ('database')
scores = cross_val_score(pipe, X, y, cv=5)
print(scores)
print(f'Mean: {scores.mean()} +/- {scores.std()}')
enviar = input("Enviar? (y/n) ")

if enviar != 'y':
    sys.exit("Envio abortado")

print("OK! Enviando...\n")

# Treinando com a base de treino inteira
print(' - Treinando o modelo')
pipe.fit(X, y)

# Realizando previsões com o arquivo de teste ('app')
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('diabetes_app.csv')
data_app = data_app[feature_cols]
y_pred = pipe.predict(data_app)
# y_pred = neigh.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

DEV_KEY = "VNW"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")
