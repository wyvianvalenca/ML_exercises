#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
"""
import sys
from datetime import datetime

import pandas as pd
import requests

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

LOG_FILE = 'results.txt'

def log_result(desc: str, response: requests.Response, internal_score: str) -> None:
    """ Saves the server's responde in a .txt file"""
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write("\n\n" + desc.upper())
        file.write(f"\n{datetime.now()}")
        file.write("\nCross Validation:")
        file.write(f"\n\t {internal_score}")
        file.write("Servidor:")
        for key, value in response.json().items():
            file.write(f"\n\t{key}: {value}")

print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')

# Criando par X (colunas independentes) e y (resultado) para algorítmo de aprendizagem de máquina.
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')

# Caso queira modificar as colunas consideradas basta alterar o array a seguir.
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols]
y = data.Outcome

# Criando imputadores de valores
print(' - Criando imputador')
imputador_zeros = SimpleImputer(strategy="constant", fill_value=0)
imputador_media = SimpleImputer()
imputador_mediana = SimpleImputer(strategy="median")
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
pipe = make_pipeline(imputador_mediana, statistics_scaler, neigh)

# Testando com validacao cruzada na base de testes ('database')
print(' - Testando o modelo com validacao cruzada')
scores = cross_val_score(pipe, X, y, cv=5)
print(f'\t{scores}')
internal_score: str = f'\tMean: {scores.mean()} +/- {scores.std()}'
print(internal_score)
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
print(" - Resposta do servidor:\n", r.text, "\n")

log = input("Deseja registrar esse resultado? (y/n) ")
if log == 'y':
    desc: str = input("Descreva essa tentativa: ")
    log_result(desc, r, internal_score)
