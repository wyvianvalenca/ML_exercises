#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>

- Tratamento de zeros ocultos
- Undersampling (Balanceamento de classes)
- KNNImputer para preenchimento de NaN
@author: Aydano Machado <aydano.machado@gmail
- RobustScaler para escala
- Classificador: KNeighborsClassifier(n_neighbors=3)
"""

import sys
from datetime import datetime

import pandas as pd
import numpy as np
import requests

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

LOG_FILE = 'results.txt'

def log_result(desc: str, response: requests.Response, internal_score: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write("\n\n" + desc.upper())
        file.write(f"\n{datetime.now()}")
        file.write("\nCross Validation (Stratified):")
        file.write(f"\n\t {internal_score}")
        file.write("\nServidor:")
        for key, value in response.json().items():
            file.write(f"\n\t{key}: {value}")

print('\n - Lendo o ficheiro com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')

cols_with_zeros = ['Glucose', 'BloodPressure', 'BMI']
data[cols_with_zeros] = data[cols_with_zeros].replace(0, np.nan)

feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols].copy()
y = data.Outcome

print(' - Montando Pipeline Preditivo (Escala -> Imputação -> KNN)')
# Pipeline garante que não há fuga de dados (data leakage)
pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('imputer', KNNImputer(n_neighbors=5, weights='distance')),
    ('classifier', KNeighborsClassifier(n_neighbors=3))
])

print(' - Configurando Stratified K-Fold Cross Validation')
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

print(' - Testando o modelo com validação cruzada estratificada (10-fold)')
scores = cross_val_score(pipe, X, y, cv=skf)
internal_score = f'\tMean: {scores.mean():.4f} +/- {scores.std():.4f}'
print(f'Lista de exatidões (accuracies): {np.round(scores, 4)}')
print(internal_score)

enviar = input("\nEnviar predições para o servidor? (y/n) ")
if enviar != 'y':
    sys.exit("Envio abortado")

print("OK! Treinando com todos os dados e preparando envio...\n")
pipe.fit(X, y)

print(' - Lendo e processando os dados de envio (diabetes_app.csv)')
data_app = pd.read_csv('diabetes_app.csv')

# ATENÇÃO: Os dados do app recebem a mesma regra de zero = NaN
data_app[cols_with_zeros] = data_app[cols_with_zeros].replace(0, np.nan)
X_app = data_app[feature_cols].copy()

y_pred = pipe.predict(X_app)

URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"
DEV_KEY = "VNW"

data_send = {
    'dev_key': DEV_KEY,
    'predictions': pd.Series(y_pred).to_json(orient='values')
}
r = requests.post(url=URL, data=data_send)
print(" - Resposta do servidor:\n", r.text, "\n")

log = input("Deseja registar este resultado? (y/n) ")
if log == 'y':
    desc = input("Descreva esta tentativa: ")
    log_result(desc, r, internal_score)