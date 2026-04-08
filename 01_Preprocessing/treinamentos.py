from sklearn.base import BaseEstimator
from typing_extensions import Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import cross_val_score, LeaveOneOut

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests

def train_and_score(desc, X, y, 
                    classifier: BaseEstimator, 
                    scaler: BaseEstimator | None = None, 
                    imputator: BaseEstimator | None = None):

    steps: list[tuple[str, BaseEstimator]] = []

    if scaler is not None:
        steps.append(("scaler", scaler))

    if imputator is not None:
        steps.append(('imputador', imputator))

    steps.append(('classifier', classifier))

    estimator: Pipeline = Pipeline(steps=steps)

    print(f'\n - Testando {desc}')

    scores = cross_val_score(estimator, X, y, cv=10)
    print(f'K-FOLD CROSS ----- \nMedia: {scores.mean():.3f} DP: {scores.std():.3f}')

    leave = cross_val_score(estimator, X, y, cv=LeaveOneOut())
    print(f'LEAVE ONE OUT ----- \nMedia: {leave.mean():.3f} DP: {leave.std():.3f}')


print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')

# Criando par X (colunas independentes) e y (resultado) para algorítmo de aprendizagem de máquina.
print(' - Criando X e y para o algoritmo de ML a partir do arquivo diabetes_dataset')

# Caso queira modificar as colunas consideradas basta alterar o array a seguir.
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols]
y = data.Outcome

# Criando par X e y sem NA nenhum
data_NaNfree = data.dropna() # cerca de 190 linhas
X_NaNfree = data_NaNfree[feature_cols]
y_NaNfree = data_NaNfree.Outcome

# Criando imputadores de valores
print(' - Criando imputadores')
imp_zeros = SimpleImputer(strategy="constant", fill_value=0, add_indicator=True) # imputador
imp_means = SimpleImputer() # imputador
imp_knn = KNNImputer()

# Criando escaladores (normalizacao)
min_max_scaler = MinMaxScaler()

# Criando um modelo
print(' - Criando classificador KNN')
clf_knn = KNeighborsClassifier(n_neighbors=3) # classificador

# Testes internos
# train_and_score('sem nenhum NaN', X_NaNfree, y_NaNfree, clf_knn)
# train_and_score('com zeros no lugar de NaN', X, y, clf_knn, imp_zeros)
# train_and_score('com a media no lugar de NaN', X, y, clf_knn, imp_means)
train_and_score('com KNN Imputer e Normalizacao, sem coluna insulina', X, y, clf_knn, min_max_scaler, imp_knn)
