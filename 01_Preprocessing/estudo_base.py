from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt
import pandas as pd

# ler os dados
training = pd.read_csv('diabetes_dataset.csv')
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


training_no_NaN = training.dropna()
print("--- DESCRICAO SEM NaN ---")
print(training_no_NaN.describe())

# IMPUTANDO ZERO
imp_zeros = SimpleImputer(strategy="constant", fill_value=0, add_indicator=False) # imputador
training_imp_zeros = imp_zeros.fit_transform(training[feature_cols])
# print("--- DESCRICAO IMPUTACAO DE ZEROS ---")
# print(pd.DataFrame(training_imp_zeros, columns=[feature_cols]).describe())

# IMPUTANDO MEDIA
imp_mean = SimpleImputer() # imputador
training_imp_mean = imp_mean.fit_transform(training[feature_cols])
# print("--- DESCRICAO IMPUTACAO COM MEDIA ---")
# print(pd.DataFrame(training_imp_mean, columns=[feature_cols]).describe())

imp_knn = KNNImputer(n_neighbors=5)
training_imp_knn = imp_knn.fit_transform(training[feature_cols])
# print("--- DESCRICAO IMPUTACAO KNN ---")
# print(pd.DataFrame(training_imp_knn, columns=[feature_cols]).describe())

# IMPUTANDO MEDIANA
imp_median = SimpleImputer(strategy="median") # imputador
training_imp_median = imp_median.fit_transform(training[feature_cols])
print("--- DESCRICAO IMPUTACAO COM MEDIANA ---")
print(pd.DataFrame(training_imp_median, columns=[feature_cols]).describe())

# VERIFICANDO DISTRIBUICAO NORMAL

