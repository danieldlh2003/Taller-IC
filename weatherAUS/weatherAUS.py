import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier

dataset = 'weatherAUS.csv'
data2 = pd.read_csv(dataset)

#nos deshacemos de datos nulos
data2.dropna(axis=0,how='any', inplace=True)

#limpiamos los datos
data2.drop(['Date', 'Sunshine', 'Evaporation', 'Humidity9am', 'Humidity3pm', 
            'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm',
            'Location', 'MinTemp', 'MaxTemp', 'WindSpeed9am', 'WindSpeed3pm',
            'WindGustDir', 'WindDir9am', 'WindDir3pm'], axis=1, inplace=True)

#convertimos categorias en numeros
data2.RainTomorrow.replace(['No', 'Yes'], [0, 1], inplace=True)
data2.RainToday.replace(['No', 'Yes'], [0, 1], inplace=True)

#dividimos los datos en dos
data2_train = data2[:27000]
data2_test = data2[27000:]

x = np.array(data2_train.drop(['RainTomorrow'], 1))
y = np.array(data2_train.RainTomorrow) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


x_test_out = np.array(data2_test.drop(['RainTomorrow'], 1))
y_test_out = np.array(data2_test.RainTomorrow)

#REGRESION LOGISTICA

# Seleccionar un modelo
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

# Entreno el modelo
logreg.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')

# MAQUINA DE SOPORTE VECTORIAL

# Seleccionar un modelo
svc = SVC(gamma='auto')

# Entreno el modelo
svc.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')

# ARBOL DE DECISIÓN

# Seleccionar un modelo
arbol = DecisionTreeClassifier()

# Entreno el modelo
arbol.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Decisión Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')

#VECINOS MAS CERCANOS

#eseleccionamos el modelo
kn = KNeighborsRegressor(n_neighbors=10)

#entrenamos el modelo
kn.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('VECINOS MAS CERCANOS')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {kn.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {kn.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {kn.score(x_test_out, y_test_out)}')

#RANDOM FOREST

#seleccionar un modelo
rf = RandomForestClassifier()

#entrenamos el modelo
rf.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Random Forest')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {rf.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {rf.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {rf.score(x_test_out, y_test_out)}')




