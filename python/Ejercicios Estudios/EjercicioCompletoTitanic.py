#Importamos todas las librerias necesarias
import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import mitosheet
import pydotplus

from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score 
from sklearn.metrics import recall_score

from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data

#cargamos el dataset de entrenamiento y el de prueba y los combinamos
train_df = pd.read_csv('Titanic_train.csv')
test_df = pd.read_csv('Titanic_test.csv')
combine = [train_df, test_df]
dataset = sn.load_dataset("titanic") #este dataset importado de panda no es igual al que tenemos en csv

print('Columnas del dataset:')
print(train_df.keys())
print()


print('Primeras 20 filas del dataset:')
print(train_df.head(20))
print()

print('Ultimas filas del dataset:')
print(train_df.tail())
print()

print('Características del dataset:')
print(train_df.info())
print()

print('Descripcion de las variables numericas del dataset:')
print(train_df.describe())
print()

print('Descripcion de las variables categoricas del dataset:')
print(train_df.describe(include=['O']))
print()

#Uso libreria profiling para hacer analisis exploratorio antes de cambiar el dataset
# =============================================================================
# =============================================================================
# prof = ProfileReport(train_df)
# prof.to_file('titanic_original_train.html')
# # =============================================================================
# =============================================================================

#Uso libreria mito para analisis exploratorio solo en jupiter lab
#mitosheet.sheet(dataset, analysis_to_replay="id-xoswchqnxh")

#Analisis de correlacion entre variables y Survived que es nuestro target

print('Correlacion entre Pclass y Survived')
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print()

print('Correlacion entre Sex y Survived')
print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print()

print('Correlacion entre SibSp y Survived')
print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print()

print('Correlacion entre Parch y Survived')
print(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print()

#Analisis grafico de la relacion de las variables con Survived

#Age y Survived
grid = sn.FacetGrid(train_df, col='Survived')
grid.map(plt.hist, 'Age', bins=20)
np.warnings.filterwarnings('ignore') #ignoro warnings con valores nulos

#Age, Pclass y Survived
#grid = sn.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sn.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
np.warnings.filterwarnings('ignore')

#Analisis de variables categoricas Sex, Pclass y Survived
grid = sn.FacetGrid(train_df, col='Embarked')
#grid = sn.FacetGrid(train_df, row='embarked', size=2.2, aspect=1.6)
grid.map(sn.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
np.warnings.filterwarnings('ignore') #ignoro warnings con valores nulos

#Analisis de variables categoricas Sex, embarked y Survived
grid = sn.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
#grid = sn.FacetGrid(train_df, row='embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sn.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
np.warnings.filterwarnings('ignore') #ignoro warnings con valores nulos

#Borramos categorias del dataset que consideramos no importantes
train_df.pop('Ticket')
train_df.pop('Cabin')
test_df.pop('Ticket')
test_df.pop('Cabin')
#train_df.drop(['Ticket'], axis=1)
#test_df.drop(['Ticket','Cabin'], axis=1)
print('Características del dataset despues del drop:')
print(train_df.info())
print()

combine = [train_df, test_df]

#Extraigo los titulos de las nombres
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#Vemos que titulos corresponden a femenino y cuales a masculino
print('Titulos extraidos de los nombres y su relacionc con el sexo:')
print(pd.crosstab(train_df['Title'],dataset['Sex']))
print()

#Reemplazo los titulos de esa epoca por otros mas sencillos
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

print('Correlacion entre Title y Survived')
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print()

#Mapeo los titulos con numeros
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# =============================================================================
# print('Verificamos el dataset y vemos como quedo la columna titulo:')
# print(train_df.head() )
# print()
# =============================================================================

#Mapeamos la columna Sex con 0 para male y 1 para female
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#lo mismo podriamos realizarlo con la funcion get_dummies
#train_df = pd.get_dummies(train_df,columns=["Sex"],drop_first=True) 
#test_df = pd.get_dummies(test_df,columns=["Sex"],drop_first=True)

#Borramos columnas innecesarias
#train_df = train_df.drop(['Name','PassengerId'], axis=1)
#test_df = test_df.drop(['Name','PassengerId'], axis=1)
train_df.pop('Name')
train_df.pop('PassengerId')
test_df.pop('Name')
test_df.pop('PassengerId')
combine = [train_df, test_df]

# =============================================================================
# print('Siempre verificamos el dataset y vemos como quedaron las columnas')
# print(train_df.head() )
# print()
# =============================================================================

#Para completar los valores faltantes deAge, use números aleatorios entre la media y la desviación estándar, 
#basados ​​en conjuntos de combinaciones de Pclass y Sexo.
#Primero creamos la matriz para adivinarr
guess_ages = np.zeros((2,3))
guess_ages

#Iteraremos sobre la dimensión Sex (0 ó 1) y Pclass (1, 2, 3) para calcular los valores de Age para las 6 combinaciones.
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int(( age_guess/0.5 + 0.5 ) * 0.5)
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

# =============================================================================
# print('Siempre verificamos el dataset y vemos como quedaron las columnas')
# print(train_df.head() )
# print()
# =============================================================================

#Creamos una columna con la cantidad de integrantes de la familia combinando Parch y SibSp para luego poder borrarlas
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
train_df.pop('SibSp')
train_df.pop('Parch')
test_df.pop('SibSp')
test_df.pop('Parch')
combine = [train_df, test_df]
#Vemos la relacion entre la nueva categoria y survived
print('Correlacion entre FamilySize y Survived')
print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print()

#Creamos una columna donde pongamos si se encuentra solo o no asi podemos borrar la columna de tamaño de la familia
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

#Vemos la relacion entre la nueva categoria y survived
print('Correlacion entre IsAlone y Survived')
print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print()
train_df.pop('FamilySize')
test_df.pop('FamilySize')
combine = [train_df, test_df]

#Creamos una categoria combinando Pclass y Age
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))

#Todavia necesitamos sacar los nan que quedan en la columna Embarked, al ser solo dos registros usaremos el mas usado para completarlo
#averiguamos cual es el registro masrepetido en la columna embarked
freq_port = train_df.Embarked.dropna().mode()[0]
print('El valor mas usado en la columna Embarked es:')
print(freq_port)
print()

#completamos los nan con el valor mas repetido y mapeamos los valores de embarked con 0, 1 y 2
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

print('Correlacion entre Embarked y Survived')
print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print()

#Rellenamos los nan con el valor mas repetido
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

#Hacemos 4 bandas para dividir la tarifa
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

#vemos la relacion que tienen esas bandas con survived
print('Correlacion entre FareBand y Survived')
print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))
print()

#Mapeamos los 4 rangos de las bandas de tarifa con 0,1,2 y 3 y borramos la columna fareband
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

#comprobamos que en el dataset solo tengamos datos munericos y que no queden nan en ningun lado
print('Características del dataset de entrenamiento:')
print(train_df.info())
print()
print('Características del dataset de testeo:')
print(test_df.info())
print()

#sacamos los dataset a archivo para poder usarlo cuando queramos
#train_df.to_csv('train_out.csv', index=False)
#test_df.to_csv('test_out.csv', index=False)

#Una vez realizado el analisis exploratorio elegimos cuales son las variables a analizar y cual sera nuestro target

#Seleccionamos las columnas que queremos analizar
X = train_df.iloc[:, [1,2,3,4,5,6,7,8]].values

#Defino los datos correspondientes a la columna "resultado"
y = train_df.Survived

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#Defino el algoritmo a utilizar
#Naive Bayes
algoritmo = GaussianNB()

#Entreno el modelo
algoritmo.fit(X_train, y_train)

#Realizo una predicción
y_pred = algoritmo.predict(X_test)

#Verifico la matriz de Confusión
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión Bayes Ingenuo:')
print(matriz)
print()

sn.heatmap(matriz,annot=True)
plt.show()

#Exactitud - La exactitud (accuracy) mide el porcentaje de casos que el modelo ha acertado
print("Exactitud Bayes Ingenuo: ")
print(accuracy_score(y_test, y_pred))
print()

# Precision - Con la métrica de precisión podemos medir la calidad del modelo de machine learning en tareas de clasificación.
#Responde a la pregunta ¿qué porcentaje de lo identificado como positivo es realmente correcto?
print("Precision Bayes Ingenuo: ")
print(precision_score(y_test, y_pred))
print()

#rendimiento combinado de la precisión y la sensibilidad  
print("Rendimiento Bayes Ingenuo: ")
print(f1_score(y_test, y_pred))
print()

# Recall o Sensibilidad - ¿Qué porcentaje de los valores positivos fueron bien identificados?
print("Sensibilidad Bayes Ingenuo: ")
print(recall_score(y_test, y_pred))
print()

#Defino el algoritmo a utilizar
#Arbol de desicion con profundidad maxima de 4
algoritmo2 = DecisionTreeClassifier(random_state = 0, max_depth = 6)
algoritmo2.fit(X_train, y_train)

#Entreno el modelo
algoritmo2.fit(X_train, y_train)

#Realizo una predicción
y_pred = algoritmo2.predict(X_test)

#Verifico la matriz de Confusión
matriz2 = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión Arbol de desicion:')
print(matriz2)
print()

sn.heatmap(matriz2,annot=True)
plt.show()

#Exactitud - La exactitud (accuracy) mide el porcentaje de casos que el modelo ha acertado
print("Exactitud Arbol de desicion:")
print(accuracy_score(y_test, y_pred))
print()

# Precision - Con la métrica de precisión podemos medir la calidad del modelo de machine learning en tareas de clasificación.
#Responde a la pregunta ¿qué porcentaje de lo identificado como positivo es realmente correcto?
print("Precision Arbol de desicion:")
print(precision_score(y_test, y_pred))
print()

#rendimiento combinado de la precisión y la sensibilidad  
print("Rendimiento Arbol de desicion: ")
print(f1_score(y_test, y_pred))
print()

# Recall o Sensibilidad - ¿Qué porcentaje de los valores positivos fueron bien identificados?
print("Sensibilidad Arbol de desicion:")
print(recall_score(y_test, y_pred))
print()
#Visualizamos en un archivo
# =============================================================================
# tree.plot_tree(algoritmo2)
# =============================================================================
#Visualizamos en un archivo
dot_data = export_graphviz(algoritmo2,class_names = ['Pclass','Sex','Age','Fare','Embarked','Title','IsAlone','Age*Class'],filled = True, rounded = True,special_characters = True)
graph = graph_from_dot_data(dot_data)
graph.write_png('arbol_produndidad_6.png')

#Defino el algoritmo a utilizar
#Maquina de soporte vectorial kernel lineal
algoritmo3 = svm.SVC(kernel="linear")

#Entreno el modelo
algoritmo3.fit(X_train, y_train)

#Realizo una predicción
y_pred = algoritmo3.predict(X_test)

#Verifico la matriz de Confusión
matriz3 = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión Maquina soporte vectorial lineal:')
print(matriz3)
print()

# =============================================================================
sn.heatmap(matriz3,annot=True)
plt.show()
# =============================================================================

#Exactitud - La exactitud (accuracy) mide el porcentaje de casos que el modelo ha acertado
print("Exactitud Maquina soporte vectorial lineal::")
print(accuracy_score(y_test, y_pred))
print()

# Precision - Con la métrica de precisión podemos medir la calidad del modelo de machine learning en tareas de clasificación.
#Responde a la pregunta ¿qué porcentaje de lo identificado como positivo es realmente correcto?
print("Precision Maquina soporte vectorial lineal:")
print(precision_score(y_test, y_pred))
print()

#rendimiento combinado de la precisión y la sensibilidad  
print("Rendimiento Maquina soporte vectorial lineal: ")
print(f1_score(y_test, y_pred))
print()

# Recall o Sensibilidad - ¿Qué porcentaje de los valores positivos fueron bien identificados?
print("Sensibilidad Maquina soporte vectorial lineal:")
print(recall_score(y_test, y_pred))
print()

#Defino el algoritmo a utilizar
#Maquina de soporte vectorial kernel polinomio grado 2
algoritmo3 = svm.SVC(kernel='poly', degree=2)

#Entreno el modelo
algoritmo3.fit(X_train, y_train)

#Realizo una predicción
y_pred = algoritmo3.predict(X_test)

#Verifico la matriz de Confusión
matriz3 = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión Maquina soporte vectorial polinomio 2°:')
print(matriz3)
print()

# =============================================================================
sn.heatmap(matriz3,annot=True)
plt.show()
# =============================================================================

#Exactitud - La exactitud (accuracy) mide el porcentaje de casos que el modelo ha acertado
print("Exactitud Maquina soporte vectorial polinomio 2°:")
print(accuracy_score(y_test, y_pred))
print()

# Precision - Con la métrica de precisión podemos medir la calidad del modelo de machine learning en tareas de clasificación.
#Responde a la pregunta ¿qué porcentaje de lo identificado como positivo es realmente correcto?
print("Precision Maquina soporte vectorial polinomio 2°:")
print(precision_score(y_test, y_pred))
print()

#rendimiento combinado de la precisión y la sensibilidad  
print("Rendimiento Maquina soporte vectorial polinomio 2°: ")
print(f1_score(y_test, y_pred))
print()

# Recall o Sensibilidad - ¿Qué porcentaje de los valores positivos fueron bien identificados?
print("Sensibilidad Maquina soporte vectorial polinomio 2°:")
print(recall_score(y_test, y_pred))
print()

#Defino el algoritmo a utilizar
#Maquina de Discriminante  lineal
algoritmo4 = LinearDiscriminantAnalysis()

#Entreno el modelo
algoritmo4.fit(X_train, y_train)

#Realizo una predicción
y_pred = algoritmo4.predict(X_test)

#Verifico la matriz de Confusión
matriz4 = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión Discriminante lineal:')
print(matriz3)
print()

# =============================================================================
sn.heatmap(matriz4,annot=True)
plt.show()
# =============================================================================

#Exactitud - La exactitud (accuracy) mide el porcentaje de casos que el modelo ha acertado
print("Exactitud Discriminante lineal:")
print(accuracy_score(y_test, y_pred))
print()

# Precision - Con la métrica de precisión podemos medir la calidad del modelo de machine learning en tareas de clasificación.
#Responde a la pregunta ¿qué porcentaje de lo identificado como positivo es realmente correcto?
print("Precision Discriminante lineal:")
print(precision_score(y_test, y_pred))
print()

#rendimiento combinado de la precisión y la sensibilidad  
print("Rendimiento Discriminante lineal: ")
print(f1_score(y_test, y_pred))
print()

# Recall o Sensibilidad - ¿Qué porcentaje de los valores positivos fueron bien identificados?
print("Sensibilidad Discriminante lineal:")
print(recall_score(y_test, y_pred))
print()

#Defino el algoritmo a utilizar
#Maquina de Discriminante Cuadratico
algoritmo4 = LinearDiscriminantAnalysis(n_components=1                                                                  )

#Entreno el modelo
algoritmo4.fit(X_train, y_train)

#Realizo una predicción
y_pred = algoritmo4.predict(X_test)

#Verifico la matriz de Confusión
matriz4 = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión Discriminante lineal:')
print(matriz4)
print()

# =============================================================================
sn.heatmap(matriz4,annot=True)
plt.show()
# =============================================================================

#Exactitud - La exactitud (accuracy) mide el porcentaje de casos que el modelo ha acertado
print("Exactitud Discriminante lineal:")
print(accuracy_score(y_test, y_pred))
print()

# Precision - Con la métrica de precisión podemos medir la calidad del modelo de machine learning en tareas de clasificación.
#Responde a la pregunta ¿qué porcentaje de lo identificado como positivo es realmente correcto?
print("Precision Discriminante lineal:")
print(precision_score(y_test, y_pred))
print()

#rendimiento combinado de la precisión y la sensibilidad  
print("Rendimiento Discriminante lineal: ")
print(f1_score(y_test, y_pred))
print()

# Recall o Sensibilidad - ¿Qué porcentaje de los valores positivos fueron bien identificados?
print("Sensibilidad Discriminante lineal:")
print(recall_score(y_test, y_pred))
print()

#Defino el algoritmo a utilizar
#Maquina de Discriminante Cuadratico
algoritmo6 = QuadraticDiscriminantAnalysis()

#Entreno el modelo
algoritmo6.fit(X_train, y_train)

#Realizo una predicción
y_pred = algoritmo6.predict(X_test)

#Verifico la matriz de Confusión
matriz6 = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión Discriminante quadratico:')
print(matriz6)
print()

# =============================================================================
sn.heatmap(matriz6,annot=True)
plt.show()
# =============================================================================

#Exactitud - La exactitud (accuracy) mide el porcentaje de casos que el modelo ha acertado
print("Exactitud Discriminante quadratico:")
print(accuracy_score(y_test, y_pred))
print()

# Precision - Con la métrica de precisión podemos medir la calidad del modelo de machine learning en tareas de clasificación.
#Responde a la pregunta ¿qué porcentaje de lo identificado como positivo es realmente correcto?
print("Precision Discriminante quadratico:")
print(precision_score(y_test, y_pred))
print()

#rendimiento combinado de la precisión y la sensibilidad  
print("Rendimiento Discriminante quadratico: ")
print(f1_score(y_test, y_pred))
print()

# Recall o Sensibilidad - ¿Qué porcentaje de los valores positivos fueron bien identificados?
print("Sensibilidad Discriminante quadratico:")
print(recall_score(y_test, y_pred))
print()


#Defino el algoritmo a utilizar
#Redes Neuronales
algoritmo7 = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)

#Entreno el modelo
algoritmo7.fit(X_train, y_train)

#Realizo una predicción
y_pred = algoritmo7.predict(X_test)

#Verifico la matriz de Confusión
matriz7 = confusion_matrix(y_test, y_pred)
print('Matriz de Redes Neuronales:')
print(matriz7)
print()

# =============================================================================
sn.heatmap(matriz7,annot=True)
plt.show()
# =============================================================================

#Exactitud - La exactitud (accuracy) mide el porcentaje de casos que el modelo ha acertado
print("Exactitud Redes Neuronales:")
print(accuracy_score(y_test, y_pred))
print()

# Precision - Con la métrica de precisión podemos medir la calidad del modelo de machine learning en tareas de clasificación.
#Responde a la pregunta ¿qué porcentaje de lo identificado como positivo es realmente correcto?
print("Precision Redes Neuronales:")
print(precision_score(y_test, y_pred))
print()

#rendimiento combinado de la precisión y la sensibilidad  
print("Rendimiento Redes Neuronales: ")
print(f1_score(y_test, y_pred))
print()

# Recall o Sensibilidad - ¿Qué porcentaje de los valores positivos fueron bien identificados?
print("Sensibilidad Redes Neuronales:")
print(recall_score(y_test, y_pred))
print()

# Cada fila es un ejemplo de entrenamiento, cada columna es una característica  [X1, X2, X3]
print("X_train antes de array",X_train)
print("y_train antes de array",y_train)

#Defino el algoritmo a utilizar
#Redes Neuronales en numpy con funcion sigmoid

X=X_train;
y=np.array((y_train),dtype=float).reshape(712,1)
print("X despues de array",X)
print("y despue de array",y)

# Activación de la Función
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivada de la función Sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)

# Definición de la clase NeuralNetwork
class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x
        self.weights1= np.random.rand(self.input.shape[1],4) # Consideramos que tenemos 4 nodos en la capa oculta
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np. zeros(y.shape)
        
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2
        
    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
    
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()
        

NN = NeuralNetwork(X,y)
for i in range(1500): # entrenamos la red neuronal con 1500 iteraciones #1000 - #2000
    if i % 100 ==0: 
        print ("Iteración Número" + str(i) + "\n")
        #print ("Entrada: \n" + str(X))
        #print ("Salida Actual: \n" + str(y))
        #print ("Predicción de la Salida: \n" + str(NN.feedforward()))
        print ("Pérdida: \n" + str(np.mean(np.square(y - NN.feedforward())))) # media de la suma de la perdida de cuadrados
        print ("\n")
        
    NN.train(X,y)

print(NN)
#Verifico la matriz de Confusión
# =============================================================================
# matriz8 = confusion_matrix(y_test, NN)
# print('Matriz de Redes Neuronales:')
# print(matriz8)
# print()
# 
# # =============================================================================
# sn.heatmap(matriz8,annot=True)
# plt.show()
# # =============================================================================
# 
# #Exactitud - La exactitud (accuracy) mide el porcentaje de casos que el modelo ha acertado
# print("Exactitud Redes Neuronales:")
# print(accuracy_score(y_test, y_pred))
# print()
# 
# # Precision - Con la métrica de precisión podemos medir la calidad del modelo de machine learning en tareas de clasificación.
# #Responde a la pregunta ¿qué porcentaje de lo identificado como positivo es realmente correcto?
# print("Precision Redes Neuronales:")
# print(precision_score(y_test, y_pred))
# print()
# 
# #rendimiento combinado de la precisión y la sensibilidad  
# print("Rendimiento Redes Neuronales: ")
# print(f1_score(y_test, y_pred))
# print()
# 
# # Recall o Sensibilidad - ¿Qué porcentaje de los valores positivos fueron bien identificados?
# print("Sensibilidad Redes Neuronales:")
# print(recall_score(y_test, y_pred))
# print()
# =============================================================================
