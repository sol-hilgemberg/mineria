import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Simulación de datos
np.random.seed(42)
data = {
    'fecha': pd.date_range(start='2023-01-01', periods=1000, freq='D'),
    'horas_maquina': np.random.normal(100, 20, 1000),
    'temperatura': np.random.normal(70, 5, 1000),
    'vibracion': np.random.normal(0.5, 0.1, 1000),
    'produccion': np.random.normal(1500, 200, 1000),
    'fallo': np.random.choice([0, 1], size=1000, p=[0.85, 0.15])
}
df = pd.DataFrame(data)

# Convertir la columna 'fecha' a datetime y eliminarla del análisis
df['fecha'] = pd.to_datetime(df['fecha'])
df = df.drop(columns=['fecha'])

# Visualización de la distribución de 'fallo'
print(df['fallo'].value_counts())
sns.countplot(data=df, x='fallo', palette='Set2')
plt.title('Distribución de Fallos en Maquinaria')
plt.show()

# Verificar si hay valores nulos
print("Valores nulos en el dataset:\n", df.isnull().sum())

# Visualización de la distribución de 'horas_maquina'
plt.figure(figsize=(10, 6))
sns.histplot(df['horas_maquina'], kde=True, color='blue', bins=30)
plt.title('Distribución de Horas de Maquinaria')
plt.show()

# Visualización de la matriz de correlación
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación')
plt.show()

# Definir las características (X) y la variable objetivo (y)
X = df[['horas_maquina', 'temperatura', 'vibracion', 'produccion']]
y = df['fallo']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar SMOTE para sobremuestrear la clase minoritaria
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Crear y entrenar el modelo con Random Forest
model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10, random_state=42, class_weight='balanced')
model.fit(X_resampled, y_resampled)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# Graficar la matriz de confusión
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()

# Mostrar la importancia de las características
importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title('Importancia de las Características')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.show()

# Visualizar la distribución de las características del conjunto de entrenamiento
sns.pairplot(df[['horas_maquina', 'temperatura', 'vibracion', 'produccion', 'fallo']], hue='fallo')
plt.show()

# Evaluar el modelo sin usar SMOTE (en caso de comparar con los resultados anteriores)
model_no_smote = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model_no_smote.fit(X_train, y_train)

# Realizar predicciones y evaluar el modelo sin SMOTE
y_pred_no_smote = model_no_smote.predict(X_test)
print("Reporte de clasificación (sin SMOTE):\n", classification_report(y_test, y_pred_no_smote))
print("Matriz de confusión (sin SMOTE):\n", confusion_matrix(y_test, y_pred_no_smote))

# Graficar la matriz de confusión para el modelo sin SMOTE
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_no_smote), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión (Sin SMOTE)')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()
