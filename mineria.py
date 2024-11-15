import pandas as pd
import numpy as np

# Simulamos un conjunto de datos sobre fallos de maquinaria y producción minera
np.random.seed(42)

# Simulación de datos
data = {
    'fecha': pd.date_range(start='2023-01-01', periods=1000, freq='D'),
    'horas_maquina': np.random.normal(100, 20, 1000),  # Horas de funcionamiento
    'temperatura': np.random.normal(70, 5, 1000),       # Temperatura de la maquinaria
    'vibracion': np.random.normal(0.5, 0.1, 1000),      # Vibración
    'produccion': np.random.normal(1500, 200, 1000),    # Producción diaria (toneladas)
    'fallo': np.random.choice([0, 1], size=1000, p=[0.85, 0.15])  # 0: no fallo, 1: fallo de maquinaria
}

# Crear un DataFrame
df = pd.DataFrame(data)

# Mostrar las primeras filas
print(df.head())


# Convertir la columna 'fecha' a tipo datetime (si no está ya)
df['fecha'] = pd.to_datetime(df['fecha'])

# Eliminar columnas no necesarias para el análisis (en este caso, 'fecha' no es relevante para modelar)
df = df.drop(columns=['fecha'])

# Verificar valores nulos
print(df.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# Visualización de distribuciones
plt.figure(figsize=(10, 6))
sns.histplot(df['horas_maquina'], kde=True, color='blue', bins=30)
plt.title('Distribución de Horas de Maquinaria')
plt.show()

# Visualización de correlaciones
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Definir las características (X) y la variable objetivo (y)
X = df[['horas_maquina', 'temperatura', 'vibracion', 'produccion']]
y = df['fallo']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

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
