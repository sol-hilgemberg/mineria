import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import plotly.express as px

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

# Eliminar la columna 'fecha' para el análisis
df = df.drop(columns=['fecha'])

# Página de configuración
st.title('Dashboard de Predicción de Fallos en Maquinaria')

# Verificar valores nulos
st.subheader("Revisión de datos nulos")
st.write(df.isnull().sum())

# Visualización de la distribución de 'horas_maquina' usando Plotly para interactividad
st.subheader('Distribución de Horas de Maquinaria')
fig1 = px.histogram(df, x='horas_maquina', nbins=30, title='Distribución de Horas de Maquinaria', marginal='box')
st.plotly_chart(fig1)

# Visualización de la matriz de correlación con Plotly
st.subheader('Matriz de Correlación')
fig2 = px.imshow(df.corr(), text_auto=True, title="Matriz de Correlación")
st.plotly_chart(fig2)

# Preparación del modelo
X = df[['horas_maquina', 'temperatura', 'vibracion', 'produccion']]
y = df['fallo']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
st.subheader('Reporte de Clasificación')
st.text(classification_report(y_test, y_pred))

# Mostrar la matriz de confusión
st.subheader('Matriz de Confusión')
fig3, ax3 = plt.subplots(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax3)
st.pyplot(fig3)

# Mostrar la importancia de las características
st.subheader('Importancia de las Características')
importances = model.feature_importances_
feature_names = X.columns
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names, ax=ax4)
st.pyplot(fig4)

# Opcional: Balanceo de clases con SMOTE
use_smote = st.checkbox("Aplicar SMOTE para Sobremuestreo", value=True)

if use_smote:
    st.subheader("Aplicación de SMOTE para Sobremuestreo")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    model.fit(X_resampled, y_resampled)

    # Realizar predicciones y evaluar el modelo con SMOTE
    y_pred_smote = model.predict(X_test)
    st.subheader('Reporte de Clasificación con SMOTE')
    st.text(classification_report(y_test, y_pred_smote))

    # Mostrar la matriz de confusión con SMOTE
    st.subheader('Matriz de Confusión con SMOTE')
    fig5, ax5 = plt.subplots(figsize=(6, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_smote), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax5)
    st.pyplot(fig5)

# Interactividad para hacer predicciones nuevas
st.subheader('Predicción de Fallos en Nueva Entrada')
horas_maquina = st.slider('Horas de Maquinaria', min_value=0, max_value=500, value=100)
temperatura = st.slider('Temperatura', min_value=50, max_value=100, value=70)
vibracion = st.slider('Vibración', min_value=0.0, max_value=1.0, value=0.5)
produccion = st.slider('Producción', min_value=1000, max_value=3000, value=1500)

# Crear un DataFrame para la predicción
new_data = pd.DataFrame({
    'horas_maquina': [horas_maquina],
    'temperatura': [temperatura],
    'vibracion': [vibracion],
    'produccion': [produccion]
})

# Realizar la predicción
prediction = model.predict(new_data)
if prediction == 1:
    st.write("¡Alerta! Se predice un fallo en la maquinaria.")
else:
    st.write("La maquinaria está funcionando normalmente.")
