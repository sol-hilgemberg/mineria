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
from fpdf import FPDF
import base64

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

# Balanceo con SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Crear y entrenar el modelo de Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
st.subheader('Reporte de Clasificación')
report = classification_report(y_test, y_pred, output_dict=True)
st.text(classification_report(y_test, y_pred))

# Mostrar la matriz de confusión
st.subheader('Matriz de Confusión')
fig3, ax3 = plt.subplots(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax3)
st.pyplot(fig3)

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
probabilidad = model.predict_proba(new_data)[0][1]

# Mostrar resultado
if prediction == 1:
    st.write(f"**¡Alerta! Se predice un fallo en la maquinaria con una probabilidad de {probabilidad:.2f}.**")
else:
    st.write(f"La maquinaria está funcionando normalmente. Probabilidad de fallo: {probabilidad:.2f}")

# Generar informe PDF
def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Informe de Predicción de Fallos en Maquinaria", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Horas de Maquinaria: {horas_maquina}", ln=True)
    pdf.cell(200, 10, txt=f"Temperatura: {temperatura}", ln=True)
    pdf.cell(200, 10, txt=f"Vibración: {vibracion}", ln=True)
    pdf.cell(200, 10, txt=f"Producción: {produccion}", ln=True)
    pdf.cell(200, 10, txt=f"Predicción: {'Fallo' if prediction == 1 else 'Normal'}", ln=True)
    pdf.cell(200, 10, txt=f"Probabilidad de fallo: {probabilidad:.2f}", ln=True)
    
    # Guardar archivo temporalmente
    pdf_file = "informe_prediccion.pdf"
    pdf.output(pdf_file)
    return pdf_file

if st.button("Descargar Informe en PDF"):
    pdf_file = generate_pdf()
    with open(pdf_file, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="informe_prediccion.pdf">Descargar Informe PDF</a>'
        st.markdown(href, unsafe_allow_html=True)
