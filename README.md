Objetivo del Proyecto

El objetivo de este proyecto es crear un dashboard interactivo usando Streamlit para predecir fallos en maquinaria industrial basándose en datos operacionales como:

    Horas de uso de la máquina
    Temperatura
    Vibración
    Producción

Este tipo de sistemas es común en la mantenimiento predictivo, donde se busca predecir fallos antes de que ocurran, lo cual puede ahorrar tiempo y dinero al evitar paradas imprevistas y costosas.
Flujo General del Proyecto

    Generación de Datos Simulados: Para este proyecto, se simulan datos de máquinas industriales. Los datos generados incluyen:
        horas_maquina: El tiempo en horas que la máquina ha estado en funcionamiento.
        temperatura: La temperatura de la máquina.
        vibracion: La vibración de la máquina, que puede ser un indicador de problemas mecánicos.
        produccion: La cantidad de producción realizada en la máquina.
        fallo: La variable dependiente que indica si la máquina ha fallado (1) o no (0). Este es el valor que intentamos predecir.

    Limpieza de los Datos: El dataset generado contiene la columna fecha, que no es útil para el modelo de predicción, por lo que la eliminamos.

    Visualización Exploratoria:
        Distribución de Horas de Maquinaria: Un histograma para mostrar cómo varían las horas de funcionamiento de la máquina. Esto puede ayudar a entender si las máquinas se usan más en ciertos rangos de tiempo.
        Matriz de Correlación: Para entender cómo las diferentes variables (horas de uso, temperatura, vibración y producción) se correlacionan entre sí. Esto es crucial para saber cuáles son las características más relevantes para predecir los fallos.

    Entrenamiento del Modelo: Usamos un modelo de Random Forest para la predicción de fallos en la maquinaria. Este modelo se entrena con un conjunto de datos que contiene las variables independientes (horas de máquina, temperatura, vibración y producción) y la variable dependiente fallo.

    Evaluación del Modelo:
        Reporte de Clasificación: Una vez que el modelo ha hecho las predicciones sobre los datos de prueba, mostramos métricas de desempeño como precisión, recall, F1-score, etc. Estos indican qué tan bien está funcionando el modelo.
        Matriz de Confusión: Esta visualización nos dice cuántos fallos fueron correctamente predichos, cuántos no fallos fueron correctamente predichos, etc.

    Ajuste del Modelo con SMOTE: Debido a que los datos tienen un desequilibrio en la clase fallo (es decir, hay muchos más casos de maquinaria funcionando normalmente que de fallos), utilizamos SMOTE (Synthetic Minority Over-sampling Technique). Esta técnica genera ejemplos sintéticos de la clase minoritaria (en este caso, los fallos) para balancear el conjunto de datos. Esto ayuda a mejorar el rendimiento del modelo para detectar fallos, especialmente cuando la clase minoritaria es muy pequeña.

    Predicción Interactiva: Finalmente, el dashboard permite al usuario ingresar valores para horas de máquina, temperatura, vibración y producción a través de sliders interactivos. Luego, el modelo realiza una predicción basada en esos valores y le indica al usuario si es probable que la maquinaria falle o no.

Componentes del Proyecto

    Interfaz de Usuario con Streamlit: Streamlit se usa para crear una interfaz de usuario sencilla y atractiva, en la que los usuarios pueden interactuar con los controles deslizantes y ver los resultados de las predicciones en tiempo real.

    Algunas secciones de la interfaz incluyen:
        Gráficos interactivos: Como la distribución de las horas de funcionamiento de la maquinaria o la matriz de correlación entre las características.
        Predicciones: Los usuarios pueden ingresar nuevos valores y recibir una predicción sobre si la máquina fallará o no.

    Modelo de Random Forest: El Random Forest es un modelo de aprendizaje supervisado que utiliza múltiples árboles de decisión para hacer predicciones. En este caso, lo usamos para predecir si una máquina fallará o no en función de sus características operativas.

    SMOTE (para balanceo de clases): El uso de SMOTE es especialmente útil cuando tienes un conjunto de datos desbalanceado, es decir, cuando una clase (en este caso, los fallos) está subrepresentada en comparación con la otra clase (la maquinaria que no falla). SMOTE genera ejemplos sintéticos para balancear las clases.

    Evaluación del Modelo: Después de entrenar el modelo, se evalúa utilizando:
        Reporte de clasificación: Para obtener métricas como precisión, recall, F1-score, etc.
        Matriz de Confusión: Una herramienta visual para evaluar la capacidad del modelo de clasificar correctamente las instancias de cada clase (fallo o no fallo).

Tecnologías Utilizadas

    Streamlit: Para crear la interfaz interactiva del dashboard.
    Pandas: Para manejar y manipular los datos.
    NumPy: Para la generación de números aleatorios y operaciones matemáticas.
    Scikit-learn: Para el modelo de predicción (Random Forest) y la evaluación del modelo.
    Seaborn y Matplotlib: Para la visualización estática de los datos.
    Plotly: Para la visualización interactiva de los gráficos.
    Imbalanced-learn (SMOTE): Para balancear el conjunto de datos con técnicas de sobremuestreo.

Interactividad en el Dashboard

El dashboard tiene varias características interactivas que permiten a los usuarios:

    Explorar los datos: Ver la distribución de las horas de funcionamiento, la correlación entre las variables, etc.
    Evaluar el modelo: Ver las métricas del modelo y cómo se comporta en los datos.
    Realizar predicciones: Los usuarios pueden ajustar los valores de las variables de entrada (horas de máquina, temperatura, vibración, y producción) y ver si el modelo predice un fallo o no.

Flujo de Uso del Dashboard

    Explorar los Datos: El usuario puede ver los gráficos interactivos sobre la distribución de las variables y la matriz de correlación.
    Ver el Rendimiento del Modelo: Después de entrenar el modelo, el usuario puede ver el reporte de clasificación y la matriz de confusión para entender cómo funciona el modelo.
    Ajustar SMOTE: El usuario tiene la opción de aplicar SMOTE para balancear los datos y mejorar la capacidad de predicción del modelo.
    Realizar Predicciones: Usando los sliders, el usuario puede ingresar valores y hacer que el modelo prediga si una nueva instancia de maquinaria fallará o no.

Posibles Mejoras

    Optimización del Modelo: Actualmente se usa un modelo de Random Forest, pero se podrían probar otros modelos como SVM, XGBoost o redes neuronales para mejorar la precisión.
    Integración con Datos Reales: En un entorno real, el modelo podría alimentarse de datos en tiempo real sobre el estado de la maquinaria.
    Notificaciones: Si el modelo predice un fallo, se podrían generar alertas o notificaciones para que los operarios puedan tomar acción antes de que ocurra el fallo.

Este proyecto es un ejemplo de cómo aplicar técnicas de machine learning y mantenimiento predictivo para crear un sistema que ayude a las empresas a evitar fallos en sus maquinarias, mejorando la eficiencia operativa y reduciendo costos.
