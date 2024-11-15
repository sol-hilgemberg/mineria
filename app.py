import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np

# Generar un conjunto de datos simulado
np.random.seed(42)
data = {
    'Mes': ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
            'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'],
    'Producción (toneladas)': np.random.randint(1000, 5000, 12),
    'Temperatura (°C)': np.random.uniform(60, 90, 12),
    'Vibración (m/s²)': np.random.uniform(0.2, 1.0, 12)
}
df = pd.DataFrame(data)

# Inicializar la aplicación Dash con Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout de la aplicación
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Interactividad en Dash: Análisis de Producción Minera"), width=12)),

    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='dropdown',
                options=[
                    {'label': 'Producción (toneladas)', 'value': 'Producción (toneladas)'},
                    {'label': 'Temperatura (°C)', 'value': 'Temperatura (°C)'},
                    {'label': 'Vibración (m/s²)', 'value': 'Vibración (m/s²)'}
                ],
                value='Producción (toneladas)',  # Valor por defecto
                style={'width': '100%'}
            ), 
            width=6
        ),
        dbc.Col(
            dcc.Slider(
                id='slider-range',
                min=df['Producción (toneladas)'].min(),
                max=df['Producción (toneladas)'].max(),
                step=100,
                value=df['Producción (toneladas)'].mean(),
                marks={i: str(i) for i in range(int(df['Producción (toneladas)'].min()), 
                                               int(df['Producción (toneladas)'].max()) + 500, 500)}
            ), 
            width=6
        )
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='graph'), width=12)
    ]),

    dbc.Row([
        dbc.Col(html.Div(id='hover-data', style={'marginTop': 20}), width=12),
        dbc.Col(html.Button("Descargar CSV", id="btn_csv"), width=2),
        dcc.Download(id="download-dataframe-csv")
    ])
], fluid=True)

# Callback para actualizar el gráfico y mostrar texto con el valor del slider
@app.callback(
    [dash.dependencies.Output('graph', 'figure'),
     dash.dependencies.Output('hover-data', 'children')],
    [dash.dependencies.Input('dropdown', 'value'),
     dash.dependencies.Input('slider-range', 'value')]
)
def update_graph_and_text(selected_column, selected_value):
    filtered_df = df[df['Producción (toneladas)'] <= selected_value]
    fig = px.scatter(filtered_df, x='Mes', y=selected_column, labels={'x': 'Mes', 'y': selected_column})
    text_output = f'Mostrando datos con producción menor o igual a {selected_value} toneladas'
    return fig, text_output

# Callback para descargar CSV
@app.callback(
    dash.dependencies.Output("download-dataframe-csv", "data"),
    [dash.dependencies.Input("btn_csv", "n_clicks")],
    prevent_initial_call=True
)
def download_csv(n_clicks):
    return dcc.send_data_frame(df.to_csv, "datos_produccion.csv")

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
