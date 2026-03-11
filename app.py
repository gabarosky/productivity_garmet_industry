# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:34:36 2026

@author: gabri
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
from io import BytesIO



# ------------------------------------------------------------
# 1. Cargar artefactos (con caché)
# ------------------------------------------------------------
@st.cache_resource
def cargar_artefactos():
    pipeline = joblib.load('model/modelo_pipeline.pkl')
    explainer = joblib.load('model/explainer.pkl')
    with open('model/feature_metadata.json', 'r') as f:
        metadata = json.load(f)
    return pipeline, explainer, metadata

pipeline, explainer, metadata = cargar_artefactos()

# Extraer información de metadata
numeric_cols = list(metadata['numeric_ranges'].keys())
numeric_ranges = metadata['numeric_ranges']
cat_cols = list(metadata['categorical_categories'].keys())
cat_categories = metadata['categorical_categories']

# ------------------------------------------------------------
# 2. Definir la función del usuario (sin modificar)
# ------------------------------------------------------------
def plot_waterfall(sample, preprocessor, model, explainer, sample_name=''):
    ''' (misma función que antes, sin cambios) '''
    # transform the sample
    sample_transformed = preprocessor.transform(sample)
    feature_names = preprocessor.get_feature_names_out()
    sample_transformed_df = pd.DataFrame(sample_transformed, columns=feature_names)

    # calculate the prediction
    pred_class = model.predict(sample_transformed_df.iloc[[0]].values)[0]
    
    # calculate Shapley values
    shap_val = explainer.shap_values(sample_transformed_df, approximate=True)
    
    if isinstance(shap_val, list):
        shap_list = shap_val
    else:
        #change the format 3D (n_samples, n_features, n_classes) to list of n_classes arrays of shape (n_samples, n_features)
        shap_list = [shap_val[:, :, i] for i in range(shap_val.shape[2])]

    # Change names and data for visualization
    display_data = []
    display_names = []

    for feat_name in feature_names:
        base_name = feat_name.split("__")[-1]          # quit prefix num__ o cat__
        
        if base_name in sample.columns:
            # If the name is a feature from the original dataframe
            value = sample[base_name].iloc[0]
            display_data.append(value)
            display_names.append(f"{base_name}")
        else:
            # If the name isn't a feature from the original dataframe, then the name is name and value OHE
            parts = base_name.split('_', 1)
            display_data.append(parts[1])
            display_names.append(f"{parts[0]}")  
    
    # plot
    fig, axes = plt.subplots(2, 2, figsize=(25, 12))
    axes = axes.flatten()

    # base values for each class
    ev = explainer.expected_value
    
    for i, ax in enumerate(axes):
        
        plt.sca(ax)
        
        # calculate explanation for the sample
        exp = shap.Explanation(
            values=shap_list[i][0], 
            base_values=ev[i] if hasattr(ev, '__len__') else ev,
            data=np.array(display_data), 
            feature_names=display_names
        )

        # plot
        shap.plots.waterfall(exp, max_display=12, show=False)
        
        # format
        ax.set_title(f"Class {i}", fontsize=14, pad=15)
        ax.tick_params(axis='y', labelsize=8)
        for text in ax.texts:
            text.set_fontsize(7)

    # fit space between subplots
    plt.subplots_adjust(wspace=1.5, hspace=0.8)
    plt.suptitle(f"Waterfall plots for the sample {sample_name} - Prediction: class {pred_class}",
             fontsize=16, fontweight='bold', y=1.02)

    plt.show()
    
# ------------------------------------------------------------
# 3. Interfaz de usuario: sidebar dos columnas
# ------------------------------------------------------------    

# Configuración de la página
st.set_page_config(page_title="Predictor con SHAP", layout="wide")
st.title("🔮 Predicción interactiva con explicación SHAP (4 clases)")
# Expander con instrucciones (visible por defecto)
with st.expander("📘 Instrucciones", expanded=True):
    st.markdown("""
    1. Ajusta los valores de las variables en el panel izquierdo.
    2. Presiona el botón **Predecir** para calcular la clase y el gráfico Waterfall.
    3. El gráfico muestra la contribución de cada variable a la predicción para cada una de las 4 clases.
    4. Puedes modificar los valores y volver a presionar **Predecir** para actualizar.
    """)



# Crear el formulario de entrada en la barra lateral
st.sidebar.header("📊 Parámetros de entrada")
st.markdown("---")

# Diccionario para almacenar los valores ingresados
input_data = {}
    

# Widgets para numéricas (se colocan aquí, no en sidebar)
input_data['targeted_productivity'] = st.sidebar.slider(
    "**Targeted productivity**",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.01,
    format="%.2f"
)
input_data['smv'] = st.sidebar.slider(
    "**SMV**",
    min_value=2.9,
    max_value=54.56,
    value=15.26,
    step=0.01,
    format="%.2f"
)
input_data['wip'] = st.sidebar.slider(
    "**Work in Progress**",
    min_value=0,
    max_value=24000,
    value=1039,
    step=1,
    format="%.2f"
)
input_data['over_time'] = st.sidebar.slider(
    "**Over-time**",
    min_value=0,
    max_value=25920,
    value=3960,
    step=1,
    format="%.2f"
)
input_data['incentive'] = st.sidebar.slider(
    "**Incentive**",
    min_value=0,
    max_value=3600,
    value=0,
    step=1,
    format="%.2f"
)
input_data['idle_time'] = st.sidebar.slider(
    "**Idle Time**",
    min_value=0,
    max_value=300,
    value=0,
    step=1,
    format="%.2f"
)
input_data['idle_men'] = st.sidebar.slider(
    "**Idle Men**",
    min_value=0,
    max_value=45,
    value=0,
    step=1,
    format="%.2f"
)
input_data['no_of_style_change'] = st.sidebar.slider(
    "**No of Style Changes**",
    min_value=0,
    max_value=2,
    value=0,
    step=1,
    format="%.2f"
)
input_data['no_of_workers'] = st.sidebar.slider(
    "**No of Workers**",
    min_value=2,
    max_value=90,
    value=34,
    step=1,
    format="%.2f"
)
# Widgets para variables categóricas
for col in cat_cols:
    options = cat_categories[col]
    input_data[col] = st.sidebar.selectbox(
        f"**{col}**",
        options=options,
        index=0
    )


col_izquierda, col_derecha = st.columns([1, 3])

with col_izquierda:
    st.header("🔢 Clase predicha")
    # Placeholder para la predicción (se llenará después de presionar botón)
    prediccion_placeholder = st.empty()
    # Mostrar un texto provisional
    prediccion_placeholder.markdown("## _ _ _")
    
    
    

    
    
    # Botón para activar la predicción
    predecir = st.button("🔮 Predecir / Actualizar", type="primary")
    
    # # Mostrar valores actuales (opcional, para referencia)
    # with st.expander("📋 Ver valores seleccionados"):
    #     for col, val in input_data.items():
    #         st.write(f"{col}: {val}")

with col_derecha:
    st.header("🌊 Waterfall SHAP (4 clases)")
    # Placeholder para la imagen
    waterfall_placeholder = st.empty()
    # Mensaje inicial
    waterfall_placeholder.info("Presiona 'Predecir' para generar los gráficos")

# ------------------------------------------------------------
# 4. Lógica de predicción y generación de gráficos (solo al presionar el botón)
# ------------------------------------------------------------
if predecir:
    with st.spinner("Calculando predicción y SHAP..."):
        # Crear DataFrame con los datos ingresados
        df_input = pd.DataFrame([input_data])
        
        # Extraer componentes del pipeline
        preprocessor = pipeline.named_steps['preprocess']
        model = pipeline.named_steps['model']
        
        # Predicción
        prediccion = pipeline.predict(df_input)[0]
        
        # Cambiar backend de matplotlib para evitar ventanas emergentes
        import matplotlib
        matplotlib.use('Agg')
        
        # Llamar a la función del usuario (genera la figura)
        plot_waterfall(
            sample=df_input,
            preprocessor=preprocessor,
            model=model,
            explainer=explainer,
            sample_name='Muestra actual'
        )
        
        # Capturar la figura generada
        fig = plt.gcf()
        
        # Convertir la figura a bytes
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        imagen_bytes = buf.getvalue()
        plt.close(fig)  # liberar memoria
        
        # Actualizar los placeholders
        with col_izquierda:
            prediccion_placeholder.markdown(f"## {prediccion}")
        
        with col_derecha:
            waterfall_placeholder.image(imagen_bytes, use_container_width=True)
        
        # Opcional: guardar en session_state para persistencia (por si se recarga la página)
        st.session_state.prediccion = prediccion
        st.session_state.waterfall_bytes = imagen_bytes

# ------------------------------------------------------------
# 5. Si ya se había predicho antes (por session_state), mostrar los resultados al cargar la página
# ------------------------------------------------------------
else:
    if 'prediccion' in st.session_state and 'waterfall_bytes' in st.session_state:
        with col_izquierda:
            prediccion_placeholder.markdown(f"## {st.session_state.prediccion}")
        with col_derecha:
            waterfall_placeholder.image(st.session_state.waterfall_bytes, use_container_width=True)