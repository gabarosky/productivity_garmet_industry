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
# 1. Load artifacts
# ------------------------------------------------------------
@st.cache_resource
def cargar_artefactos():
    pipeline = joblib.load('model/modelo_pipeline.pkl')
    explainer = joblib.load('model/explainer.pkl')
    # with open('model/feature_metadata.json', 'r') as f:
    #     metadata = json.load(f)
    # return pipeline, explainer, metadata
    return pipeline, explainer

# pipeline, explainer, metadata = cargar_artefactos()
pipeline, explainer = cargar_artefactos()

# Extraer información de metadata
# numeric_cols = list(metadata['numeric_ranges'].keys())
# numeric_ranges = metadata['numeric_ranges']
# cat_cols = list(metadata['categorical_categories'].keys())
# cat_categories = metadata['categorical_categories']

# ------------------------------------------------------------
# 2. function for waterfalls
# ------------------------------------------------------------
def plot_waterfall(sample, preprocessor, model, explainer):
    ''' This function take a sample of data, calculate Shapley values and plot a waterfall for 
    category in order to show the impact of each eature in the final prediction.
    Variables:
    sample: a dataframe with the sample of variables to predict and analize.
    model: the model used for predictions
    prepreocessor: the preprocessor used for transform the data.
    explainer: the Shapley explainer trained with data and model.'''
    # transform the sample
    sample_transformed = preprocessor.transform(sample)
    feature_names = preprocessor.get_feature_names_out()
    sample_transformed_df = pd.DataFrame(sample_transformed, columns=feature_names)

    # calculate the prediction
    # pred_class = model.predict(sample_transformed_df.iloc[[0]].values)[0]
    
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
    plt.show()
    
    
# ------------------------------------------------------------
# 3.  Layout setup: sidebar and two columns
# ------------------------------------------------------------
st.set_page_config(page_title="Prediction with SHAP", layout="wide")
st.title("🧵 Garment productivity prediction and interactive setting")
# Expander with instructions
with st.expander("📘 Instructions", expanded=False):
    st.markdown("""
    1. Set the variable values in the left sidebar.
    2. Click the **Predict / update** button to generate a productivity prediction and a Waterfall chart explanation.
    3. The plot displays the contribution of each variable to each class.
    4. You can adjust the values and click **Predict / update** again to obtain an updated prediction.
    """)


# Create the input form at sidebar
st.sidebar.header("📊 Input settings")
st.markdown("---")

# Dict for the values
input_data = {}
   
# Slider for numericals
input_data['targeted_productivity'] = st.sidebar.slider(
    "**Targeted productivity**",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.01,
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
input_data['smv'] = st.sidebar.slider(
    "**SMV**",
    min_value=2.9,
    max_value=55.0,
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
# Select box for categorical 
input_data['no_of_style_change'] = st.sidebar.selectbox(
    "**No. of style changes**",
    options=[0,1,2],
    index=0
)
input_data['day'] = st.sidebar.selectbox(
    "**Day**",
    options=['Monday','Tuesday','Wednesday','Thursday','Saturday','Sunday'],
    index=0
)
input_data['quarter'] = st.sidebar.selectbox(
    "**Quarter**",
    options=[1,2,3,4,5],
    index=0
)
input_data['department'] = st.sidebar.selectbox(
    "Department",
    options=['finishing', 'sewing'],
    index=0
)
input_data['team'] = st.sidebar.selectbox(
    "Team",
    options=[1,2,3,4,5,6,7,8,9,10,11,12],
    index=0
)
# # Widgets para variables categóricas
# for col in cat_cols:
#     options = cat_categories[col]
#     input_data[col] = st.sidebar.selectbox(
#         f"**{col}**",
#         options=options,
#         index=0
#     )


left_col, right_col = st.columns([1, 3])

with left_col:
    st.header("Predicted class")
    # Placeholder for prediction
    prediction_placeholder = st.empty()
    # Provisory text
    prediction_placeholder.markdown("## _ _ _")
    
    

    
    
    # Button for activate prediction
    predecir = st.button("Predict / update", type="primary")
    
    # # Mostrar valores actuales (opcional, para referencia)
    # with st.expander("📋 Ver valores seleccionados"):
    #     for col, val in input_data.items():
    #         st.write(f"{col}: {val}")

with right_col:
    st.header("Waterfall explanatory for classes")
    # Placeholder para la imagen
    waterfall_placeholder = st.empty()
    # Mensaje inicial
    waterfall_placeholder.info("Click 'Predict / update' for generate plots")

# ------------------------------------------------------------
# 4. Prediction and plot generation
# ------------------------------------------------------------
if predecir:
    with st.spinner("Calculating predictions and SHAP..."):
        # Create DataFrame with the input data
        df_input = pd.DataFrame([input_data])
        
        # extract the preprocessor from pipeline
        preprocessor = pipeline.named_steps['preprocess']
        model = pipeline.named_steps['model']
        
        # make prediction
        prediction = pipeline.predict(df_input)[0]
        
        # change the matplotlib backend for avoiding emengent windows
        import matplotlib
        matplotlib.use('Agg')
        
        # call the function for waterfalls
        plot_waterfall(
            sample=df_input,
            preprocessor=preprocessor,
            model=model,
            explainer=explainer
        )
        
        # capture the figure
        fig = plt.gcf()
        
        # convert the figure to bytes
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        imagen_bytes = buf.getvalue()
        plt.close(fig)  # free memory
        
        # Update the placeholders
        with left_col:
            prediction_placeholder.markdown(f"## {prediction}")
        
        with right_col:
            waterfall_placeholder.image(imagen_bytes, use_container_width=True)
        
        # n session_state para persistencia (por si se recarga la página)
        st.session_state.prediction = prediction
        st.session_state.waterfall_bytes = imagen_bytes

# ------------------------------------------------------------
# 5. If a prediction was previously made (via session_state), display the results upon page load.
# ------------------------------------------------------------
else:
    if 'prediction' in st.session_state and 'waterfall_bytes' in st.session_state:
        with left_col:
            prediction_placeholder.markdown(f"## {st.session_state.prediction}")
        with right_col:
            waterfall_placeholder.image(st.session_state.waterfall_bytes, use_container_width=True)