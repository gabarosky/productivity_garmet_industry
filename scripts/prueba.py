# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:31:26 2026

@author: gabri
"""

# pruebas/test_artefactos.py
import joblib
import pandas as pd
import json
import shap
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_waterfall(sample, preprocessor, model, explainer,sample_name=''):
    ''' This function take a sample of data, calculate Shapley values and plot a waterfall for 
    category in order to show the impact of each eature in the final prediction.
    Variables:
    sample: a dataframe with the sample of variables to predict and analize.
    model: the model used for predictions
    prepreocessor: the preprocessor used for transform the data.
    explainer: the Shapley explainer trained with data and model.
    sample_name: a string for the title'''
    
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

def test_carga_y_prediccion():
    # Cargar pipeline
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '..', 'datasets', 'garments_worker_productivity.csv')    
    MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')
    pipeline = joblib.load(MODEL_DIR+'/modelo_pipeline.pkl')
    print("Pipeline cargado correctamente")

    # Cargar explainer
    explainer = joblib.load(MODEL_DIR+'/explainer.pkl')
    print("Explainer cargado correctamente")

    # Cargar metadatos (para conocer las columnas originales)
    with open(MODEL_DIR+'/feature_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Crear un dato de prueba: una fila con valores típicos
    # Debes usar las columnas originales (numéricas + categóricas)
    numeric_cols = list(metadata['numeric_ranges'].keys())
    cat_cols = list(metadata['categorical_categories'].keys())
    
    # Ejemplo: tomar valores medios de las numéricas y primera categoría de las categóricas
    datos_prueba = {}
    for col in numeric_cols:
        datos_prueba[col] = metadata['numeric_ranges'][col]['min']  # o un valor medio
    for col in cat_cols:
        datos_prueba[col] = metadata['categorical_categories'][col][0]
    
    df_prueba = pd.DataFrame([datos_prueba])
    print("Dato de prueba:\n", df_prueba)

    # Predecir
    prediccion = pipeline.predict(df_prueba)[0]
    print("Predicción:", prediccion)

    # Calcular SHAP
    X_transform = pipeline.named_steps['preprocess'].transform(df_prueba)
    shap_values = explainer.shap_values(X_transform)
    
    # Si es multiclase, shap_values es una lista; tomamos la clase 1 por ejemplo
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
        expected = explainer.expected_value[1]
    else:
        shap_vals = shap_values
        expected = explainer.expected_value
    plot_waterfall(df_prueba, pipeline.named_steps['preprocess'], pipeline.named_steps['model'], explainer,sample_name='')
    
    print("SHAP values calculados, forma:", shap_vals.shape)
    print("Expected value:", expected)
    print("¡Todo funcionó correctamente!")

if __name__ == "__main__":
    test_carga_y_prediccion()