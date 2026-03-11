# scripts/train_model.py
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import shap
import sys
import os

# Add scripts to path
sys.path.append(os.path.dirname(__file__))
from data_preprocessing import load_and_clean_data, create_target, get_feature_lists, prepare_X_y


RANDOM_STATE = 43

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'datasets', 'garments_worker_productivity.csv')    
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')    
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    # 1. Load and prepare data
    print("Loading dataset...")
    df = load_and_clean_data(DATA_PATH)
    df = create_target(df)
    numeric_cols, categorical_cols = get_feature_lists()
    X, y, label_encoder = prepare_X_y(df, numeric_cols, categorical_cols)

    # 2. Set preprocessor and pipeline
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'))
    ])

    # 3. Hiperparamter tunning
    print("Hiperparameter tuning...")
    param_distributions = {
        "model__n_estimators": [100, 200, 500, 800],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ['sqrt', 'log2', None],
        "model__bootstrap": [True, False],
        "model__class_weight": ['balanced', 'balanced_subsample', None]
    }
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=25,
        cv=skf,
        scoring='neg_mean_absolute_error',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X, y)
    best_pipeline = search.best_estimator_
    print("Best paramteres:", search.best_params_)

    # 4. Save the pipeline
    print("Saving pipeline...")
    joblib.dump(best_pipeline, os.path.join(MODEL_DIR, 'modelo_pipeline.pkl'))

    # 5. Create and save SHAP explainer 
    print("Creating SHAP explainer...")
    best_model = search.best_estimator_
    final_model = best_model.named_steps['model']
    preprocessor = best_model.named_steps["preprocess"]
    # transform data
    X_transformed = preprocessor.transform(X)
    # get transformed feature names
    feature_names = preprocessor.get_feature_names_out()
    # make a dataframe of transformed
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    # create explainer with train data
    background = shap.sample(X_transformed_df, 100)  # 100 random samples for speed up
    explainer = shap.Explainer(final_model, background)
    joblib.dump(explainer, os.path.join(MODEL_DIR, 'explainer.pkl'))

    # 6. Save label encoder
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    # 7. Get and save feature metadata
    print("Generating metadata...")
    # Final feature names (after one-hot encoding)
    feature_names = best_pipeline.named_steps['preprocess'].get_feature_names_out()
    feature_names_clean = [name.split('__')[-1] for name in feature_names]

    # Ranges for numeric features (using original X, before transformation)
    numeric_ranges = {}
    for col in numeric_cols:
        numeric_ranges[col] = {
            'min': float(X[col].min()),
            'max': float(X[col].max())
        }

    # Categories for categorical variables
    categorical_categories = {}
    for col in categorical_cols:
        categorical_categories[col] = X[col].unique().tolist()

    metadata = {
        'numeric_ranges': numeric_ranges,
        'categorical_categories': categorical_categories,
        'feature_names_final': feature_names_clean
    }

    with open(os.path.join(MODEL_DIR, 'feature_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Process completed! Artifacts saved in", MODEL_DIR)

if __name__ == "__main__":
    main()