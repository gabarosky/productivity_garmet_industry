# scripts/data_preprocessing.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(filepath):
    """Load CSV, basic clean and imputation."""
    df = pd.read_csv(filepath)
    df.drop_duplicates(inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df['department'] = df['department'].str.strip()
    df['department'] = df['department'].replace('sweing', 'sewing')
    df['wip'] = df['wip'].fillna(0)
    return df

def create_target(df):
    """create the target (prod_cat) based on actual_productivity."""
    df['prod_cat'] = pd.cut(
        df['actual_productivity'],
        bins=[0, 0.7, 0.8, 0.9, float('inf')],
        labels=[0, 1, 2, 3],
        right=False
    )
    return df

def get_feature_lists():
    """Define categorical and numerical features"""
    numeric_cols = [
        'targeted_productivity', 'smv', 'wip', 'over_time',
        'incentive', 'idle_time', 'idle_men', 'no_of_workers'
    ]
    categorical_cols = ['department']
    return numeric_cols, categorical_cols

def prepare_X_y(df, numeric_cols, categorical_cols, target_col='prod_cat'):
    """Split features and target, y apply LabelEncoder to y."""
    X = df[numeric_cols + categorical_cols]
    y = df[target_col]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, le


if __name__ == "__main__":  
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, '..', 'datasets', 'garments_worker_productivity.csv')    
    df = load_and_clean_data(data_path)
    df = create_target(df)
    num_cols, cat_cols = get_feature_lists()
    X, y, le = prepare_X_y(df, num_cols, cat_cols)
    print("X shape:", X.shape)
    print("Classes:", le.classes_)