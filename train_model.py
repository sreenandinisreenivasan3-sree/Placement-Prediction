# train_model.py - Only run this manually when needed
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def train_model():
    """Function to train and save the model"""
    print("="*50)
    print("Starting model training...")
    print("="*50)
    
    # Look for CSV file
    csv_path = "campus_placement_data.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Available files: {os.listdir('.')}")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {csv_path} successfully!")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False
    
    # Preprocessing
    if 'specialization' in df.columns:
        df['specialization'].fillna('None', inplace=True)
    
    columns_to_drop = ["student_id", "salary_lpa"]
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Prepare features and target
    X = df.drop('placed', axis=1)
    y = df['placed']
    
    # Identify column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Create preprocessing pipeline
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ])
    
    best_params = {
        'subsample': 0.8,
        'n_estimators': 300,
        'min_child_weight': 5,
        'max_depth': 3,
        'learning_rate': 0.1,
        'gamma': 0,
        'colsample_bytree': 1.0
    }
    
    k_features = min(20, X.shape[1])
    final_model = ImbPipeline([
        ("prep", preprocess),
        ("select", SelectKBest(score_func=f_classif, k=k_features)),
        ("smote", SMOTE(random_state=42)),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            **best_params
        ))
    ])
    
    print("\n🔄 Training model...")
    final_model.fit(X, y)
    print("✅ Training complete!")
    
    # Save model
    joblib.dump(final_model, 'placement_model.pkl')
    joblib.dump(X.columns.tolist(), 'columns.pkl')
    joblib.dump(num_cols, 'num_cols.pkl')
    joblib.dump(cat_cols, 'cat_cols.pkl')
    
    print("\n✅ Model saved successfully!")
    return True

if __name__ == "__main__":
    # This only runs when executed directly, not when imported
    train_model()
