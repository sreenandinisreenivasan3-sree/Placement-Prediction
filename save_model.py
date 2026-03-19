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

print("="*50)
print("Starting model training...")
print("="*50)

# IMPORTANT: Update this path to where your CSV file is located
# It's probably in your Downloads folder
csv_path = r"C:\Users\lenovo\Downloads\campus_placement_data.csv"

try:
    df = pd.read_csv(csv_path)
    print(f"✅ CSV file loaded successfully!")
    print(f"   Shape: {df.shape}")
except FileNotFoundError:
    print(f"❌ Error: Could not find CSV file at: {csv_path}")
    print("   Please check if the file exists in your Downloads folder")
    exit()

# Handle missing values
df['specialization'].fillna('None', inplace=True)
df = df.drop(columns=["student_id", "salary_lpa"], errors="ignore")
print(f"✅ Data preprocessed successfully!")

# Separate features and target
X = df.drop('placed', axis=1)
y = df['placed']

print(f"   Features: {X.shape[1]} columns")
print(f"   Target distribution: {y.value_counts().to_dict()}")

# Identify column types
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"   Numeric columns: {len(num_cols)}")
print(f"   Categorical columns: {len(cat_cols)}")

# Create preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

# Best parameters from your model
best_params = {
    'subsample': 0.8,
    'n_estimators': 300,
    'min_child_weight': 5,
    'max_depth': 3,
    'learning_rate': 0.1,
    'gamma': 0,
    'colsample_bytree': 1.0
}

# Create the pipeline
final_model = ImbPipeline([
    ("prep", preprocess),
    ("select", SelectKBest(score_func=f_classif, k=20)),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        **best_params
    ))
])

print("\n🔄 Training model... (this may take a few minutes)")

# Train on ALL data
final_model.fit(X, y)

print("✅ Model training complete!")

# Save the model and column information
joblib.dump(final_model, 'placement_model.pkl')
joblib.dump(X.columns.tolist(), 'columns.pkl')
joblib.dump(num_cols, 'num_cols.pkl')
joblib.dump(cat_cols, 'cat_cols.pkl')

print("\n✅ All files saved successfully!")
print(f"📁 Location: {os.getcwd()}")
print("\nFiles created:")
print("   - placement_model.pkl")
print("   - columns.pkl")
print("   - num_cols.pkl")
print("   - cat_cols.pkl")
print("\nYou can now run: python -m streamlit run app.py")