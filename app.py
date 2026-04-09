import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Placement Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
    }
    .prediction-box {
        background-color: #10B981;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .info-box {
        background-color: #3B82F6;
        padding: 1rem;
        border-radius: 5px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎓 Campus Placement Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Check if model files exist
MODEL_FILES = ['placement_model.pkl', 'columns.pkl', 'num_cols.pkl', 'cat_cols.pkl']
all_files_exist = all(os.path.exists(f) for f in MODEL_FILES)

# Debug info in sidebar
with st.sidebar:
    st.header("ℹ️ System Status")
    if all_files_exist:
        st.success("✅ Model files found!")
        for f in MODEL_FILES:
            st.write(f"✓ {f}")
    else:
        st.warning("⚠️ Some model files are missing:")
        for f in MODEL_FILES:
            if not os.path.exists(f):
                st.write(f"✗ {f}")
    
    st.write("---")
    st.write(f"📁 Working directory: `{os.getcwd()}`")
    
    # Show all files in directory
    with st.expander("📂 Available files"):
        files = os.listdir('.')
        for file in files:
            st.write(f"- {file}")

# Load model with caching
@st.cache_resource
def load_placement_model():
    """Load the trained model and required components"""
    try:
        # Check if all files exist
        if not all(os.path.exists(f) for f in MODEL_FILES):
            return None, None, None, None
        
        model = joblib.load('placement_model.pkl')
        columns = joblib.load('columns.pkl')
        num_cols = joblib.load('num_cols.pkl')
        cat_cols = joblib.load('cat_cols.pkl')
        
        return model, columns, num_cols, cat_cols
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

# Try to load model
model, feature_columns, num_cols, cat_cols = load_placement_model()

if model is None:
    st.error("""
    ❌ **Model not loaded!**
    
    Please train the model first by running:
    ```bash
    python save_model.py
