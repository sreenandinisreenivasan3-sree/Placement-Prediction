import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
    st.error("❌ Model not loaded! Please train the model first by running: python save_model.py")
    st.stop()

st.success("✅ Model loaded successfully!")

# Create input form
st.header("📊 Student Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Academic Information")
    cgpa = st.slider("CGPA", 5.0, 10.0, 7.5, 0.01)
    internships = st.number_input("Number of Internships", 0, 5, 1)
    projects = st.number_input("Number of Projects", 0, 10, 2)
    workshops = st.number_input("Number of Workshops/Certifications", 0, 10, 2)

with col2:
    st.subheader("Skills & Aptitude")
    aptitude_score = st.slider("Aptitude Test Score", 0, 100, 75)
    soft_skills_score = st.slider("Soft Skills Score", 0, 100, 70)
    placement_rating = st.slider("Placement Cell Rating (1-5)", 1.0, 5.0, 3.5, 0.5)

# Specialization selection (if in feature columns)
specialization_options = ['CS', 'IT', 'ECE', 'MECH', 'CIVIL', 'EEE']
if 'specialization' in feature_columns or any('specialization' in str(col) for col in feature_columns):
    specialization = st.selectbox("Specialization", specialization_options)
else:
    specialization = None

# Predict button
if st.button("🔮 Predict Placement Chance", type="primary", use_container_width=True):
    # Prepare input data
    input_data = {
        'cgpa': cgpa,
        'internships': internships,
        'projects': projects,
        'workshops': workshops,
        'aptitude_test_score': aptitude_score,
        'soft_skills_score': soft_skills_score,
        'placement_cell_rating': placement_rating
    }
    
    # Add specialization if it exists in features
    if specialization is not None and 'specialization' in feature_columns:
        input_data['specialization'] = specialization
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure all required columns are present
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[feature_columns]
    
    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Display results
        st.markdown("---")
        st.header("🎯 Prediction Result")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown("""
                    <div class="prediction-box">
                        <h2>✅ PLACED</h2>
                        <p>Congratulations! High chance of placement</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="prediction-box" style="background-color: #EF4444;">
                        <h2>⚠️ NOT PLACED</h2>
                        <p>Need improvement in key areas</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Placement Probability", f"{probability*100:.1f}%")
        
        with col3:
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability*100,
                title={'text': "Success Rate"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#10B981"},
                    'steps': [
                        {'range': [0, 40], 'color': "#EF4444"},
                        {'range': [40, 70], 'color': "#F59E0B"},
                        {'range': [70, 100], 'color': "#10B981"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': probability*100
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.subheader("📝 Recommendations")
        
        recommendations = []
        if cgpa < 7.0:
            recommendations.append("📚 Improve your CGPA (target 7.5+)")
        if aptitude_score < 70:
            recommendations.append("📝 Take aptitude test preparation courses")
        if soft_skills_score < 70:
            recommendations.append("💬 Work on communication and soft skills")
        if internships == 0:
            recommendations.append("💼 Apply for internships to gain experience")
        if projects < 2:
            recommendations.append("🛠️ Build more projects to showcase your skills")
        
        if recommendations:
            for rec in recommendations:
                st.write(rec)
        else:
            st.write("🌟 You're on the right track! Keep maintaining your performance.")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Debug info:")
        st.write(f"Input DataFrame shape: {input_df.shape}")
        st.write(f"Expected columns: {feature_columns}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Powered by Machine Learning | Placement Prediction System</p>",
    unsafe_allow_html=True
)
