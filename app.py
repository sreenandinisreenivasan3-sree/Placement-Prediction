import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Campus Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# Check if model files exist
@st.cache_resource
def load_model():
    try:
        model = joblib.load('placement_model.pkl')
        columns = joblib.load('columns.pkl')
        num_cols = joblib.load('num_cols.pkl')
        cat_cols = joblib.load('cat_cols.pkl')
        return model, columns, num_cols, cat_cols
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run save_model.py first.")
        st.stop()

# Load model
try:
    model, all_columns, num_cols, cat_cols = load_model()
except:
    st.warning("⚠️ Model not loaded. Please train the model first by running: python save_model.py")
    st.stop()

# Title
st.title("🎓 Campus Placement Prediction System")
st.markdown("### Predict your placement chances based on academic and personal details")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=30, value=22)
    city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    
    st.subheader("Academic Performance (School)")
    ssc_percentage = st.number_input("SSC Percentage", min_value=0.0, max_value=100.0, value=70.0)
    ssc_board = st.selectbox("SSC Board", ["Central", "State", "CBSE", "ICSE"])
    
    hsc_percentage = st.number_input("HSC Percentage", min_value=0.0, max_value=100.0, value=70.0)
    hsc_board = st.selectbox("HSC Board", ["Central", "State", "CBSE", "ICSE"])
    hsc_stream = st.selectbox("HSC Stream", ["Science", "Commerce", "Arts"])

with col2:
    st.subheader("Higher Education")
    degree_percentage = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, value=70.0)
    degree_field = st.selectbox("Degree Field", ["Sci&Tech", "Commerce", "Arts", "Management"])
    
    mba_percentage = st.number_input("MBA Percentage", min_value=0.0, max_value=100.0, value=60.0)
    specialization = st.selectbox("MBA Specialization", 
                                  ["Marketing & HR", "Marketing & Finance", "Finance & HR", "None"])
    
    st.subheader("Experience & Skills")
    work_experience_months = st.number_input("Work Experience (months)", min_value=0, max_value=60, value=0)
    internships_count = st.number_input("Number of Internships", min_value=0, max_value=10, value=1)
    projects_count = st.number_input("Number of Projects", min_value=0, max_value=20, value=3)
    certifications_count = st.number_input("Number of Certifications", min_value=0, max_value=20, value=1)

# Third row for skills
st.subheader("Skills & Achievements")
col3, col4, col5 = st.columns(3)

with col3:
    technical_skills_score = st.slider("Technical Skills Score", 0, 10, 6)
    soft_skills_score = st.slider("Soft Skills Score", 0, 10, 6)
    communication_score = st.slider("Communication Score", 0, 10, 6)

with col4:
    aptitude_score = st.slider("Aptitude Score", 0, 100, 60)
    leadership_roles = st.number_input("Leadership Roles (count)", min_value=0, max_value=20, value=1)
    extracurricular_activities = st.number_input("Extracurricular Activities", min_value=0, max_value=20, value=2)

with col5:
    backlogs = st.number_input("Number of Backlogs", min_value=0, max_value=20, value=0)

# Prediction button
if st.button("🎯 Predict Placement", type="primary", use_container_width=True):
    
    # Create input dataframe
    input_data = pd.DataFrame([[  
        gender, age, city_tier, ssc_percentage, ssc_board,
        hsc_percentage, hsc_board, hsc_stream, degree_percentage,
        degree_field, mba_percentage, specialization,
        internships_count, projects_count, certifications_count,
        technical_skills_score, soft_skills_score, aptitude_score,
        communication_score, work_experience_months, leadership_roles,
        extracurricular_activities, backlogs
    ]], columns=all_columns)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    # Display result
    st.markdown("---")
    st.subheader("📊 Prediction Result")
    
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        if prediction == 1:
            st.success(f"### ✅ You are LIKELY to be placed!")
        else:
            st.error(f"### ❌ You are UNLIKELY to be placed")
    
    with result_col2:
        st.metric("Placement Probability", f"{probability:.1%}")
        st.progress(int(probability * 100))
    
    # Additional insights
    st.markdown("---")
    st.subheader("🔍 Insights")
    
    if probability > 0.7:
        st.info("🌟 Excellent chances! Focus on maintaining your skills.")
    elif probability > 0.4:
        st.warning("📈 Moderate chances. Consider improving your skills or gaining more experience.")
    else:
        st.warning("💪 Your chances are lower. Try to work on your academics, skills, and experience.")

# Sidebar with info
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/student-center.png", width=80)
    st.header("About")
    st.info("""
    This app predicts campus placement based on:
    - Academic performance
    - Skills & certifications
    - Work experience
    - Personal factors
    
    Model: XGBoost (Optimized)
    """)
    
    st.header("💡 Tips")
    st.markdown("""
    - Higher scores = better chances
    - Internships & projects help
    - Certifications boost probability
    - Keep backlogs to minimum
    """)