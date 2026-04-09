import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
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
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎓 Campus Placement Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Load model with caching
@st.cache_resource
def load_placement_model():
    """Load the trained model and required components"""
    try:
        model = joblib.load('placement_model.pkl')
        columns = joblib.load('columns.pkl')
        return model, columns
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Try to load model
model, feature_columns = load_placement_model()

if model is None:
    st.error("❌ Model not loaded! Please train the model first.")
    st.stop()

st.success(f"✅ Model loaded successfully! Expecting {len(feature_columns)} features.")

# Create input form
st.header("📊 Student Information")

# Create two columns for input organization
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    gender = st.selectbox("Gender", ["Male", "Female","Other"])
    age = st.number_input("Age", 18, 30, 21)
    city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    
    st.subheader("Academic Background")
    ssc_percentage = st.slider("SSC Percentage", 50.0, 100.0, 75.0, 0.5)
    ssc_board = st.selectbox("SSC Board", ["State Board", "CBSE", "ICSE"])
    hsc_percentage = st.slider("HSC Percentage", 50.0, 100.0, 75.0, 0.5)
    hsc_board = st.selectbox("HSC Board", ["State Board", "CBSE", "ICSE"])
    hsc_stream = st.selectbox("HSC Stream", ["Science", "Commerce", "Arts"])
    
    degree_percentage = st.slider("Degree Percentage", 50.0, 100.0, 75.0, 0.5)
    degree_field = st.selectbox("Degree Field", ["Engineering", "Science", "Commerce", "Arts"])
    mba_percentage = st.slider("MBA Percentage", 50.0, 100.0, 75.0, 0.5)
    specialization = st.selectbox("Specialization", ["Marketing", "Finance", "HR", "IT", "Operations", "None"])

with col2:
    st.subheader("Experience & Skills")
    internships_count = st.number_input("Number of Internships", 0, 5, 1)
    projects_count = st.number_input("Number of Projects", 0, 10, 2)
    certifications_count = st.number_input("Number of Certifications", 0, 10, 1)
    
    technical_skills_score = st.slider("Technical Skills Score", 0, 100, 70)
    soft_skills_score = st.slider("Soft Skills Score", 0, 100, 70)
    aptitude_score = st.slider("Aptitude Score", 0, 100, 70)
    communication_score = st.slider("Communication Score", 0, 100, 70)
    
    work_experience_months = st.number_input("Work Experience (months)", 0, 60, 0)
    leadership_roles = st.number_input("Leadership Roles", 0, 10, 0)
    extracurricular_activities = st.number_input("Extracurricular Activities", 0, 10, 2)
    backlogs = st.number_input("Number of Backlogs", 0, 20, 0)

# Predict button
if st.button("🔮 Predict Placement Chance", type="primary", use_container_width=True):
    # Prepare input data with all required features
    input_data = {
        'gender': 1 if gender == "Male" else 0,  # Encode as numeric
        'age': age,
        'city_tier': int(city_tier.split()[1]),  # Extract tier number
        'ssc_percentage': ssc_percentage,
        'ssc_board': ssc_board,
        'hsc_percentage': hsc_percentage,
        'hsc_board': hsc_board,
        'hsc_stream': hsc_stream,
        'degree_percentage': degree_percentage,
        'degree_field': degree_field,
        'mba_percentage': mba_percentage,
        'specialization': specialization,
        'internships_count': internships_count,
        'projects_count': projects_count,
        'certifications_count': certifications_count,
        'technical_skills_score': technical_skills_score,
        'soft_skills_score': soft_skills_score,
        'aptitude_score': aptitude_score,
        'communication_score': communication_score,
        'work_experience_months': work_experience_months,
        'leadership_roles': leadership_roles,
        'extracurricular_activities': extracurricular_activities,
        'backlogs': backlogs
    }
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure all required columns are present and in correct order
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
                        <h2>✅ LIKELY TO BE PLACED</h2>
                        <p>Congratulations! High chance of placement</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="prediction-box" style="background-color: #EF4444;">
                        <h2>⚠️ NEEDS IMPROVEMENT</h2>
                        <p>Low chance of placement based on current profile</p>
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
        st.subheader("📝 Recommendations for Improvement")
        
        recommendations = []
        if ssc_percentage < 70:
            recommendations.append("📚 Improve your academic scores")
        if degree_percentage < 70:
            recommendations.append("🎓 Focus on improving your degree percentage")
        if technical_skills_score < 70:
            recommendations.append("💻 Enhance your technical skills through online courses")
        if soft_skills_score < 70:
            recommendations.append("🗣️ Work on communication and soft skills")
        if internships_count == 0:
            recommendations.append("💼 Apply for internships to gain practical experience")
        if projects_count < 2:
            recommendations.append("🛠️ Build more projects to showcase your skills")
        if aptitude_score < 70:
            recommendations.append("📝 Practice aptitude tests regularly")
        if work_experience_months == 0 and internships_count == 0:
            recommendations.append("🏢 Consider internships or entry-level positions for experience")
        if backlogs > 0:
            recommendations.append("📖 Clear your backlogs to improve academic standing")
        
        if recommendations:
            for rec in recommendations:
                st.write(rec)
        else:
            st.write("🌟 Excellent profile! You're well-prepared for placements. Keep maintaining your performance.")
        
        # Show input summary
        with st.expander("View Input Summary"):
            st.write("### Profile Summary")
            summary_df = pd.DataFrame([input_data])
            st.dataframe(summary_df)
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Debug info:")
        st.write(f"Input DataFrame shape: {input_df.shape}")
        st.write(f"Expected columns: {feature_columns[:5]}... (total {len(feature_columns)})")
        st.write(f"Input columns: {input_df.columns.tolist()}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Powered by Machine Learning | Placement Prediction System</p>",
    unsafe_allow_html=True
)
