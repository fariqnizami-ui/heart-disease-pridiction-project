import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Heart Disease AI", page_icon="🫀", layout="wide")

# 2. Load the Model
@st.cache_resource
def load_model():
    return joblib.load('heart_disease_model.pkl')

model = load_model()

# 3. Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# 4. Header Section
st.title("🫀 Cardiovascular Risk Assessment Tool")
st.markdown("---")

# 5. Organizing Inputs into Columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Basic Profile")
    age = st.slider("Patient Age", 1, 100, 45)
    sex = st.radio("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help="0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic")

with col2:
    st.subheader("🩺 Clinical Vitals")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])

with col3:
    st.subheader("⚡ Stress Test Data")
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.radio("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("ST Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored by Flourosopy", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia Type", [0, 1, 2, 3])

# Create the Feature DataFrame
input_dict = {
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
    'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
}
df_input = pd.DataFrame([input_dict])

st.markdown("---")

# 6. Prediction Logic
if st.button("Generate Diagnostic Report"):
    prediction = model.predict(df_input)
    prediction_proba = model.predict_proba(df_input)
    confidence = np.max(prediction_proba) * 100

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric(label="Model Confidence", value=f"{confidence:.1f}%")
    
    with res_col2:
        if prediction[0] == 1:
            st.error("### ⚠️ Result: High Risk Detected")
            st.write("The model suggests a high probability of heart disease. Clinical consultation is recommended.")
        else:
            st.success("### ✅ Result: Low Risk Detected")
            st.write("The model suggests no immediate signs of cardiovascular disease based on these parameters.")

st.info("**Disclaimer:** This tool is for educational purposes and based on the UCI Heart Disease dataset. It should not be used as a substitute for professional medical advice.")
