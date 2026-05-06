import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

# 2. Load the Model (Generated in your notebook cell [61])
@st.cache_resource
def load_model():
    return joblib.load('heart_disease_model.pkl')

model = load_model()

# 3. Sidebar Inputs (Matching your 13 features)
st.sidebar.header("Patient Clinical Data")

def user_input_features():
    age = st.sidebar.slider("Age", 1, 100, 50)
    sex = st.sidebar.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.number_input("Serum Cholestoral", 100, 600, 240)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST", [0, 1, 2])
    ca = st.sidebar.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

# 4. Main Panel
st.title("❤️ Heart Disease Prediction Tool")
st.write("Enter patient data in the sidebar to predict heart disease risk.")

st.subheader("Patient Summary")
st.write(df_input)

if st.button("Run Diagnostic"):
    # Prediction
    prediction = model.predict(df_input)
    prediction_proba = model.predict_proba(df_input)

    st.subheader("Result")
    if prediction[0] == 1:
        st.error("⚠️ High Risk: Heart Disease Likely")
    else:
        st.success("✅ Low Risk: Heart Disease Unlikely")
    
    st.write(f"**Confidence:** {np.max(prediction_proba)*100:.2f}%")
