import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('heart_disease_model.pkl')

st.title("Heart Disease Predictor")

# Add input fields for your features (age, chol, etc.)
age = st.number_input("Age", min_value=1, max_value=120, value=25)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)

if st.button("Predict"):
    # Make sure the features are in the same order as your X_train
    prediction = model.predict([[age, chol, ...]]) # Add all other features here
    st.write(f"Prediction: {'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease'}")