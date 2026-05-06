# 1. Collect all inputs (matching your training columns)
age = st.number_input("Age", value=25)
sex = st.selectbox("Sex (1=M, 0=F)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [0, 1])
restecg = st.selectbox("Resting ECG results (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise", value=0.0)
slope = st.selectbox("Slope of peak exercise ST segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (0-3)", [0, 1, 2, 3])

# 2. Arrange them in a list
input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

if st.button("Predict"):
    # 3. Pass the full list to the model
    prediction = model.predict(input_data)
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
    st.success(f"Prediction: {result}")
