import streamlit as st
import pandas as pd
import pickle

# Load pipeline
@st.cache_resource
def load_pipeline(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pipe = load_pipeline("C:/Users/nadaa/OneDrive/Desktop/Heart Disease Project/models/final_model.pkl")


# UI
st.title("❤️ Heart Disease Prediction App")

st.write("Enter the patient details below:")

# Collect inputs
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (cp)", [1, 2, 3, 4])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=70, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=6.0, step=0.1, value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST (slope)", [1, 2, 3])
ca = st.selectbox("Number of Major Vessels Colored (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", [3, 6, 7])

# Predict
if st.button("🔍 Predict"):
    input_df = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg,
        "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal
    }])

    prediction = pipe.predict(input_df)[0]

    if prediction == 0:
        st.success("✅ No Heart Disease Detected")
    else:
        st.error("⚠️ Heart Disease Detected")
