import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
    <style>
    body {
        background-color: #ffffff;
    }
    .main {
        color: #000000;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #0B8F6C;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5em 1.5em;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #09775c;
    }
    h1, h2, h3, .stSubheader {
        color: #0B8F6C;
    }
    </style>
""", unsafe_allow_html=True)

model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient information below to get a prediction:")

col1, col2 = st.columns(2)
Pregnancies = col1.number_input("Pregnancies", min_value=0, max_value=20)
Glucose = col2.number_input("Glucose", min_value=0, max_value=200)
BloodPressure = col1.number_input("Blood Pressure", min_value=0, max_value=150)
SkinThickness = col2.number_input("Skin Thickness", min_value=0, max_value=100)
Insulin = col1.number_input("Insulin", min_value=0, max_value=900)
BMI = col2.number_input("BMI", min_value=0.0, max_value=70.0)
DPF = col1.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0)
Age = col2.number_input("Age", min_value=1, max_value=120)

if st.button("🔍 Predict"):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DPF, Age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader("📄 Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠️ The model predicts: **Diabetic** ({prob*100:.2f}% confidence)")
    else:
        st.success(f"✅ The model predicts: **Not Diabetic** ({(1 - prob)*100:.2f}% confidence)")

    st.caption(f"⏱️ Prediction generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    st.subheader("📊 Prediction Probability (Bar)")

    fig, ax = plt.subplots(figsize=(6, 2.5))
    labels = ['Not Diabetic', 'Diabetic']
    values = [(1 - prob) * 100, prob * 100]
    colors = ["#F1D272", "#9AD5FC"]

    bars = ax.barh(labels, values, color=colors)
    ax.set_xlim([0, 100])
    ax.set_xlabel("Confidence (%)", fontsize=12)
    ax.set_title("Prediction Confidence", loc='left', fontsize=14)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va='center')

    st.pyplot(fig)
