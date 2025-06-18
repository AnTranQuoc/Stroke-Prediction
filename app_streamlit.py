import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load mô hình và scaler
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")  # list cột đã fit

st.title("Stroke Prediction App")

# Nhập thông tin người dùng
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 1, 100, 30)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
work_related_stress = st.selectbox("Work Related Stress", ["No", "Yes"])
urban_residence = st.selectbox("Urban Residence", ["No", "Yes"])
avg_glucose_level = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
bmi = st.number_input("BMI", 10.0, 60.0, 22.0)
smokes = st.selectbox("Smokes", ["Never", "Formerly", "Smokes"])

if st.button("Predict Stroke Risk"):
    # Chuyển đổi các input thành features
    data = {
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "work_related_stress": 1 if work_related_stress == "Yes" else 0,
        "urban_residence": 1 if urban_residence == "Yes" else 0,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        # One-hot cho gender
        "gender_Female": 1 if gender == "Female" else 0,
        "gender_Male": 1 if gender == "Male" else 0,
        "gender_Other": 1 if gender == "Other" else 0,
        # One-hot cho smokes
        "smokes_formerly": 1 if smokes == "Formerly" else 0,
        "smokes_never": 1 if smokes == "Never" else 0,
        "smokes_smokes": 1 if smokes == "Smokes" else 0,
    }

    # Chuyển về DataFrame
    df = pd.DataFrame([data])

    # Đảm bảo đúng format
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scale dữ liệu
    df_scaled = scaler.transform(df)

    # Dự đoán
    prediction = model.predict(df_scaled)[0]
    proba = model.predict_proba(df_scaled)[0][1] if hasattr(model, "predict_proba") else None

    st.success(f"🩺 Stroke Prediction: {'⚠️ Likely' if prediction==1 else '✅ Unlikely'}")
    if proba:
        st.info(f"Probability of stroke: {proba:.2%}")
