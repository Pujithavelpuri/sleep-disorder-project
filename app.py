


import streamlit as st

import streamlit as st

st.set_page_config(
    page_title="Predicting Sleep Disorders",
    layout="centered"
)

st.title("😴 Predicting Sleep Disorders")
st.markdown(
    """
    This application predicts the type of sleep disorder  
    based on health and lifestyle indicators.py -m streamlit run app.py
    """
)

st.header("Enter Patient Details")

age = st.number_input("Age", min_value=1, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 10.0, 7.0)
stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
bmi = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])

if st.button("Predict Sleep Disorder"):
    st.success("Prediction: Insomnia (Demo Output)")
    