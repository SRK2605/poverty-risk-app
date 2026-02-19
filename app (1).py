import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("poverty_model.pkl")

st.set_page_config(page_title="Temporary Poverty Predictor", layout="centered")

st.title("Temporary Poverty Early Warning System")
st.write("AI-Based Income & Financial Risk Prediction")

st.sidebar.header("Enter User Details")

age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.slider("Education Level (Encoded)", 0, 15, 5)
occupation = st.sidebar.slider("Occupation (Encoded)", 0, 20, 5)
marital = st.sidebar.slider("Marital Status (Encoded)", 0, 5, 1)
hours = st.sidebar.slider("Hours per Week", 1, 80, 40)

time_strain = 1 if hours > 50 else 0

input_df = pd.DataFrame({
    'age': [age],
    'education_num': [education],
    'occupation_num': [occupation],
    'marital_num': [marital],
    'time_strain': [time_strain]
})

if st.button("Predict"):

    prediction = model.predict(input_df)[0]
    risk_score = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    # Income Category
    if prediction == 1:
        st.error("Predicted Income: <=50K (Higher Financial Risk)")
    else:
        st.success("Predicted Income: >50K (Lower Financial Risk)")

    # Risk Probability
    st.metric("Risk Probability", f"{risk_score:.2f}")

    # Risk Level Interpretation
    if risk_score > 0.75:
        st.warning("High Vulnerability")
        st.write("Recommendation: Immediate financial planning support advised.")
    elif risk_score > 0.5:
        st.info("Moderate Risk")
        st.write("Recommendation: Monitor expenses and stabilize income.")
    else:
        st.success("Low Risk")
        st.write("Financial condition appears stable.")
