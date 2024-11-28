import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Load the trained model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'churn_updated_compressed.joblib')
    model = joblib.load(model_path)
    return model

model = load_model()

# Define the Streamlit app
st.title("Customer Churn Prediction App")

# User input
st.header("Input Customer Data")
age = st.number_input("Age", min_value=18, max_value=100, step=1)
gender = st.selectbox("Gender", options=["Male", "Female"])
avg_time_spent = st.number_input("Average Time Spent (minutes)", min_value=0.0, step=0.1)
avg_transaction_value = st.number_input("Average Transaction Value", min_value=0.0, step=0.1)
avg_frequency_login_days = st.number_input("Average Frequency of Login (days)", min_value=0.0, step=0.1)
points_in_wallet = st.number_input("Points in Wallet", min_value=0.0, step=0.1)

# Convert inputs to a format suitable for the model
gender_encoded = 1 if gender == "Male" else 0
input_data = np.array([age, gender_encoded, avg_time_spent, avg_transaction_value, avg_frequency_login_days, points_in_wallet]).reshape(1, -1)

# Predict
if st.button("Predict Churn Risk Score"):
    prediction = model.predict(input_data)[0]
    st.write(f"Predicted Churn Risk Score: {prediction}")
