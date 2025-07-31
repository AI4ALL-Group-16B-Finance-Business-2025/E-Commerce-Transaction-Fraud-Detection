import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("WRS_model.sav")

model = load_model()

# App title and info
st.title('Credit Card Fraud Detection')
st.markdown('Enter transaction data to predict if it is fraudulent.')

# Input fields for V1 to V28 and Amount_scaled
feature_values = {}
st.header("Enter Transaction Features")

for i in range(1, 29):  # Display Features V1 to V28
    feature_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.01)

feature_values["Amount_scaled"] = st.number_input("Amount_scaled", value=0.0, step=0.01)

# Predict button
if st.button("Predict Fraud"):
    input_data = pd.DataFrame([feature_values])
    result = model.predict(input_data)[0] # grab input
    st.write(f"Prediction: {'Fraud' if result == 1 else 'Not Fraud'}")
