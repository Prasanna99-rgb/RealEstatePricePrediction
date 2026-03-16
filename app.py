import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("XGBR.pkl", "rb"))

st.set_page_config(page_title="XGBoost Prediction App", layout="centered")

st.title("🔮 XGBoost Regression Prediction App")
st.write("Enter the feature values to get prediction")

# Input fields (8 features)
f1 = st.number_input("MedInc")
f2 = st.number_input("HouseAge")
f3 = st.number_input("AveRooms")
f4 = st.number_input("AveBedrms")
f5 = st.number_input("Population")
f6 = st.number_input("AveOccup")
f7 = st.number_input("Latitude")
f8 = st.number_input("Longitude")

# Prediction button
if st.button("Predict"):

    features = np.array([[f1, f2, f3, f4, f5, f6, f7, f8]])

    prediction = model.predict(features)

    st.success(f"Prediction Result: {prediction[0]}")
