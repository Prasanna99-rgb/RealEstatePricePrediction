import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("XGBR.pkl","rb"))

st.title("California House Price Prediction")

MedInc = st.number_input("Median Income",0.0,15.0)
HouseAge = st.number_input("House Age",1,52)
AveRooms = st.number_input("Average Rooms",0.0,15.0)
AveBedrms = st.number_input("Average Bedrooms",0.0,5.0)
Population = st.number_input("Population",0,40000)
AveOccup = st.number_input("Average Occupancy",0.0,1250.0)
Latitude = st.number_input("Latitude",32.5,42.0)
Longitude = st.number_input("Longitude",-124.5,-114.0)

if st.button("Predict Price"):
    
    features = np.array([[MedInc,HouseAge,AveRooms,AveBedrms,
                          Population,AveOccup,Latitude,Longitude]])
    
    prediction = model.predict(features)

    st.success(f"Predicted House Price: {prediction[0]}")
