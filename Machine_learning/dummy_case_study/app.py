import streamlit as st
import pandas as pd
import numpy as np
import pickle

# load the trained model
with open('knn_reg_model.pkl', 'rb') as f:
    knn_reg = pickle.load(f)

# load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Car Price Prediction App")
st.write("Enter the car features to predict the price.")
# input features
kilometers_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000)
engine_cc = st.number_input("Engine CC", min_value=500, max_value=5000, value=1500)
max_power_bhp = st.number_input("Max Power (bhp)", min_value=20, max_value=500, value=100)
mileage_kmpl = st.number_input("Mileage (kmpl)", min_value=5, max_value=50, value=18)   
seats = st.number_input("Seats", min_value=2, max_value=10, value=5)
brand = st.selectbox("Brand", ['BMW', 'Ford', 'Honda', 'Toyota'])
fuel_type = st.selectbox("Fuel Type", ['Diesel', 'Petrol'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
owner_type = st.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])

# create a dataframe for the input features 

sample_data = {
    'Kilometers_Driven': kilometers_driven,
    'Engine_CC': engine_cc,
    'Max_Power_bhp': max_power_bhp,
    'Mileage_kmpl': mileage_kmpl,
    'Seats': seats,
    'Brand_BMW': 0,
    'Brand_Ford': 0,
    'Brand_Honda': 1,
    'Brand_Toyota': 0,
    'Fuel_Type_Diesel': 0,
    'Fuel_Type_Petrol': 1,
    'Transmission_Manual': 1,
    'Owner_Type_Second Owner': 0,
    'Owner_Type_Third Owner': 0,
    'Owner_Type_Fourth & Above Owner': 0
}

input_df = pd.DataFrame([sample_data])
# scale the input features
numeric_features = ['Kilometers_Driven', 'Engine_CC', 'Max_Power_bhp', 'Mileage_kmpl', 'Seats']
input_df = scaler.transform(input_df[numeric_features])
# take categorical features as is
input_df 
# predict the price
if st.button("Predict Price"):
    predicted_price = knn_reg.predict(input_df)
    st.write(f"The predicted price of the car is: ₹ {predicted_price[0]:.2f}")
