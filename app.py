import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title("Used Car Price Prediction")

# Input fields for user
brand_model = st.text_input("Brand & Model")
varient = st.text_input("Variant")
fuel_type = st.selectbox("Fuel Type", ["PETROL", "DIESEL"])
driven_kilometers = st.number_input("Driven Kilometers", min_value=0.0, step=100.0)
transmission = st.selectbox("Transmission", ["MANUAL", "AUTOMATIC"])
owner = st.selectbox("Owner Type", ["1st Owner", "2nd Owner", "3rd Owner"])
location = st.text_input("Location")

# Predict button
if st.button("Predict Price"):
    # Create a DataFrame for the input
    input_data = {
        'Brand & Model': [brand_model],
        'Varient': [varient],
        'Fuel Type': [fuel_type],
        'Driven Kilometers': [driven_kilometers],
        'Transmission': [transmission],
        'Owner': [owner],
        'Location': [location]
    }
    input_df = pd.DataFrame(input_data)
    
    # Preprocess input like the training data
    # (Ensure the same preprocessing steps are applied)
    input_df['Driven Kilometers'] = input_df['Driven Kilometers'].astype(float)
    input_df = pd.get_dummies(input_df, columns=['Brand & Model', 'Varient', 'Fuel Type', 'Transmission', 'Owner', 'Location'], drop_first=True)
    
    # Align input_df with training data columns
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]
    
    # Predict
    predicted_price = model.predict(input_df)
    st.success(f"The predicted price of the car is ₹{predicted_price[0]:,.2f}")
