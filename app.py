import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature names
with open('car_price_model.pkl', 'rb') as file:
    model, feature_names = joblib.load(file)

# Streamlit app title
st.title("Used Car Price Prediction")

# Create input fields for user input
fuel_type = st.selectbox("Fuel Type", ["DIESEL", "PETROL", "CNG", "ELECTRIC"])
driven_km = st.text_input("Driven Kilometers (e.g., 50000)")
transmission = st.selectbox("Transmission", ["MANUAL", "AUTOMATIC"])
owner = st.selectbox("Owner Type", ["1st Owner", "2nd Owner", "3rd Owner", "4th Owner or more"])

# Predict button
if st.button("Predict Price"):
    try:
        # Preprocess user inputs
        data = pd.DataFrame({
            'Driven Kilometers': [float(driven_km)],
            'Fuel Type_DIESEL': [1 if fuel_type == "DIESEL" else 0],
            'Fuel Type_PETROL': [1 if fuel_type == "PETROL" else 0],
            'Fuel Type_CNG': [1 if fuel_type == "CNG" else 0],
            'Fuel Type_ELECTRIC': [1 if fuel_type == "ELECTRIC" else 0],
            'Transmission_AUTOMATIC': [1 if transmission == "AUTOMATIC" else 0],
            'Owner_2nd Owner': [1 if owner == "2nd Owner" else 0],
            'Owner_3rd Owner': [1 if owner == "3rd Owner" else 0],
            'Owner_4th Owner or more': [1 if owner == "4th Owner or more" else 0]
        })

        # Align input data with the feature names
        data = data.reindex(columns=feature_names, fill_value=0)

        # Predict the price
        predicted_price = model.predict(data)[0]
        st.success(f"The estimated price of the car is ₹{round(predicted_price, 2)}")
    except Exception as e:
        st.error(f"Error: {e}. Please check your inputs.")
