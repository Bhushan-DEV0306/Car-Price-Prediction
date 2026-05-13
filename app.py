import streamlit as st
import pandas as pd
import pickle
import os

# ----------------------------
# LOAD MODEL + COLUMNS SAFELY
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
columns_path = os.path.join(BASE_DIR, "columns.pkl")

model = pickle.load(open(model_path, "rb"))
model_columns = pickle.load(open(columns_path, "rb"))

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Enter car details below:")

# ----------------------------
# INPUTS
# ----------------------------
year = st.number_input("Year", 1990, 2026, 2015)
km_driven = st.number_input("KM Driven", 0, 1000000, 50000)
mileage = st.number_input("Mileage", 0.0, 50.0, 18.0)
engine = st.number_input("Engine (CC)", 500, 5000, 1200)
max_power = st.number_input("Max Power", 10, 500, 80)
torque = st.number_input("Torque", 0, 500, 100)
seats = st.number_input("Seats", 2, 10, 5)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
seller = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
brand = st.selectbox("Brand", [
    "Maruti","Hyundai","Mahindra","Tata","Honda","Ford",
    "Toyota","Chevrolet","Renault","Volkswagen","Nissan",
    "Skoda","Datsun","Mercedes-Benz","BMW","Fiat","Audi",
    "Jeep","Mitsubishi","Volvo","Jaguar","Force","Isuzu",
    "Ambassador","Daewoo","Land","MG","Kia","Lexus",
    "Ashok","Opel"
])

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
car_age = 2026 - year

# Fuel encoding
fuel_diesel = 1 if fuel == "Diesel" else 0
fuel_lpg = 1 if fuel == "LPG" else 0
fuel_petrol = 1 if fuel == "Petrol" else 0

# Seller encoding
seller_individual = 1 if seller == "Individual" else 0
seller_dealer = 1 if seller == "Dealer" else 0
seller_trustmark = 1 if seller == "Trustmark Dealer" else 0

# Transmission encoding
trans_manual = 1 if transmission == "Manual" else 0

# Brand encoding
brands = [
    "Maruti","Hyundai","Mahindra","Tata","Honda","Ford",
    "Toyota","Chevrolet","Renault","Volkswagen","Nissan",
    "Skoda","Datsun","Mercedes-Benz","BMW","Fiat","Audi",
    "Jeep","Mitsubishi","Volvo","Jaguar","Force","Isuzu",
    "Ambassador","Daewoo","Land","MG","Kia","Lexus",
    "Ashok","Opel"
]

brand_dict = {f"brand_{b}": 0 for b in brands}
if brand in brands:
    brand_dict[f"brand_{brand}"] = 1

# ----------------------------
# FINAL INPUT DATA
# ----------------------------
input_data = {
    "year": year,
    "km_driven": km_driven,
    "mileage": mileage,
    "engine": engine,
    "max_power": max_power,
    "torque": torque,
    "seats": seats,
    "car_age": car_age,

    "fuel_Diesel": fuel_diesel,
    "fuel_LPG": fuel_lpg,
    "fuel_Petrol": fuel_petrol,

    "seller_type_Individual": seller_individual,
    "seller_type_Dealer": seller_dealer,
    "seller_type_Trustmark Dealer": seller_trustmark,

    "transmission_Manual": trans_manual,
}

input_data.update(brand_dict)

input_df = pd.DataFrame([input_data])

# 🔥 IMPORTANT FIX (THIS SOLVES YOUR ERROR)
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Predicted Price: ₹ {round(prediction, 2)}")
    except Exception as e:
        st.error(f"Error: {e}")