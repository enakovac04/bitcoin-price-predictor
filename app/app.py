import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import datetime

# Load the trained LSTM model
model = tf.keras.models.load_model("../models/lstm_model.keras")  # Ensure this model exists

# Load Bitcoin price data
df = pd.read_csv("../data/processed_bitcoin_prices.csv", parse_dates=["Date"], index_col="Date")

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
df["Close_Scaled"] = scaler.fit_transform(df[["Close"]])

# Function to make predictions
def predict_price(n_days):
    last_30_days = df["Close_Scaled"].values[-30:]  # Get the last 30 days
    input_data = np.array(last_30_days).reshape(1, 30, 1)

    # Make a prediction
    predicted_scaled = model.predict(input_data)
    predicted_price = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))[0][0]

    return predicted_price

# Streamlit UI
st.title("Bitcoin Price Predictor")
st.write("Predict future Bitcoin prices using an LSTM model.")

# User input: Select prediction date
selected_date = st.date_input("Select a date for prediction", datetime.date.today())
days_ahead = (selected_date - datetime.date.today()).days

if days_ahead <= 0:
    st.error("Please select a future date.")
else:
    predicted_price = predict_price(days_ahead)
    st.success(f"Predicted Bitcoin Price on {selected_date}: **${predicted_price:.2f}**")

# Show past vs predicted trends
st.subheader("Bitcoin Price Trends")
st.line_chart(df["Close"])
