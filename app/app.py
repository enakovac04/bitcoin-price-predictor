import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# ======== Load Model & Data =========

# Load the trained LSTM model
try:
    model = tf.keras.models.load_model("../models/lstm_model.keras")  # Ensure this model exists
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Load Bitcoin price data
df = pd.read_csv("../data/processed_bitcoin_prices.csv", parse_dates=["Date"], index_col="Date")

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
df["Close_Scaled"] = scaler.fit_transform(df[["Close"]])

# ======== Prediction Function =========

# Store past predictions so we don't recalculate them
prediction_cache = {}

def predict_price(n_days):
    """
    Predict Bitcoin price for the next 'n_days' using the trained LSTM model.
    Uses caching to store previously predicted values for efficiency.
    """
    if n_days in prediction_cache:
        return prediction_cache[n_days]  # Return cached result if available

    last_30_days = df["Close_Scaled"].values[-30:].reshape(1, 30, 1)  # Get the last 30 days
    predicted_prices = []

    for _ in range(n_days):
        predicted_scaled = model.predict(last_30_days)  # Predict the next day
        predicted_price = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))[0][0]
        predicted_prices.append(predicted_price)

        # Shift input to include the new predicted value
        new_input = np.append(last_30_days[:, 1:, :], predicted_scaled.reshape(1, 1, 1), axis=1)
        last_30_days = new_input

    prediction_cache[n_days] = predicted_prices[-1]  # Store in cache
    return predicted_prices[-1]  # Return final predicted price


def get_actual_price(date):
    """ Fetch actual Bitcoin price from Yahoo Finance for the selected past date. """
    btc = yf.Ticker("BTC-USD")
    history = btc.history(start=date, end=date + datetime.timedelta(days=1))
    if not history.empty:
        return history["Close"].values[0]
    return None

# ======== Streamlit UI =========

# Custom Styling
st.markdown("""
    <style>
        body { background-color: #f8f9fa; }
        .stApp {
            background: linear-gradient(to right, #f8f9fa, #e9ecef);
            padding: 20px;
        }
        h1 {
            color: #343a40;
            text-align: center;
            font-size: 42px;
            font-weight: bold;
        }
        .subheader {
            color: #6c757d;
            text-align: center;
            font-size: 20px;
        }
        .success-box, .warning-box {
            font-size: 22px;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .success-box {
            background-color: #d4edda;
            border-left: 6px solid #28a745;
        }
        .warning-box {
            background-color: #fff3cd;
            border-left: 6px solid #ffc107;
        }
        .footer {
            text-align: center;
            font-size: 16px;
            color: #6c757d;
            margin-top: 30px;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1>📉 Bitcoin Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">🔮 Predict future Bitcoin prices using an advanced LSTM model.</p>', unsafe_allow_html=True)

# User input: Select prediction date
selected_date = st.date_input("📅 Select a date for prediction:", datetime.date.today())
days_ahead = (selected_date - datetime.date.today()).days

if days_ahead < 0:
    # Handle past date selection
    actual_price = get_actual_price(selected_date)
    if actual_price:
        st.markdown(f"""
        <div class="warning-box">
            ⚠ <b>Warning:</b> The selected date is in the past. This is not a prediction.<br>
            📅 <b>Bitcoin Price on {selected_date}:</b> <span style="color:#ff6600;"><b>${actual_price:.2f}</b></span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠ No data available for the selected past date.")

elif days_ahead == 0:
    st.error("⚠ Please select a future date.")
else:
    predicted_price = predict_price(days_ahead)
    st.markdown(f"""
    <div class="success-box">
        📅 <b>Predicted Bitcoin Price on {selected_date}:</b> <span style="color:#28a745;"><b>${predicted_price:.2f}</b></span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<p class="footer">🚀 Data fetched from Yahoo Finance and analyzed using Deep Learning (LSTM).</p>', unsafe_allow_html=True)
