import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
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

def predict_price(n_days):
    """
    Predict Bitcoin price for the next 'n_days' using the trained LSTM model.
    """
    last_30_days = df["Close_Scaled"].values[-30:].reshape(1, 30, 1)  # Get the last 30 days
    predicted_prices = []

    for _ in range(n_days):
        predicted_scaled = model.predict(last_30_days)  # Predict the next day
        predicted_price = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))[0][0]
        predicted_prices.append(predicted_price)

        # Shift the input data to include the newly predicted value
        new_input = np.append(last_30_days[:, 1:, :], predicted_scaled.reshape(1, 1, 1), axis=1)
        last_30_days = new_input

    return predicted_prices

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
        .main { background-color: #f8f9fa; }
        h1 {
            color: #ff6600;
            text-align: center;
            font-size: 36px;
        }
        .stAlert {
            font-size: 20px;
            text-align: center;
        }
        .success-box {
            font-size: 22px;
            background-color: #d4edda;
            padding: 15px;
            border-radius: 5px;
        }
        .warning-box {
            font-size: 22px;
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Bitcoin Price Predictor")
st.subheader("🔮 Predict future Bitcoin prices using an advanced LSTM model.")

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
    predicted_prices = predict_price(days_ahead)
    predicted_price = predicted_prices[-1]  # Last predicted value for selected date

    st.markdown(f"""
    <div class="success-box">
        📅 <b>Predicted Bitcoin Price on {selected_date}:</b> <span style="color:#28a745;"><b>${predicted_price:.2f}</b></span>
    </div>
    """, unsafe_allow_html=True)

    # ======== Graph: Show Historical & Next 30-Day Predictions =========

    future_days = 30
    future_dates = [df.index[-1] + pd.Timedelta(days=i) for i in range(1, future_days + 1)]
    future_df = pd.DataFrame({"Date": future_dates, "Predicted Close": predict_price(future_days)})
    future_df.set_index("Date", inplace=True)

    # Plot historical & predicted prices
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index[-100:].to_numpy(), df["Close"][-100:].to_numpy(), label="📉 Actual Prices", color="blue")
    ax.plot(future_df.index.to_numpy(), future_df["Predicted Close"].to_numpy(), label="🔮 Predicted Next 30 Days", linestyle="dashed", color="red")
    ax.set_title("Bitcoin Price Prediction for Next 30 Days")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()

    # Show the updated graph
    st.pyplot(fig)

st.markdown("🚀 *Data fetched from Yahoo Finance and analyzed using Deep Learning (LSTM).*")
