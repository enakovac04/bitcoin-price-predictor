import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def fetch_bitcoin_data(start_date="2021-01-01"):
    """
    Fetch historical Bitcoin price data from Yahoo Finance up to the current date.
    """
    print("Starting data fetch...")

    try:
        # Dynamically set the end date as today
        end_date = datetime.today().strftime('%Y-%m-%d')

        # Download BTC data from Yahoo Finance
        btc = yf.download("BTC-USD", start=start_date, end=end_date, interval="1d")
        print("✅ Data fetched successfully!")

        # Reset index to ensure "Date" is a proper column
        btc.reset_index(inplace=True)

        # Rename columns correctly
        btc.columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"][:len(btc.columns)]

        # Get absolute path to the data directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")

        # Ensure the 'data' directory exists
        os.makedirs(data_dir, exist_ok=True)

        # Save to CSV without extra headers
        file_path = os.path.join(data_dir, "bitcoin_prices.csv")
        btc.to_csv(file_path, index=False)  # index=False prevents extra headers

        print(f"✅ Data correctly saved to: {file_path}")  

    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    fetch_bitcoin_data()
