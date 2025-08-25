import os
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import alpaca_trade_api as tradeapi
import pytz
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from kaggle_secrets import UserSecretsClient
API_KEY = os.environ.get('APCA_API_KEY_ID')
SECRET_KEY = os.environ.get('APCA_API_SECRET_KEY')


# ============================
# Configuration
# ============================
BASE_URL = "https://paper-api.alpaca.markets" # Use "https://api.alpaca.markets" for live trading
TICKER = 'CCLD' # Use a real, liquid stock ticker for better results. 'C' (Citigroup) is an example.
INTERVAL = '5Min'
BARS_TO_FETCH = 100 # Needs to be greater than your window_size
PRIMARY_WINDOW_SIZE = 10
CONFIDENCE_THRESHOLD = 0.60
TAKE_PROFIT_PCT = 0.01 # 1%
STOP_LOSS_PCT = 0.008 # 0.8%
NY = 'America/New_York'

# ============================
# Helper Functions
# ============================
def check_market_open(api):
    """Checks if the market is open and waits if necessary."""
    clock = api.get_clock()
    if not clock.is_open:
        time_to_open = (clock.next_open - clock.timestamp).total_seconds()
        print(f"Market is closed. Waiting {time_to_open:.0f} seconds until it re-opens at {clock.next_open.astimezone(pytz.timezone(NY)).strftime('%H:%M:%S')}.")
        time.sleep(time_to_open + 60) # Add a buffer to ensure the market is truly open
    else:
        print("Market is currently open.")

def get_latest_data(api, symbol, timeframe, limit):
    """Fetches latest bars from Alpaca and returns a DataFrame."""
    bars = api.get_bars(symbol, timeframe, limit=limit).df
    bars.index = bars.index.tz_convert(NY)
    bars.columns = [col.capitalize() for col in bars.columns]
    return bars.copy()

def create_features(df):
    """Applies the same feature engineering as the training script."""
    df['returns'] = df['Close'].pct_change()
    df['rsi'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd_diff()
    boll = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['bollinger_wband'] = boll.bollinger_wband()
    minutes_in_day = 24 * 60
    df['time_of_day_sin'] = np.sin(2 * np.pi * (df.index.hour * 60 + df.index.minute) / minutes_in_day)
    df['time_of_day_cos'] = np.cos(2 * np.pi * (df.index.hour * 60 + df.index.minute) / minutes_in_day)
    return df.dropna()

def get_signal(df, primary_model, filter_model):
    """
    Analyzes the latest bar and returns a trading signal.
    Returns 1 for a BUY signal, 0 otherwise.
    """
    features_to_use = ['returns', 'rsi', 'macd', 'bollinger_wband', 'time_of_day_sin', 'time_of_day_cos']
    
    if len(df) < PRIMARY_WINDOW_SIZE:
        print("Not enough data to form a feature vector.")
        return 0, 0
    
    latest_features = df[features_to_use].iloc[-PRIMARY_WINDOW_SIZE:].values.flatten()
    primary_pred = primary_model.predict(latest_features.reshape(1, -1))[0]
    
    if primary_pred == 2:
        primary_probs = primary_model.predict_proba(latest_features.reshape(1, -1))[0]
        confidence = primary_probs.max()
        volatility = df['bollinger_wband'].iloc[-1]
        
        meta_features = np.array([confidence, volatility]).reshape(1, -1)
        filter_confidence = filter_model.predict_proba(meta_features)[0][1]
        
        if filter_confidence > CONFIDENCE_THRESHOLD:
            print(f"✅ Filter passed. Confidence: {filter_confidence:.2f}")
            return 1, filter_confidence
        else:
            print(f"❌ Filter failed. Confidence: {filter_confidence:.2f}")
    
    return 0, 0

# ============================
# Main Trading Loop
# ============================
if __name__ == '__main__':
    if not all([API_KEY, SECRET_KEY]):
        print("Please set your API keys as environment variables.")
        exit()

    try:
        api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
        print(f"Connected to Alpaca API successfully for paper trading.")
        
        primary_model = joblib.load('primary_model.pkl')
        filter_model = joblib.load('filter_model.pkl')
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        exit()

    while True:
        try:
            # --- Wait for market to open if it's closed ---
            check_market_open(api)
            
            account = api.get_account()
            positions = api.list_positions()
            has_position = len(positions) > 0
            
            print(f"\n[{pd.Timestamp.now().tz_localize(NY)}] Checking for new bar...")
            
            df_raw = get_latest_data(api, TICKER, INTERVAL, BARS_TO_FETCH)
            df = create_features(df_raw)
            
            signal, confidence = get_signal(df, primary_model, filter_model)
            
            if signal == 1 and not has_position:
                buying_power = float(account.buying_power)
                quantity_to_buy = int(0.95 * buying_power / df['Close'].iloc[-1])
                
                if quantity_to_buy > 0:
                    try:
                        entry_price = df['Close'].iloc[-1]
                        take_profit_price = round(entry_price * (1 + TAKE_PROFIT_PCT), 2)
                        stop_loss_price = round(entry_price * (1 - STOP_LOSS_PCT), 2)

                        api.submit_order(
                            symbol=TICKER,
                            qty=quantity_to_buy,
                            side='buy',
                            type='market',
                            time_in_force='day',
                            order_class='bracket',
                            take_profit={'limit_price': take_profit_price},
                            stop_loss={'stop_price': stop_loss_price, 'limit_price': stop_loss_price}
                        )
                        print(f"🚀 BUY order placed for {quantity_to_buy} shares of {TICKER}.")
                        print(f"Entry: ${entry_price:.2f}, TP: ${take_profit_price:.2f}, SL: ${stop_loss_price:.2f}")

                    except Exception as e:
                        print(f"Failed to submit order: {e}")
                else:
                    print("Not enough buying power to place an order.")

            elif signal == 0:
                print("No trading signal from the models.")
            
            print(f"Current Account Balance: ${float(account.equity):.2f}, Positions: {len(positions)}")
            
            # Wait for the next 5-minute interval
            time.sleep(300)

        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            # If an error occurs, wait a shorter time and retry
            time.sleep(60)
