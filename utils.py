import os
import re

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


# ==========================================================
# Configuration
# ==========================================================

WINDOW_SIZE = 60

MODEL_DIR = "lstm_model"

MODEL_CACHE = {}


# ==========================================================
# Supported Stocks
# ==========================================================

STOCK_MAP = {

    "AAPL": "Apple Inc.",

    "AMZN": "Amazon",

    "GOOGL": "Alphabet Inc.",

    "MSFT": "Microsoft",

    "TSLA": "Tesla"

}


# ==========================================================
# Load LSTM Model
# ==========================================================

def get_model(stock):

    if stock not in MODEL_CACHE:

        model_path = os.path.join(

            MODEL_DIR,

            f"{stock}.h5"

        )

        if not os.path.exists(model_path):

            raise FileNotFoundError(

                f"Model not found : {model_path}"

            )

        MODEL_CACHE[stock] = load_model(

            model_path,

            compile=False

        )

    return MODEL_CACHE[stock]


# ==========================================================
# Parse Period
# Example:
# 6 months
# 2 years
# ==========================================================

def parse_period(text):

    text = text.strip().lower()

    match = re.match(

        r"(\d+(?:\.\d+)?)\s*(month|months|year|years)",

        text

    )

    if not match:

        raise Exception(

            "Use values like "

            "'6 months' or '2 years'."

        )

    value = float(

        match.group(1)

    )

    unit = match.group(2)

    if "month" in unit:

        yahoo_period = f"{max(1,int(value))}mo"

        trading_days = max(

            21,

            int(value*21)

        )

        years = value / 12

    else:

        yahoo_period = f"{max(1,int(value))}y"

        trading_days = max(

            252,

            int(value*252)

        )

        years = value

    return (

        yahoo_period,

        trading_days,

        years

    )


# ==========================================================
# Download Historical Data
# ==========================================================

def download_stock(

    ticker,

    period

):

    df = yf.download(

        ticker,

        period=period,

        auto_adjust=True,

        progress=False

    )

    if df.empty:

        raise Exception(

            f"No market data found for {ticker}"

        )

    df.reset_index(

        inplace=True

    )

    return df


# ==========================================================
# Detect Date Column
# ==========================================================

def detect_date_column(df):

    for col in df.columns:

        parsed = pd.to_datetime(

            df[col],

            errors="coerce"

        )

        if parsed.notna().sum() > len(df) * 0.5:

            df["Date"] = parsed

            return df

    raise Exception(

        "Unable to detect Date column."

    )
    # ==========================================================
# Detect Closing Price Column
# ==========================================================

def detect_price_column(df):
    columns = {str(c).strip().lower(): c for c in df.columns}

    for name in ["close", "adj close", "closing price", "close price"]:
        if name in columns:
            return columns[name]

    raise Exception("Close price column not found.")

    possible_columns = [

        "close",

        "adj close",

        "closing price",

        "close price"

    ]

    for column in possible_columns:

        if column in df.columns:

            return column

    numeric_columns = df.select_dtypes(

        include=np.number

    ).columns.tolist()

    if numeric_columns:

        return numeric_columns[-1]

    raise Exception(

        "Unable to detect closing price column."

    )


# ==========================================================
# Clean Stock Data
# ==========================================================

def clean_stock_data(df):

    # If Date is the index, convert it to a column
    if "Date" not in df.columns:
        df = df.reset_index()

    # Find a date column
    if "Date" not in df.columns:
        for col in df.columns:
            if str(col).lower() in ["date", "datetime"]:
                df.rename(columns={col: "Date"}, inplace=True)
                break

    # Still not found? Detect it automatically
    if "Date" not in df.columns:
        df = detect_date_column(df)

    price_column = detect_price_column(df)

    df["Close"] = pd.to_numeric(df[price_column], errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.dropna(subset=["Date", "Close"])

    df = df.sort_values("Date").reset_index(drop=True)

    return df
# ==========================================================
# Scale Prices
# ==========================================================

def scale_prices(prices):

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(

        prices

    )

    return scaler, scaled


# ==========================================================
# Create Prediction Window
# ==========================================================

def get_prediction_window(

    scaled_prices

):

    if len(scaled_prices) < WINDOW_SIZE:

        raise Exception(

            f"Minimum {WINDOW_SIZE} rows required."

        )

    return scaled_prices[-WINDOW_SIZE:]


# ==========================================================
# Predict Future Prices
# ==========================================================

def predict_future(

    model,

    last_window,

    scaler,

    future_days

):

    predictions = []

    current = last_window.copy()

    for _ in range(future_days):

        X = current.reshape(

            1,

            WINDOW_SIZE,

            1

        )

        prediction = model.predict(

            X,

            verbose=0

        )[0][0]

        predictions.append(

            prediction

        )

        current = np.vstack(

            (

                current[1:],

                [[prediction]]

            )

        )

    predictions = np.array(

        predictions

    ).reshape(

        -1,

        1

    )

    predictions = scaler.inverse_transform(

        predictions

    )

    return predictions.flatten()


# ==========================================================
# Generate Future Business Dates
# ==========================================================

def generate_future_dates(

    last_date,

    future_days

):

    return pd.bdate_range(

        start=last_date,

        periods=future_days + 1

    )[1:]
    # ==========================================================
# Load Data from Yahoo or Uploaded CSV
# ==========================================================

def load_data(

    stock,

    period,

    uploaded_file=None,

    cache_dir="cache"

):

    if uploaded_file and uploaded_file.filename:
        df = pd.read_csv(uploaded_file)
        return clean_stock_data(df)

    cache_file = os.path.join(

        cache_dir,

        f"{stock}.csv"

    )

    if os.path.exists(cache_file):

        try:

            df = pd.read_csv(

                cache_file,

                skiprows=[1]

            )

        except Exception:

            df = pd.read_csv(cache_file)

    else:

        df = download_stock(

            stock,

            period

        )

    return clean_stock_data(df)


# ==========================================================
# Prepare Data for Prediction
# ==========================================================

def prepare_prediction_data(

    df

):

    prices = df["Close"].values.reshape(-1, 1)

    scaler, scaled_prices = scale_prices(prices)

    last_window = get_prediction_window(

        scaled_prices

    )

    return (

        prices,

        scaler,

        last_window

    )


# ==========================================================
# Current Price
# ==========================================================

def get_current_price(

    prices

):

    return float(

        prices[-1][0]

    )


# ==========================================================
# Future Price
# ==========================================================

def get_future_price(

    predictions

):

    return float(

        predictions[-1]

    )


# ==========================================================
# Profit Percentage
# ==========================================================

def calculate_profit(

    current_price,

    future_price

):

    return (

        (

            future_price

            -

            current_price

        )

        /

        current_price

    ) * 100


# ==========================================================
# Complete Prediction Pipeline
# ==========================================================

def run_prediction(

    stock,

    period,

    future_days,

    uploaded_file=None

):

    df = load_data(

        stock,

        period,

        uploaded_file

    )

    prices, scaler, last_window = prepare_prediction_data(df)

    model = get_model(stock)

    predictions = predict_future(

        model,

        last_window,

        scaler,

        future_days

    )

    current_price = get_current_price(

        prices

    )

    future_price = get_future_price(

        predictions

    )

    return {

        "dataframe": df,

        "prices": prices,

        "predictions": predictions,

        "current_price": current_price,

        "future_price": future_price,

        "profit_percent": calculate_profit(

            current_price,

            future_price

        )

    }


# ==========================================================
# Module Exports
# ==========================================================

__all__ = [

    "WINDOW_SIZE",

    "MODEL_DIR",

    "STOCK_MAP",

    "get_model",

    "parse_period",

    "download_stock",

    "detect_date_column",

    "detect_price_column",

    "clean_stock_data",

    "scale_prices",

    "get_prediction_window",

    "predict_future",

    "generate_future_dates",

    "load_data",

    "prepare_prediction_data",

    "get_current_price",

    "get_future_price",

    "calculate_profit",

    "run_prediction"

]