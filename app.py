import os
import numpy as np
import pandas as pd

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import plotly.graph_objs as go
import plotly.offline as pyo
import yfinance as yf

app = Flask(__name__)

CACHE_DIR = "cache"
MODEL_DIR = "lstm_model"
WINDOW_SIZE = 60

STOCK_MAP = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "GOOGL": "Google",
    "MSFT": "Microsoft",
    "TSLA": "Tesla"
}

FUTURE_DAYS_MAP = {
    "1y": 252,
    "3y": 756,
    "5y": 1260
}

MODEL_CACHE = {}


def get_model(stock):
    if stock not in MODEL_CACHE:
        path = os.path.join(MODEL_DIR, f"{stock}.h5")
        MODEL_CACHE[stock] = load_model(path, compile=False)

    return MODEL_CACHE[stock]


def predict_future(model, window, scaler, days):

    predictions = []
    current_window = window.copy()

    for _ in range(days):

        X = current_window.reshape(1, WINDOW_SIZE, 1)

        pred = model.predict(X, verbose=0)[0][0]

        predictions.append(pred)

        current_window = np.vstack((current_window[1:], [[pred]]))

    predictions = np.array(predictions).reshape(-1, 1)

    return scaler.inverse_transform(predictions)


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    graph_url = None

    selected_stock = ""
    selected_past = ""
    selected_future = ""

    if request.method == "POST":

        selected_stock = request.form.get("stock")
        selected_past = request.form.get("past_range")
        selected_future = request.form.get("future_range")

        manual_stock = request.form.get("manual_stock")
        uploaded_file = request.files.get("stock_file")

        try:

            # DATA SOURCE SELECTION

            if selected_stock == "UPLOAD" and uploaded_file:

                df = pd.read_csv(uploaded_file)

            elif selected_stock == "MANUAL" and manual_stock:

                df = yf.download(manual_stock, period=selected_past)
                df.reset_index(inplace=True)
                selected_stock = manual_stock

            else:

                path = os.path.join(CACHE_DIR, f"{selected_stock}.csv")
                df = pd.read_csv(path, skiprows=[1])

            # FIX MULTI INDEX FROM YFINANCE

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # VALIDATE DATA

            if "Close" not in df.columns:
                raise Exception("Close column not found in dataset")

            # CLEAN DATA

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            close_series = pd.Series(df["Close"])
            close_series = pd.to_numeric(close_series, errors="coerce")

            df["Close"] = close_series

            df = df.dropna(subset=["Date", "Close"])
            df = df.sort_values("Date").reset_index(drop=True)

            prices = df["Close"].values.reshape(-1, 1)

            if len(prices) < WINDOW_SIZE:
                raise Exception("Not enough historical data for prediction")

            scaler = MinMaxScaler()
            scaled_prices = scaler.fit_transform(prices)

            # LOAD MODEL

            if selected_stock in STOCK_MAP:
                model = get_model(selected_stock)
            else:
                model = get_model("AAPL")

            # PREDICTION

            last_window = scaled_prices[-WINDOW_SIZE:]

            future_days = FUTURE_DAYS_MAP[selected_future]

            preds = predict_future(model, last_window, scaler, future_days)

            future_price = preds[-1][0]
            current_price = prices[-1][0]

            profit_percent = ((future_price - current_price) / current_price) * 100

            years = int(selected_future[0])
            yearly_profit = profit_percent / years

            # INVESTMENT DECISION

            if yearly_profit >= 12:
                decision = "Long-Term Investment"
                color = "green"

            elif yearly_profit >= 4:
                decision = "Moderate Investment"
                color = "orange"

            else:
                decision = "Not Recommended"
                color = "red"

            # GRAPH

            last_date = df["Date"].iloc[-1]

            future_dates = pd.bdate_range(start=last_date, periods=future_days + 1)

            trace_current = go.Scatter(
                x=[future_dates[0]],
                y=[current_price],
                mode="markers+text",
                marker=dict(color="green", size=14),
                text=["Current Price"],
                textposition="top center",
                name="Current Price"
            )

            trace_prediction = go.Scatter(
                x=[future_dates[-1]],
                y=[future_price],
                mode="markers+text",
                marker=dict(color="red", size=14),
                text=["Predicted Price"],
                textposition="top center",
                name="Prediction"
            )

            fig = go.Figure(data=[trace_current, trace_prediction])

            fig.update_layout(
                title=f"{selected_stock} Price Prediction",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                template="plotly_white"
            )

            graph_url = pyo.plot(fig, output_type="div", include_plotlyjs=False)

            result = {
                "stock": selected_stock,
                "current_price": round(current_price, 2),
                "future_price": round(future_price, 2),
                "profit_percent": round(profit_percent, 2),
                "decision": decision,
                "color": color,
                "years": selected_future
            }

        except Exception as e:
            result = {"error": str(e)}

    return render_template(
        "index.html",
        result=result,
        graph_url=graph_url,
        stocks=STOCK_MAP,
        selected_stock=selected_stock,
        selected_past=selected_past,
        selected_future=selected_future
    )


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)