# app.py
# -----------------------------------
# Flask App for LSTM-based Stock Prediction
# Uses Iterative Multi-Step Forecasting
# -----------------------------------

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import io
import base64

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# -----------------------------------
# CONFIG
# -----------------------------------

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

# -----------------------------------
# ITERATIVE FUTURE PREDICTION
# -----------------------------------

def predict_future(model, last_window, scaler, future_days):

    window = last_window.copy()
    predictions = []

    for _ in range(future_days):
        input_data = window.reshape(1, window.shape[0], 1)
        next_scaled = model.predict(input_data, verbose=0)[0][0]

        predictions.append(next_scaled)
        window = np.append(window[1:], next_scaled)

    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)

    return predictions


# -----------------------------------
# HOME
# -----------------------------------

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

        try:

            # Load CSV
            csv_path = os.path.join(CACHE_DIR, f"{selected_stock}.csv")
            df = pd.read_csv(csv_path)
            df.columns = [c.strip().lower() for c in df.columns]

            if "close" in df.columns:
                prices = df["close"]
            else:
                prices = df["adj close"]

            prices = pd.to_numeric(prices, errors="coerce").dropna()
            prices = prices.values.reshape(-1, 1)

            # Scale
            scaler = MinMaxScaler()
            scaled_prices = scaler.fit_transform(prices)

            # Load model
            model_path = os.path.join(MODEL_DIR, f"{selected_stock}.h5")
            model = load_model(model_path)

            # Last window
            last_window = scaled_prices[-WINDOW_SIZE:].flatten()

            future_days = FUTURE_DAYS_MAP[selected_future]

            # Get future predictions
            future_predictions = predict_future(
                model, last_window, scaler, future_days
            )

            future_price = future_predictions[-1][0]

            current_price = prices[-1][0]

            profit_percent = ((future_price - current_price) / current_price) * 100

            # -----------------------------------
            # DECISION LOGIC
            # -----------------------------------

            years = int(selected_future[0])

            adjusted_profit = profit_percent / years

            if adjusted_profit >= 12:
                decision = "Long-Term Investment"
                decision_color = "green"
            elif adjusted_profit >= 4:
                decision = "Moderate / Short-Term Investment"
                decision_color = "orange"
            else:
                decision = "Not Recommended"
                decision_color = "red"

            volatility = np.std(prices[-60:]) / current_price * 100

            if volatility > 8 and years >= 3:
                decision = "High Risk – Not Recommended"
                decision_color = "red"

            # -----------------------------------
            # GRAPH CREATION
            # -----------------------------------

            past_prices = prices.flatten()

            future_line = future_predictions.flatten()

            x_past = range(len(past_prices))
            x_future = range(len(past_prices), len(past_prices) + len(future_line))

            plt.figure(figsize=(8,4))

            plt.plot(x_past, past_prices, color="blue", label="Past Prices")
            plt.plot(x_future, future_line, color="red", label="Predicted Future")

            plt.legend()
            plt.title(f"{selected_stock} Stock Prediction")
            plt.xlabel("Time")
            plt.ylabel("Price")

            img = io.BytesIO()
            plt.savefig(img, format="png")
            img.seek(0)

            graph_url = base64.b64encode(img.getvalue()).decode()

            plt.close()

            # -----------------------------------

            result = {
                "stock": selected_stock,
                "current_price": round(current_price, 2),
                "future_price": round(future_price, 2),
                "profit_percent": round(profit_percent, 2),
                "decision": decision,
                "color": decision_color,
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


# -----------------------------------
# RUN
# -----------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)