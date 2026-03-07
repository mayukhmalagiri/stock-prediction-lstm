# app.py
# -----------------------------------
# Flask App for LSTM-based Stock Prediction
# -----------------------------------

import os
import numpy as np
import pandas as pd

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import plotly.graph_objs as go
import plotly.offline as pyo

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

            # Date column
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            else:
                df["date"] = pd.date_range(start="2000-01-01", periods=len(df))

            # Price column
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

            last_window = scaled_prices[-WINDOW_SIZE:].flatten()

            future_days = FUTURE_DAYS_MAP[selected_future]

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
            # GRAPH SECTION
            # -----------------------------------

            if selected_past == "6m":
                past_days = 126
            elif selected_past == "1y":
                past_days = 252
            else:
                past_days = 756

            dates = df["date"].iloc[-past_days:]
            past_prices = prices.flatten()[-past_days:]

            future_line = future_predictions.flatten()
            future_line = future_line - future_line[0] + past_prices[-1]

            last_date = dates.iloc[-1]

            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),  periods=len(future_line), freq="D")

            # connect future to past
            future_line = np.concatenate([[past_prices[-1]], future_line])
            future_dates = list(future_dates)
            future_dates.insert(0, last_date)

            past_trace = go.Scatter(
                x=dates,
                y=past_prices,
                mode='lines',
                name='Past Prices',
                line=dict(color='blue', width=3)
            )

            future_trace = go.Scatter(
                x=future_dates,
                y=future_line,
                mode='lines',
                name='Predicted Future',
                line=dict(color='red', width=3, dash='dash')
            )

            layout = go.Layout(
                title=f"{selected_stock} Stock Price Forecast",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Price per Share ($)"),
                template="plotly_white",
                shapes=[
                    dict(
                        type="line",
                        x0=last_date,
                        x1=last_date,
                        y0=min(past_prices),
                        y1=max(future_line),
                        line=dict(color="gray", dash="dot")
                    )
                ]
            )

            fig = go.Figure(data=[past_trace, future_trace], layout=layout)

            graph_url = pyo.plot(fig, output_type='div', include_plotlyjs=False)

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