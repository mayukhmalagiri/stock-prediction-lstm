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

MODEL_CACHE = {}

# -----------------------------------
# LOAD MODEL
# -----------------------------------

def get_model(stock):

    if stock not in MODEL_CACHE:

        model_path = os.path.join(MODEL_DIR, f"{stock}.h5")

        MODEL_CACHE[stock] = load_model(model_path, compile=False)

    return MODEL_CACHE[stock]


# -----------------------------------
# PREDICTION FUNCTION
# -----------------------------------

def predict_future(model, last_window, scaler, future_days):

    window = last_window.copy()
    predictions = []

    for _ in range(future_days):

        X = window.reshape(1, WINDOW_SIZE, 1)

        pred = model.predict(X, verbose=0)[0][0]

        predictions.append(pred)

        window = np.vstack((window[1:], [[pred]]))

    predictions = np.array(predictions).reshape(-1,1)

    return scaler.inverse_transform(predictions)


# -----------------------------------
# HOME
# -----------------------------------

@app.route("/", methods=["GET","POST"])
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

            # -----------------------------------
            # LOAD CSV
            # -----------------------------------

            path = os.path.join(CACHE_DIR, f"{selected_stock}.csv")

            # skip ticker row
            df = pd.read_csv(path, skiprows=[1])

            df["Date"] = pd.to_datetime(df["Date"])
            df["Close"] = pd.to_numeric(df["Close"])

            df = df.sort_values("Date").reset_index(drop=True)

            prices = df["Close"].values.reshape(-1,1)

            # -----------------------------------
            # SCALE DATA
            # -----------------------------------

            scaler = MinMaxScaler()

            scaled_prices = scaler.fit_transform(prices)

            # -----------------------------------
            # MODEL
            # -----------------------------------

            model = get_model(selected_stock)

            last_window = scaled_prices[-WINDOW_SIZE:]

            future_days = FUTURE_DAYS_MAP[selected_future]

            predictions = predict_future(
                model,
                last_window,
                scaler,
                future_days
            )

            future_price = predictions[-1][0]
            current_price = prices[-1][0]

            profit_percent = ((future_price-current_price)/current_price)*100

            # -----------------------------------
            # DECISION
            # -----------------------------------

            years = int(selected_future[0])

            yearly_profit = profit_percent/years

            if yearly_profit >= 12:
                decision="Long-Term Investment"
                color="green"
            elif yearly_profit >= 4:
                decision="Moderate Investment"
                color="orange"
            else:
                decision="Not Recommended"
                color="red"

            # -----------------------------------
            # HISTORICAL GRAPH DATA
            # -----------------------------------

            if selected_past=="6m":
                days=126
            elif selected_past=="1y":
                days=252
            else:
                days=756

            hist_df = df.iloc[-days:]

            hist_dates = hist_df["Date"]
            hist_prices = hist_df["Close"]

            last_date = hist_dates.iloc[-1]
            last_price = hist_prices.iloc[-1]

            # -----------------------------------
            # FUTURE GRAPH DATA
            # -----------------------------------

            future_dates = pd.bdate_range(start=last_date, periods=future_days+1)

            future_prices = np.linspace(last_price, future_price, len(future_dates))

            # -----------------------------------
            # PLOTLY GRAPH
            # -----------------------------------

            hist_trace = go.Scatter(
                x=hist_dates,
                y=hist_prices,
                mode="lines",
                name="Historical Price",
                line=dict(color="blue", width=3)
            )

            future_trace = go.Scatter(
                x=future_dates,
                y=future_prices,
                mode="lines",
                name="Predicted Price",
                line=dict(color="red", width=3, dash="dash")
            )

            fig = go.Figure(data=[hist_trace, future_trace])

            fig.update_layout(
                title=f"{selected_stock} Stock Price Forecast",
                xaxis_title="Date",
                yaxis_title="Price per Share ($)",
                template="plotly_white"
            )

            graph_url = pyo.plot(
                fig,
                output_type="div",
                include_plotlyjs=False
            )

            result = {
                "stock": selected_stock,
                "current_price": round(current_price,2),
                "future_price": round(future_price,2),
                "profit_percent": round(profit_percent,2),
                "decision": decision,
                "color": color,
                "years": selected_future
            }

        except Exception as e:

            result={"error":str(e)}

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

    port = int(os.environ.get("PORT",5000))

    app.run(host="0.0.0.0", port=port)