import os
import re
import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    request
)

from tensorflow.keras.models import (
    load_model
)

from sklearn.preprocessing import (
    MinMaxScaler
)

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

MODEL_CACHE = {}


def parse_period(text):

    text = text.strip().lower()

    match = re.match(

        r"(\d+(?:\.\d+)?)\s*"
        r"(month|months|year|years)",

        text
    )

    if not match:

        raise Exception(

            "Please enter periods like "
            "'6 months', "
            "'18 months' "
            "or "
            "'2 years'"
        )

    value = float(
        match.group(1)
    )

    unit = match.group(2)

    if "month" in unit:

        yf_period = (
            f"{max(1,int(value))}mo"
        )

        trading_days = max(
            21,
            int(value * 21)
        )

        years = value / 12

    else:

        yf_period = (
            f"{max(1,int(value))}y"
        )

        trading_days = max(
            252,
            int(value * 252)
        )

        years = value

    return (
        yf_period,
        trading_days,
        years
    )


def get_model(stock):

    if stock not in MODEL_CACHE:

        model_path = os.path.join(

            MODEL_DIR,

            f"{stock}.h5"
        )

        MODEL_CACHE[stock] = (

            load_model(
                model_path,
                compile=False
            )
        )

    return MODEL_CACHE[stock]


def download_stock(

    ticker,
    period

):

    df = yf.download(

        ticker,

        period=period,

        auto_adjust=True
    )

    if df.empty:

        raise Exception(

            f"No data returned "
            f"for ticker {ticker}"
        )

    df.reset_index(
        inplace=True
    )

    return df


def predict_future(

    model,

    window,

    scaler,

    future_days

):

    predictions = []

    current_window = (
        window.copy()
    )

    for _ in range(
        future_days
    ):

        X = current_window.reshape(

            1,

            WINDOW_SIZE,

            1
        )

        prediction = (

            model.predict(
                X,
                verbose=0
            )[0][0]
        )

        predictions.append(
            prediction
        )

        current_window = np.vstack(

            (

                current_window[1:],

                [[prediction]]
            )
        )

    predictions = np.array(

        predictions

    ).reshape(-1, 1)

    return scaler.inverse_transform(
        predictions
    )


def calculate_confidence(

    prices,

    current_price

):

    lookback = min(
        120,
        len(prices)
    )

    volatility = np.std(

        prices[-lookback:]
    )

    confidence = (

        95
        -
        (
            volatility
            /
            current_price
        )
        * 100
    )

    confidence = max(
        55,
        min(
            95,
            confidence
        )
    )

    return round(
        confidence,
        1
    )
def determine_risk(
    profit_percent
):

    absolute_profit = abs(
        profit_percent
    )

    if absolute_profit > 50:

        return "High"

    elif absolute_profit > 20:

        return "Medium"

    return "Low"


@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)
def index():

    result = None

    graph_url = None

    selected_stock = ""

    selected_past = ""

    selected_future = ""

    if request.method == "POST":

        selected_stock = (

            request.form.get(
                "stock"
            )
        )

        selected_past = (

            request.form.get(
                "past_range"
            )
        )

        selected_future = (

            request.form.get(
                "future_range"
            )
        )

        manual_stock = (

            request.form.get(
                "manual_stock"
            )
        )

        uploaded_file = (

            request.files.get(
                "stock_file"
            )
        )

        try:

            (
                past_period,
                _,
                _
            ) = parse_period(
                selected_past
            )

            (
                _,
                future_days,
                future_years
            ) = parse_period(
                selected_future
            )

            # ------------------
            # DATA SOURCE
            # ------------------

            if (

                selected_stock
                ==
                "UPLOAD"

                and

                uploaded_file

            ):

                df = pd.read_csv(
                    uploaded_file
                )

            elif (

                selected_stock
                ==
                "MANUAL"

                and

                manual_stock

            ):

                selected_stock = (

                    manual_stock
                    .upper()
                    .strip()
                )

                df = download_stock(

                    selected_stock,

                    past_period
                )

            else:

                cache_path = (

                    os.path.join(

                        CACHE_DIR,

                        f"{selected_stock}.csv"
                    )
                )

                if os.path.exists(
                    cache_path
                ):

                    df = pd.read_csv(

                        cache_path,

                        skiprows=[1]
                    )

                    if len(df) > 2000:

                        df = df.tail(
                            2000
                        )

                else:

                    df = download_stock(

                        selected_stock,

                        past_period
                    )

            # ------------------
            # MULTI INDEX FIX
            # ------------------

            if isinstance(

                df.columns,

                pd.MultiIndex

            ):

                df.columns = (

                    df.columns
                    .get_level_values(
                        0
                    )
                )

            df.columns = [

                str(col)
                .strip()
                .lower()

                for col
                in df.columns
            ]

            # ------------------
            # DATE DETECTION
            # ------------------

            date_column = None

            for col in df.columns:

                parsed = (

                    pd.to_datetime(

                        df[col],

                        errors="coerce"
                    )
                )

                if (

                    parsed.notna()
                    .sum()

                    >

                    len(df)
                    * 0.5
                ):

                    date_column = col

                    df["Date"] = (
                        parsed
                    )

                    break

            if date_column is None:

                raise Exception(

                    "Could not find "
                    "a valid date "
                    "column in "
                    "the dataset."
                )

            # ------------------
            # PRICE DETECTION
            # ------------------

            price_column = None

            price_candidates = [

                "close",

                "adj close",

                "closing price",

                "close price"
            ]

            for candidate in (
                price_candidates
            ):

                if candidate in (
                    df.columns
                ):

                    price_column = (
                        candidate
                    )

                    break

            if price_column is None:

                numeric_columns = (

                    df.select_dtypes(
                        include=np.number
                    ).columns
                )

                if (
                    len(
                        numeric_columns
                    )
                    ==
                    0
                ):

                    raise Exception(

                        "No numeric "
                        "price column "
                        "found."
                    )

                price_column = (

                    numeric_columns[-1]
                )

            df["Close"] = (

                pd.to_numeric(

                    df[
                        price_column
                    ],

                    errors="coerce"
                )
            )
                        # ------------------
            # DATA CLEANING
            # ------------------

            df = df.dropna(
                subset=[
                    "Date",
                    "Close"
                ]
            )

            df = (

                df.sort_values(
                    "Date"
                )

                .reset_index(
                    drop=True
                )
            )

            prices = (

                df["Close"]

                .values

                .reshape(
                    -1,
                    1
                )
            )

            if (

                len(prices)

                <

                WINDOW_SIZE

            ):

                raise Exception(

                    f"Need at least "

                    f"{WINDOW_SIZE} "

                    f"rows of data."
                )

            # ------------------
            # SCALING
            # ------------------

            scaler = (
                MinMaxScaler()
            )

            scaled_prices = (

                scaler.fit_transform(
                    prices
                )
            )

            # ------------------
            # MODEL LOADING
            # ------------------

            if (

                selected_stock

                in

                STOCK_MAP

            ):

                model = get_model(
                    selected_stock
                )

            else:

                model = get_model(
                    "AAPL"
                )

            # ------------------
            # FORECAST
            # ------------------

            last_window = (

                scaled_prices[
                    -WINDOW_SIZE:
                ]
            )

            preds = predict_future(

                model,

                last_window,

                scaler,

                future_days
            )

            future_price = (
                preds[-1][0]
            )

            current_price = (
                prices[-1][0]
            )

            profit_percent = (

                (
                    future_price
                    -
                    current_price
                )

                /

                current_price

            ) * 100

            yearly_profit = (

                profit_percent

                /

                future_years
            )

            confidence = (

                calculate_confidence(

                    prices,

                    current_price
                )
            )

            risk = determine_risk(
                profit_percent
            )

            # ------------------
            # INVESTMENT DECISION
            # ------------------

            if yearly_profit >= 12:

                decision = (
                    "Long-Term Investment"
                )

                color = "green"

            elif yearly_profit >= 4:

                decision = (
                    "Moderate Investment"
                )

                color = "orange"

            else:

                decision = (
                    "Not Recommended"
                )

                color = "red"

            # ------------------
            # GRAPH DATA
            # ------------------

            today = (

                pd.Timestamp
                .today()
                .normalize()
            )

            future_dates = (

                pd.bdate_range(

                    start=today,

                    periods=
                    future_days + 1
                )
            )

            future_prices = (
                preds.flatten()
            )

            trace_current = (
                go.Scatter(

                    x=[
                        future_dates[0]
                    ],

                    y=[
                        current_price
                    ],

                    mode="markers",

                    marker=dict(

                        size=12,

                        color=
                        "#22c55e"
                    ),

                    name=
                    "Current Price"
                )
            )

            trace_forecast = (
                go.Scatter(

                    x=
                    future_dates[1:],

                    y=
                    future_prices,

                    mode=
                    "lines",

                    line=dict(

                        width=4,

                        color=
                        "#38bdf8"
                    ),

                    name=
                    "Forecast"
                )
            )

            fig = go.Figure(

                data=[

                    trace_current,

                    trace_forecast
                ]
            )

            currency = (

                "₹"

                if ".NS"
                in selected_stock

                else "$"
            )

            fig.update_layout(

                title=
                f"{selected_stock} Forecast",

                xaxis_title=
                "Date",

                yaxis_title=
                f"Price ({currency})",

                template=
                "plotly_dark",

                paper_bgcolor=
                "rgba(0,0,0,0)",

                plot_bgcolor=
                "rgba(0,0,0,0)",

                font=dict(

                    color="white",

                    size=14
                ),

                title_font=dict(
                    size=22
                )
            )

            graph_url = (

                pyo.plot(

                    fig,

                    output_type=
                    "div",

                    include_plotlyjs=
                    False
                )
            )

            # ------------------
            # RESULT DATA
            # ------------------

            result = {

                "stock":
                selected_stock,

                "current_price":
                round(
                    current_price,
                    2
                ),

                "future_price":
                round(
                    future_price,
                    2
                ),

                "profit_percent":
                round(
                    profit_percent,
                    2
                ),

                "decision":
                decision,

                "color":
                color,

                "confidence":
                confidence,

                "risk":
                risk,

                "years":
                selected_future.title()
            }

        except Exception as e:

            result = {

                "error":
                str(e)
            }

    return render_template(

        "index.html",

        result=result,

        graph_url=graph_url,

        stocks=STOCK_MAP,

        selected_stock=
        selected_stock,

        selected_past=
        selected_past,

        selected_future=
        selected_future
    )


if __name__ == "__main__":

    port = int(

        os.environ.get(

            "PORT",

            5000
        )
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=True
    )
