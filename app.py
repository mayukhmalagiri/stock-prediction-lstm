import os
from datetime import datetime

import pandas as pd

from flask import Flask, render_template, request

from utils import (
    STOCK_MAP,
    parse_period,
    load_data,
    prepare_prediction_data,
    get_model,
    predict_future,
    generate_future_dates,
    get_current_price,
    get_future_price,
    calculate_profit
)

from ai import (
    analyze_prediction
)

from graph import (
    create_graph
)


# ==========================================================
# Flask Application
# ==========================================================

app = Flask(__name__)


# ==========================================================
# Configuration
# ==========================================================

CACHE_DIR = "cache"


# ==========================================================
# Home Page
# ==========================================================

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

        try:

            # =====================================
            # Read Form
            # =====================================

            selected_stock = request.form.get(

                "stock"

            )

            selected_past = request.form.get(

                "past_range"

            )

            selected_future = request.form.get(

                "future_range"

            )

            manual_stock = request.form.get(

                "manual_stock"

            )

            uploaded_file = request.files.get(

                "stock_file"

            )

            # =====================================
            # Parse Periods
            # =====================================

            (

                yahoo_period,

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

            # =====================================
            # Manual Stock
            # =====================================

            if (

                selected_stock == "MANUAL"

                and manual_stock

            ):

                selected_stock = (

                    manual_stock

                    .strip()

                    .upper()

                )

            # =====================================
            # Load Dataset
            # =====================================

            df = load_data(

                stock=selected_stock,

                period=yahoo_period,

                uploaded_file=uploaded_file,

                cache_dir=CACHE_DIR

            )

            # =====================================
            # Prepare Prediction
            # =====================================

            (

                prices,

                scaler,

                last_window

            ) = prepare_prediction_data(

                df

            )

            # =====================================
            # Load LSTM Model
            # =====================================

            model_stock = (

                selected_stock

                if selected_stock in STOCK_MAP

                else "AAPL"

            )

            model = get_model(

                model_stock

            )
                        # =====================================
            # Run LSTM Prediction
            # =====================================

            predictions = predict_future(

                model=model,

                last_window=last_window,

                scaler=scaler,

                future_days=future_days

            )

            # =====================================
            # Prices
            # =====================================

            current_price = get_current_price(

                prices

            )

            future_price = get_future_price(

                predictions

            )

            total_profit = calculate_profit(

                current_price,

                future_price

            )

            yearly_profit = (

                total_profit

                /

                future_years

            )

            # =====================================
            # Company Name
            # =====================================

            company = STOCK_MAP.get(

                selected_stock,

                selected_stock

            )

            # =====================================
            # AI Analysis
            # =====================================

            ai_result = analyze_prediction(

                prices=prices,

                current_price=current_price,

                future_price=future_price,

                yearly_profit=yearly_profit,

                total_profit=total_profit,

                company=company,

                selected_past=selected_past,

                selected_future=selected_future

            )

            # =====================================
            # Future Trading Dates
            # =====================================

            future_dates = generate_future_dates(

                df["Date"].iloc[-1],

                future_days

            )

            # =====================================
            # Generate Plotly Graph
            # =====================================

            graph_url = create_graph(

                historical_dates=df["Date"],

                historical_prices=df["Close"],

                future_dates=future_dates,

                future_prices=predictions,

                current_price=current_price

            )

            # =====================================
            # Current Time
            # =====================================

            now = datetime.now()

            generated_on = now.strftime(

                "%d %B %Y"

            )

            generated_time = now.strftime(

                "%I:%M %p"

            )

            # =====================================
            # Build Result Dictionary
            # =====================================

            result = {

                "stock": selected_stock,

                "company": company,

                "current_price": round(

                    current_price,

                    2

                ),

                "future_price": round(

                    future_price,

                    2

                ),

                "profit_percent": round(

                    total_profit,

                    2

                ),

                "yearly_profit": round(

                    yearly_profit,

                    2

                ),

                "investment_score": ai_result["score"],

                "confidence": ai_result["confidence"],

                "confidence_title": ai_result["confidence_title"],

                "confidence_text": ai_result["confidence_text"],

                "risk": ai_result["risk"],

                "risk_text": ai_result["risk_text"],

                "decision": ai_result["decision"],

                "badge": ai_result["badge"],

                "sentiment": ai_result["sentiment"],

                "sentiment_color": ai_result["sentiment_color"],

                "outlook": ai_result["outlook"],

                "insights": ai_result["insights"],

                "timeline": ai_result["timeline"],

                "summary": ai_result["summary"],

                "prediction_date": generated_on,

                "prediction_time": generated_time,

                "model": "Watermarked LSTM v2.0"

            }
        except Exception as e:

            import traceback

            traceback.print_exc()

            result = {

            "error": str(e)

            }

    return render_template(

        "index.html",

        result=result,

        graph_url=graph_url,

        stocks=STOCK_MAP,

        selected_stock=selected_stock,

        selected_past=selected_past,

        selected_future=selected_future

    )


# ==========================================================
# Health Check
# ==========================================================

@app.route("/health")

def health():

    return {

        "status": "running",

        "application": "AI Stock Predictor Pro",

        "model": "Watermarked LSTM",

        "timestamp": datetime.now().strftime(

            "%d-%m-%Y %H:%M:%S"

        )

    }


# ==========================================================
# Application Entry Point
# ==========================================================

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