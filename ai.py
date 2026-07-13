import numpy as np


# ==========================================================
# Calculate Prediction Confidence
# ==========================================================

def calculate_confidence(

    prices,

    current_price

):

    prices = np.array(prices).flatten()

    lookback = min(

        120,

        len(prices)

    )

    recent = prices[-lookback:]

    volatility = np.std(

        recent

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


# ==========================================================
# Calculate Risk Level
# ==========================================================

def calculate_risk(

    volatility,

    confidence

):

    score = (

        volatility * 100

    ) / confidence

    if score < 0.15:

        return "Low"

    elif score < 0.30:

        return "Medium"

    else:

        return "High"


# ==========================================================
# Investment Score
# ==========================================================

def investment_score(

    expected_return,

    confidence,

    risk

):

    score = 50

    score += expected_return * 0.80

    score += (confidence - 70) * 1.20

    if risk == "Low":

        score += 10

    elif risk == "Medium":

        score += 5

    score = max(

        0,

        min(

            100,

            score

        )

    )

    return round(score)


# ==========================================================
# Recommendation Engine
# ==========================================================

def recommendation(

    yearly_profit,

    confidence

):

    if yearly_profit >= 20 and confidence >= 90:

        return (

            "Strong Buy",

            "success"

        )

    elif yearly_profit >= 12:

        return (

            "Buy",

            "primary"

        )

    elif yearly_profit >= 5:

        return (

            "Hold",

            "warning"

        )

    elif yearly_profit >= -5:

        return (

            "Reduce",

            "secondary"

        )

    else:

        return (

            "Sell",

            "danger"

        )
        # ==========================================================
# Market Sentiment
# ==========================================================

def market_sentiment(

    expected_return,

    confidence

):

    if expected_return >= 15 and confidence >= 85:

        return (

            "Very Bullish",

            "#10B981"

        )

    elif expected_return >= 7:

        return (

            "Bullish",

            "#22C55E"

        )

    elif expected_return >= 0:

        return (

            "Neutral",

            "#F59E0B"

        )

    elif expected_return >= -8:

        return (

            "Bearish",

            "#F97316"

        )

    else:

        return (

            "Very Bearish",

            "#EF4444"

        )


# ==========================================================
# Investment Outlook
# ==========================================================

def investment_outlook(

    expected_return,

    confidence,

    risk

):

    if expected_return >= 15:

        message = (

            "The AI model expects strong long-term growth. "
            "Current trend indicates favorable investment potential."

        )

    elif expected_return >= 7:

        message = (

            "The stock shows positive growth potential with "
            "moderate upside."

        )

    elif expected_return >= 0:

        message = (

            "The stock appears relatively stable. Holding may "
            "be preferable to aggressive buying."

        )

    else:

        message = (

            "The model predicts weakness in the selected period. "
            "Investors should exercise caution."

        )

    return {

        "message": message,

        "confidence": confidence,

        "risk": risk

    }


# ==========================================================
# AI Insights
# ==========================================================

def generate_insights(

    current_price,

    future_price,

    profit_percent,

    confidence,

    risk

):

    insights = []

    if future_price > current_price:

        insights.append(

            "AI predicts an upward trend based on historical learning."

        )

    else:

        insights.append(

            "AI predicts a downward trend from recent price behaviour."

        )

    insights.append(

        f"Expected return: {profit_percent:.2f}%"

    )

    insights.append(

        f"Prediction confidence: {confidence:.1f}%"

    )

    insights.append(

        f"Estimated investment risk: {risk}"

    )

    if confidence >= 90:

        insights.append(

            "Prediction reliability is very high."

        )

    elif confidence >= 80:

        insights.append(

            "Prediction reliability is good."

        )

    else:

        insights.append(

            "Use technical indicators before making investment decisions."

        )

    return insights


# ==========================================================
# Confidence Description
# ==========================================================

def confidence_description(

    confidence

):

    if confidence >= 90:

        return (

            "Excellent",

            "Prediction is highly reliable."

        )

    elif confidence >= 80:

        return (

            "High",

            "Prediction reliability is good."

        )

    elif confidence >= 70:

        return (

            "Moderate",

            "Prediction should be used with supporting indicators."

        )

    else:

        return (

            "Low",

            "Prediction uncertainty is relatively high."

        )


# ==========================================================
# Risk Description
# ==========================================================

def risk_description(

    risk

):

    if risk == "Low":

        return (

            "Low market volatility with comparatively stable movement."

        )

    elif risk == "Medium":

        return (

            "Moderate volatility expected during the prediction period."

        )

    else:

        return (

            "High volatility detected. Larger price swings are possible."

        )
        # ==========================================================
# Dashboard Summary
# ==========================================================

def dashboard_summary(

    company,

    current_price,

    future_price,

    expected_return,

    confidence,

    risk,

    decision

):

    return {

        "company": company,

        "current_price": round(current_price, 2),

        "predicted_price": round(future_price, 2),

        "expected_return": round(expected_return, 2),

        "confidence": round(confidence, 1),

        "risk": risk,

        "decision": decision

    }


# ==========================================================
# Timeline Generator
# ==========================================================

def prediction_timeline(

    selected_past,

    selected_future,

    current_price,

    decision

):

    return [

        {

            "title": "Historical Data",

            "description": f"{selected_past} of historical stock prices used for AI training."

        },

        {

            "title": "Current Market",

            "description": f"Latest market price : ₹{current_price:.2f}"

        },

        {

            "title": "AI Forecast",

            "description": f"Future prediction generated for {selected_future}."

        },

        {

            "title": "Recommendation",

            "description": decision

        }

    ]


# ==========================================================
# Complete AI Analysis
# ==========================================================

def analyze_prediction(

    prices,

    current_price,

    future_price,

    yearly_profit,

    total_profit,

    company,

    selected_past,

    selected_future

):

    prices = np.array(prices).flatten()

    volatility = np.std(

        prices[-120:]

    ) / current_price

    confidence = calculate_confidence(

        prices,

        current_price

    )

    risk = calculate_risk(

        volatility,

        confidence

    )

    score = investment_score(

        total_profit,

        confidence,

        risk

    )

    decision, badge = recommendation(

        yearly_profit,

        confidence

    )

    sentiment, sentiment_color = market_sentiment(

        total_profit,

        confidence

    )

    outlook = investment_outlook(

        total_profit,

        confidence,

        risk

    )

    insights = generate_insights(

        current_price,

        future_price,

        total_profit,

        confidence,

        risk

    )

    confidence_title, confidence_text = confidence_description(

        confidence

    )

    risk_text = risk_description(

        risk

    )

    summary = dashboard_summary(

        company,

        current_price,

        future_price,

        total_profit,

        confidence,

        risk,

        decision

    )

    timeline = prediction_timeline(

        selected_past,

        selected_future,

        current_price,

        decision

    )

    return {

        "confidence": confidence,

        "risk": risk,

        "score": score,

        "decision": decision,

        "badge": badge,

        "sentiment": sentiment,

        "sentiment_color": sentiment_color,

        "outlook": outlook,

        "insights": insights,

        "confidence_title": confidence_title,

        "confidence_text": confidence_text,

        "risk_text": risk_text,

        "summary": summary,

        "timeline": timeline

    }


# ==========================================================
# Module Exports
# ==========================================================

__all__ = [

    "calculate_confidence",

    "calculate_risk",

    "investment_score",

    "recommendation",

    "market_sentiment",

    "investment_outlook",

    "generate_insights",

    "confidence_description",

    "risk_description",

    "dashboard_summary",

    "prediction_timeline",

    "analyze_prediction"

]