import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.offline as pyo



# ==========================================================
# AI STOCK PREDICTION GRAPH
# ==========================================================

def create_graph(

    historical_dates,
    historical_prices,
    future_dates,
    future_prices,
    current_price

):

    # ======================================================
    # Convert Inputs
    # ======================================================

    historical_dates = pd.to_datetime(historical_dates)
    future_dates = pd.to_datetime(future_dates)

    historical_prices = np.asarray(
        historical_prices,
        dtype=float
    ).flatten()

    future_prices = np.asarray(
        future_prices,
        dtype=float
    ).flatten()

    if len(historical_prices) == 0:
        raise Exception("Historical prices not found.")

    if len(future_prices) == 0:
        raise Exception("Prediction not found.")

    # ======================================================
    # Currency
    # ======================================================

    currency = "$"

    # ======================================================
    # Join History & Prediction
    # ======================================================

    prediction_dates = list(future_dates)
    prediction_prices = list(future_prices)

    prediction_dates.insert(
        0,
        historical_dates.iloc[-1]
    )

    prediction_prices.insert(
        0,
        current_price
    )

    # ======================================================
    # Confidence Interval
    # ======================================================

    confidence = np.linspace(
        0.015,
        0.06,
        len(prediction_prices)
    )

    prediction_prices = np.array(prediction_prices)

    upper = prediction_prices * (1 + confidence)
    lower = prediction_prices * (1 - confidence)

    # ======================================================
    # Dynamic Y Axis
    # ======================================================

    all_prices = np.concatenate(

        (

            historical_prices,

            prediction_prices

        )

    )

    ymin = np.min(all_prices) * 0.96
    ymax = np.max(all_prices) * 1.04

    # ======================================================
    # Figure
    # ======================================================

    fig = go.Figure()

    # ======================================================
    # Historical Price
    # ======================================================

    fig.add_trace(

        go.Scatter(

            x=historical_dates,

            y=historical_prices,

            mode="lines",

            name="Historical",

            line=dict(

                color="#4F8EF7",

                width=3

            ),

            fill="tozeroy",

            fillcolor="rgba(79,142,247,0.08)",

            hovertemplate=

            "<b>%{x|%d %b %Y}</b><br>"

            f"Historical : {currency}%{{y:.2f}}"

            "<extra></extra>"

        )

    )

    # ======================================================
    # Current Price Marker
    # ======================================================

    fig.add_trace(

        go.Scatter(

            x=[historical_dates.iloc[-1]],

            y=[current_price],

            mode="markers",

            name="Current",

            marker=dict(

                color="#FFD54F",

                size=14,

                line=dict(

                    color="white",

                    width=2

                )

            ),

            hovertemplate=

            "<b>Current Price</b><br>"

            f"{currency}%{{y:.2f}}"

            "<extra></extra>"

        )

    )

    # ======================================================
    # Current Price Horizontal Line
    # ======================================================

    fig.add_hline(

        y=current_price,

        line_dash="dot",

        line_width=1.5,

        line_color="#FFD54F",

        opacity=0.5

    )

    # ======================================================
    # TODAY Divider
    # ======================================================

    fig.add_vline(

        x=historical_dates.iloc[-1],

        line_dash="dash",

        line_width=2,

        line_color="#64748B"

    )

    # ======================================================
    # Current Price Label
    # ======================================================

    fig.add_annotation(

        x=historical_dates.iloc[-1],

        y=current_price,

        text=f"<b>{currency}{current_price:,.2f}</b>",

        showarrow=False,

        bgcolor="#FFD54F",

        bordercolor="#FFD54F",

        font=dict(

            color="black",

            size=11

        ),

        xshift=55

    )

    # ======================================================
    # Today Label
    # ======================================================

    fig.add_annotation(

        x=historical_dates.iloc[-1],

        y=ymax,

        text="<b>TODAY</b>",

        showarrow=False,

        bgcolor="#334155",

        bordercolor="#334155",

        font=dict(

            color="white",

            size=10

        )

    )
        # ======================================================
    # AI Forecast Line
    # ======================================================

    fig.add_trace(

        go.Scatter(

            x=prediction_dates,

            y=prediction_prices,

            mode="lines",

            name="AI Forecast",

            line=dict(

                color="#00E676",

                width=3,

                dash="dash"

            ),

            hovertemplate=

            "<b>%{x|%d %b %Y}</b><br>"

            f"Predicted : {currency}%{{y:.2f}}"

            "<extra></extra>"

        )

    )



    # ======================================================
    # Confidence Band Upper
    # ======================================================

    fig.add_trace(

        go.Scatter(

            x=prediction_dates,

            y=upper,

            mode="lines",

            line=dict(

                width=0

            ),

            hoverinfo="skip",

            showlegend=False

        )

    )



    # ======================================================
    # Confidence Band Lower
    # ======================================================

    fig.add_trace(

        go.Scatter(

            x=prediction_dates,

            y=lower,

            mode="lines",

            line=dict(

                width=0

            ),

            fill="tonexty",

            fillcolor="rgba(0,230,118,0.18)",

            hoverinfo="skip",

            showlegend=False

        )

    )



    # ======================================================
    # Forecast Background
    # ======================================================

    fig.add_vrect(

        x0=prediction_dates[0],

        x1=prediction_dates[-1],

        fillcolor="rgba(0,230,118,0.05)",

        layer="below",

        line_width=0

    )



    # ======================================================
    # Prediction Start Marker
    # ======================================================

    fig.add_trace(

        go.Scatter(

            x=[prediction_dates[0]],

            y=[prediction_prices[0]],

            mode="markers",

            showlegend=False,

            marker=dict(

                color="#00E676",

                size=10,

                symbol="diamond",

                line=dict(

                    color="white",

                    width=2

                )

            ),

            hovertemplate=

            "<b>Prediction Starts</b><br>"

            f"{currency}%{{y:.2f}}"

            "<extra></extra>"

        )

    )



    # ======================================================
    # Target Marker
    # ======================================================

    fig.add_trace(

        go.Scatter(

            x=[prediction_dates[-1]],

            y=[prediction_prices[-1]],

            mode="markers",

            showlegend=False,

            marker=dict(

                color="#00E676",

                size=18,

                symbol="star",

                line=dict(

                    color="white",

                    width=2

                )

            ),

            hovertemplate=

            "<b>Predicted Target</b><br>"

            f"{currency}%{{y:.2f}}"

            "<extra></extra>"

        )

    )



    # ======================================================
    # Target Annotation
    # ======================================================

    fig.add_annotation(

        x=prediction_dates[-1],

        y=prediction_prices[-1],

        text=f"<b>TARGET</b><br>{currency}{prediction_prices[-1]:,.2f}",

        showarrow=True,

        arrowhead=2,

        arrowwidth=2,

        arrowcolor="#00E676",

        bgcolor="#00E676",

        bordercolor="#00E676",

        font=dict(

            color="black",

            size=11

        ),

        ax=40,

        ay=-45

    )



    # ======================================================
    # Return %
    # ======================================================

    return_percent = (

        (

            prediction_prices[-1]

            -

            current_price

        )

        /

        current_price

    ) * 100



    color = "#00E676"

    if return_percent < 0:

        color = "#EF4444"



    fig.add_annotation(

        x=prediction_dates[-1],

        y=(prediction_prices[-1] + current_price) / 2,

        text=f"<b>{return_percent:.2f}%</b>",

        showarrow=False,

        bgcolor=color,

        bordercolor=color,

        font=dict(

            color="white",

            size=11

        )

    )



    # ======================================================
    # AI Forecast Label
    # ======================================================

    fig.add_annotation(

        x=prediction_dates[2],

        y=prediction_prices[2],

        text="<b>AI FORECAST</b>",

        showarrow=False,

        yshift=25,

        font=dict(

            size=12,

            color="#00E676"

        )

    )
        # ======================================================
    # PROFESSIONAL DARK LAYOUT
    # ======================================================

    fig.update_layout(

        template="plotly_dark",

        paper_bgcolor="#131722",

        plot_bgcolor="#131722",

        height=560,

        margin=dict(

            l=20,

            r=20,

            t=50,

            b=20

        ),

        title=dict(

            text="<b>AI Stock Prediction Dashboard</b>",

            x=0.5,

            xanchor="center",

            font=dict(

                size=19,

                color="white"

            )

        ),

        font=dict(

            family="Segoe UI",

            size=12,

            color="white"

        ),

        hovermode="closest",

        dragmode="zoom",

        showlegend=True,

        legend=dict(

            orientation="h",

            y=-0.16,

            x=0.5,

            xanchor="center",

            bgcolor="rgba(0,0,0,0)",

            borderwidth=0,

            font=dict(

                size=11,

                color="#CBD5E1"

            )

        )

    )



    # ======================================================
    # X AXIS
    # ======================================================

    fig.update_xaxes(

        title="",

        showgrid=True,

        gridcolor="#242D3D",

        linecolor="#3A4556",

        zeroline=False,

        showline=True,

        tickfont=dict(

            color="#CBD5E1",

            size=10

        ),

        tickformat="%b '%y",

        tickangle=0,

        rangeslider_visible=False

    )



    # ======================================================
    # Y AXIS
    # ======================================================

    fig.update_yaxes(

        title="Price",

        tickprefix="$",

        range=[

            ymin,

            ymax

        ],

        showgrid=True,

        gridcolor="#242D3D",

        zeroline=False,

        linecolor="#3A4556",

        tickfont=dict(

            color="#CBD5E1",

            size=10

        )

    )



    # ======================================================
    # BETTER HOVER
    # ======================================================

    fig.update_traces(

        hoverlabel=dict(

            bgcolor="#1E293B",

            bordercolor="#00E676",

            font=dict(

                color="white",

                size=12

            )

        )

    )



    # ======================================================
    # WATERMARK
    # ======================================================

    fig.add_annotation(

        xref="paper",

        yref="paper",

        x=0.99,

        y=0.01,

        text="Watermarked LSTM",

        showarrow=False,

        opacity=0.20,

        font=dict(

            size=10,

            color="#94A3B8"

        )

    )



    # ======================================================
    # REMOVE EXTRA SPACE
    # ======================================================

    fig.update_layout(

        autosize=True

    )
        # ======================================================
    # SHOW ONLY IMPORTANT LEGEND ITEMS
    # ======================================================

    fig.for_each_trace(

        lambda trace: trace.update(

            showlegend=(

                trace.name in [

                    "Historical",

                    "AI Forecast"

                ]

            )

        )

    )



    # ======================================================
    # MODEBAR
    # ======================================================

    config = {

        "displaylogo": False,

        "responsive": True,

        "scrollZoom": True,

        "doubleClick": "reset",

        "displayModeBar": True,

        "modeBarButtonsToRemove":[

            "lasso2d",

            "select2d",

            "hoverClosestCartesian",

            "hoverCompareCartesian",

            "toggleSpikelines",

            "autoScale2d"

        ],

        "toImageButtonOptions":{

            "format":"png",

            "filename":"AI_Stock_Prediction",

            "height":800,

            "width":1600,

            "scale":2

        }

    }



    # ======================================================
    # SMOOTH ANIMATION
    # ======================================================

    fig.update_layout(

        transition=dict(

            duration=500

        )

    )



    # ======================================================
    # REMOVE PLOT MARGINS
    # ======================================================

    fig.update_layout(

        xaxis=dict(

            automargin=True

        ),

        yaxis=dict(

            automargin=True

        )

    )



    # ======================================================
    # SPIKE LINES
    # ======================================================

    fig.update_xaxes(

        showspikes=True,

        spikecolor="#00E676",

        spikesnap="cursor",

        spikemode="across"

    )



    fig.update_yaxes(

        showspikes=True,

        spikecolor="#00E676",

        spikesnap="cursor",

        spikemode="across"

    )



    # ======================================================
    # FINAL RETURN
    # ======================================================

    return pyo.plot(

        fig,

        output_type="div",

        include_plotlyjs=False,

        config=config

    )