# -----------------------------------
# GRAPH SECTION (PROFESSIONAL)
# -----------------------------------

# -----------------------------------
# GRAPH SECTION (PROFESSIONAL)
# -----------------------------------

# determine past range from dropdown
if selected_past == "6m":
    past_days = 126
elif selected_past == "1y":
    past_days = 252
else:
    past_days = 756

# past data
dates = df["date"].iloc[-past_days:]
past_prices = prices.flatten()[-past_days:]

# predicted future values
future_line = future_predictions.flatten()

# last date of past data
last_date = dates.iloc[-1]

# generate future dates
future_dates = pd.date_range(
    start=last_date,
    periods=len(future_line),
    freq="D"
)

# connect future prediction to last past point
future_line = np.insert(future_line, 0, past_prices[-1])
future_dates = np.insert(future_dates, 0, last_date)

# past trace
past_trace = go.Scatter(
    x=dates,
    y=past_prices,
    mode='lines',
    name='Past Prices',
    line=dict(color='blue', width=3)
)

# future trace
future_trace = go.Scatter(
    x=future_dates,
    y=future_line,
    mode='lines',
    name='Predicted Future',
    line=dict(color='red', width=3, dash='dash')
)

# layout
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

# generate graph
fig = go.Figure(data=[past_trace, future_trace], layout=layout)

graph_url = pyo.plot(fig, output_type='div', include_plotlyjs=False)