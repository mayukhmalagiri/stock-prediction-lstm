# -----------------------------------
# GRAPH SECTION (PROFESSIONAL)
# -----------------------------------

# determine past range
if selected_past == "6m":
    past_days = 126
elif selected_past == "1y":
    past_days = 252
else:
    past_days = 756

dates = df["date"].iloc[-past_days:]
past_prices = prices.flatten()[-past_days:]

# future predictions
future_line = future_predictions.flatten()

last_date = dates.iloc[-1]

future_dates = pd.date_range(
    start=last_date,
    periods=len(future_line),
    freq="D"
)

# connect future to last past point
future_line = np.insert(future_line, 0, past_prices[-1])
future_dates = np.insert(future_dates, 0, last_date)

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

    # vertical line where prediction starts
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