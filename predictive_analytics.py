import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima

st.title('Stock Predictor with Auto ARIMA')

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", "AAPL")

# Download data
st.write(f"Downloading data for {ticker}...")
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')
st.write("Data downloaded.")

# Show raw data
if st.checkbox("Show Raw Data"):
    st.dataframe(data)

# Function to plot predictions
def plot_predictions(y_test, y_pred, future_index):
    fig, ax = plt.subplots()
    ax.plot(y_test.index, y_test.values, color='blue', label='Actual')
    ax.plot(future_index, y_pred, color='red', linestyle='dashed', label='Predicted')
    ax.legend()
    return fig

# Configuration for auto_arima
st.sidebar.header("Auto ARIMA Configuration")
start_p = st.sidebar.slider("start_p", 1, 5, 1)
start_q = st.sidebar.slider("start_q", 1, 5, 1)
max_p = st.sidebar.slider("max_p", 1, 5, 3)
max_q = st.sidebar.slider("max_q", 1, 5, 3)
m = st.sidebar.slider("Seasonality (m)", 1, 24, 12)
start_P = st.sidebar.slider("start_P", 0, 5, 0)
d = st.sidebar.slider("d", 0, 2, 1)
D = st.sidebar.slider("D", 0, 2, 1)
trace = st.sidebar.checkbox("Trace", True)
error_action = st.sidebar.selectbox("error_action", ["ignore", "warn", "raise", "trace"])
suppress_warnings = st.sidebar.checkbox("Suppress Warnings", True)
stepwise = st.sidebar.checkbox("Stepwise", True)

# Apply auto_arima
if st.button("Run Auto ARIMA"):
    st.write("Running Auto ARIMA...")
    stepwise_model = auto_arima(data['Close'], start_p=start_p, start_q=start_q,
                                max_p=max_p, max_q=max_q, m=m,
                                start_P=start_P, seasonal=True,
                                d=d, D=D, trace=trace,
                                error_action=error_action,
                                suppress_warnings=suppress_warnings,
                                stepwise=stepwise)

    st.write(f"AIC: {stepwise_model.aic()}")

    # Make prediction
    n_periods = st.slider("Select Number of Periods for Prediction:", 10, 100, 30)
    future_forecast, conf_int = stepwise_model.predict(n_periods=n_periods, return_conf_int=True)
    future_index = pd.date_range(data.index[-1], periods=n_periods+1, closed='right')

    # Plotting
    st.pyplot(plot_predictions(data['Close'].iloc[-n_periods:], future_forecast, future_index))

