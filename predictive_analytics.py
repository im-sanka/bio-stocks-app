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

optimization_level = st.sidebar.selectbox(
    "Optimization Level",
    ["Lenient", "Moderate", "Extreme"],
)

if optimization_level == "Lenient":
    p_values = (1, 2)
    q_values = (1, 2)
elif optimization_level == "Moderate":
    p_values = (1, 3)
    q_values = (1, 3)
else:
    p_values = (1, 5)
    q_values = (1, 5)

# Apply auto_arima
if st.button("Run Auto ARIMA"):
    st.write("Running Auto ARIMA...")
    stepwise_model = auto_arima(data['Close'], start_p=p_values[0], start_q=q_values[0],
                                max_p=p_values[1], max_q=q_values[1], m=12,
                                start_P=0, seasonal=True,
                                d=1, D=1, trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)

    st.write(f"AIC: {stepwise_model.aic()}")

    # Make prediction
    n_periods = st.slider("Select Number of Periods for Prediction:", 10, 100, 30)
    future_forecast, conf_int = stepwise_model.predict(n_periods=n_periods, return_conf_int=True)
    future_index = pd.date_range(data.index[-1], periods=n_periods+1, closed='right')

    # Plotting
    st.pyplot(plot_predictions(data['Close'].iloc[-n_periods:], future_forecast, future_index))
