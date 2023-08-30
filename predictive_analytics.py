import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima

class StockPredictor:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None
        self.model = None

    def download_data(self):
        st.write(f"Downloading data for {self.ticker}...")
        self.data = yf.download(self.ticker, start='2020-01-01', end='2023-01-01')
        st.write("Data downloaded.")

    def display_data(self):
        if st.checkbox("Show Raw Data"):
            st.dataframe(self.data)

    @st.cache(allow_output_mutation=True)  # Caching the function output
    def run_auto_arima(self, p_values, q_values):
        st.write("Running Auto ARIMA...")
        model = auto_arima(self.data['Close'], start_p=p_values[0], start_q=q_values[0],
                           max_p=p_values[1], max_q=q_values[1], m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
        st.write(f"AIC: {model.aic()}")
        return model

    def predict_and_plot(self, model, n_periods):
        future_forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
        future_index = pd.date_range(self.data.index[-1], periods=n_periods+1, closed='right')
        fig, ax = plt.subplots()
        ax.plot(self.data['Close'].index[-n_periods:], self.data['Close'].values[-n_periods:], label='Actual')
        ax.plot(future_index, future_forecast, linestyle='dashed', label='Predicted')
        ax.legend()
        st.pyplot(fig)

# Streamlit App
st.title('Stock Predictor with Auto ARIMA')

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", "AAPL")
stock_predictor = StockPredictor(ticker)

# Download data and display
stock_predictor.download_data()
stock_predictor.display_data()

# Sidebar for ARIMA configuration
st.sidebar.header("Auto ARIMA Configuration")
optimization_level = st.sidebar.selectbox(
    "Optimization Level",
    ["Lenient", "Moderate", "Extreme"],
)

p_values, q_values = (1, 2), (1, 2)  # Default values
if optimization_level == "Moderate":
    p_values, q_values = (1, 3), (1, 3)
elif optimization_level == "Extreme":
    p_values, q_values = (1, 5), (1, 5)

# Run Auto ARIMA and Predict
if st.button("Run Auto ARIMA"):
    model = stock_predictor.run_auto_arima(p_values, q_values)

if 'model' in locals():
    n_periods = st.slider("Select Number of Periods for Prediction:", 10, 100, 30)
    stock_predictor.predict_and_plot(model, n_periods)
