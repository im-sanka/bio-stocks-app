import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import yfinance as yf
from datetime import datetime, timedelta

# Function to fetch data
# def fetch_data(symbol, start_date=None, end_date=None):
#     return yf.Ticker(symbol).history(start=start_date, end=end_date)

def execute_lstm(data):
    # Function to preprocess data
    def preprocess_data(df, start_row, end_row):
        training_set = df['Open'].iloc[start_row:end_row].values
        training_set = training_set.reshape(-1, 1)
        sc = MinMaxScaler(feature_range=(0, 1))
        return sc.fit_transform(training_set), sc

    # Function to build the model
    def build_model(input_shape):
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # User Input
    col1, col2 = st.columns(2)

    df = data
    if df.empty or df['Open'].isna().any():
        st.error("The dataset is empty or contains NaN values. Cannot proceed with prediction.")
        st.stop()
    if len(df) < 61:
        st.error("Not enough data to make a prediction. Choose a longer date range.")
        st.stop()

    epochs = col1.slider('Number of epochs:', min_value=1, max_value=50, value=10)
    training_data_percent = col1.slider('Percentage of data for training:', min_value=1, max_value=99, value=80)
    forecast_days = col1.slider('Days to forecast:', min_value=1, max_value=60, value=30)
    col2.image("lstm-prediction.png")
    col2.write("This is the example of stock price prediction using LSTM method.")


    go_lstm = st.button("Predict with LSTM now!")
    if go_lstm:
        # Calculate training and test data size
        total_data_size = len(df)
        training_data_size = int(total_data_size * (training_data_percent / 100))
        test_data_size = total_data_size - training_data_size

        # Preprocess the data
        training_data, sc = preprocess_data(df, 0, training_data_size)
        test_data = df['Open'].iloc[training_data_size:].values.reshape(-1,1)


    # Create X_train and y_train
        X_train = []
        y_train = []
        for i in range(60, training_data_size):
            X_train.append(training_data[i - 50:i, 0])
            y_train.append(training_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build and train the model
        model = build_model((X_train.shape[1], 1))
        early_stop = EarlyStopping(monitor='loss', patience=5, verbose=0)
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, callbacks=[early_stop], verbose=0)

        # Create test set
        inputs = df['Open'].iloc[len(df) - len(test_data) - 60:].values.reshape(-1, 1)
        inputs = sc.transform(inputs)
        X_test = []
        for i in range(60, len(inputs)):
            X_test.append(inputs[i - 50:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Prediction
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        # Evaluation
        mse = mean_squared_error(test_data[:len(predicted_stock_price)], predicted_stock_price)
        mae = mean_absolute_error(test_data[:len(predicted_stock_price)], predicted_stock_price)

        st.write(f'Mean Squared Error: {mse}')
        # Rule-based explanation
        if mse < 0.01:
            st.info("The MSE is very low, indicating that the model has done a good job in prediction.")
        elif mse < 0.1:
            st.info("The MSE is low, suggesting the model is fairly accurate.")
        else:
            st.info("The MSE is high, suggesting that the model may not fit well to the data or there are large outliers.")

        st.write(f'Mean Absolute Error: {mae}')
        if mae < 0.01:
            st.info("The MAE is very low, indicating that the model is great at prediction.")
        elif mae < 0.1:
            st.info("The MAE is low, suggesting the model is fairly accurate.")
        else:
            st.info("The MAE is high, suggesting the model may not be capturing some aspects of the data.")


        # Create the Plotly figure for plotting real and predicted stock prices
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[training_data_size:], y=test_data.flatten(), mode='lines', name='Real Stock Price'))
        fig.add_trace(go.Scatter(x=df.index[training_data_size:], y=predicted_stock_price.flatten(), mode='lines', name='Predicted Stock Price'))
        fig.update_layout(title='Stock Price Prediction',
                          xaxis_title='Date',
                          yaxis_title='Stock Price',
                          xaxis=dict(showline=True, showgrid=False),
                          yaxis=dict(showline=True, showgrid=False))

        st.plotly_chart(fig, use_container_width=True)

        # Forecasting
        forecast = []
        current_batch = X_test[-1].reshape((1, X_test.shape[1], 1))
        for i in range(forecast_days):
            forecast_value = model.predict(current_batch)[0]
            forecast.append(forecast_value)
            current_batch = np.append(current_batch[:, 1:, :], [[forecast_value]], axis=1)

        forecast = sc.inverse_transform(np.array(forecast).reshape(-1, 1))

        # Prepare date index for the forecast data
        last_date = df.index[-1]
        forecast_index = pd.date_range(start=last_date, periods=forecast_days + 1, inclusive='right')

        # Plotting
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Open'].values, mode='lines', name='Real Stock Price', line=dict(color='dodgerblue')))
        fig.add_trace(go.Scatter(x=df.index[-len(predicted_stock_price):], y=predicted_stock_price.flatten(), mode='lines', name='Predicted Stock Price', line=dict(color='lime')))
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast.flatten(), mode='lines+markers', name='Forecasted Stock Price', line=dict(color='red')))

        fig.update_layout(title='Stock Price Prediction',
                          xaxis_title='Date',
                          yaxis_title='Stock Price',
                          xaxis=dict(showline=True, showgrid=False),
                          yaxis=dict(showline=True, showgrid=False))

        st.plotly_chart(fig, use_container_width=True)

