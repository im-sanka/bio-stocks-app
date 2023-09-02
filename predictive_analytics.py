import streamlit as st
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import plotly.graph_objs as go
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def prediction(data):

    # Initialize session state if it hasn't been initialized
    if 'model_summaries' not in st.session_state:
        st.session_state.model_summaries = {}

    # Allow the user to select the percentage of training data
    training_percentage = st.slider('Select percentage of training data:', min_value=50, max_value=90, value=80)

    # Calculate the log of data
    df_log = np.log(data['Close'])

    # Calculate the index to split the data
    split_index = int(len(df_log) * (training_percentage / 100.0))

    # Split data
    train_data, test_data = df_log[:split_index], df_log[split_index:]

    # Button to generate AutoARIMA models
    if st.subheader('Generate AutoARIMA Models'):

        model_configs = [
            (0,1,0),
            (0,1,1),
            (1,1,0),
            (1,1,1)
        ]

        model_summaries = {}

        for config in model_configs:
            model = auto_arima(train_data, start_p=config[0], start_q=config[2],
                               max_p=3, max_q=3,
                               m=1, d=config[1],
                               seasonal=False,
                               trace=False,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)

            model_summaries[f"ARIMA{config}"] = model

        st.session_state.model_summaries = model_summaries

    if st.session_state.model_summaries:
        selected_model_str = st.selectbox('Select the ARIMA model:', list(st.session_state.model_summaries.keys()))
        selected_model_tuple = tuple(map(int, re.findall("\d+", selected_model_str)))


        # Fit ARIMA model
        model = ARIMA(train_data, order=selected_model_tuple)
        fitted = model.fit()

        # Get forecast
        forecast_results = fitted.get_forecast(steps=len(test_data))
        fc = forecast_results.predicted_mean
        se = forecast_results.se_mean
        conf = forecast_results.conf_int(alpha=0.05)

        # Align time index
        fc.index = test_data.index
        lower_series = conf.iloc[:, 0]
        upper_series = conf.iloc[:, 1]
        lower_series.index = test_data.index
        upper_series.index = test_data.index

        # Create Plotly figure for ARIMA predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_data.index, y=train_data, mode='lines', name='Train Data'))
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data, mode='lines', name='Test Data'))
        fig.add_trace(go.Scatter(x=fc.index, y=fc, mode='lines', name='Predicted'))
        fig.add_trace(go.Scatter(x=lower_series.index, y=lower_series, mode='lines', name='Lower CI', line=dict(dash='dash', color='green')))
        fig.add_trace(go.Scatter(x=upper_series.index, y=upper_series, mode='lines', name='Upper CI', line=dict(dash='dash', color='green')))
        fig.update_layout(title='ARIMA Forecast', xaxis_title='Date', yaxis_title='Log of Closing Prices')
        st.plotly_chart(fig, use_container_width=True)


        st.subheader("Forecast evaluation")

        mse = mean_squared_error(test_data, fc)
        st.write(f'MSE (Mean Squared Error): {round(mse, 4)}')
        if mse < 0.05:
            st.info('The MSE is low, indicating the model has a good fit to the data.')
        elif mse < 0.2:
            st.info('The MSE is moderate, which suggests the model fits the data fairly well.')
        else:
            st.info('The MSE is high, meaning the model may not fit the data well.')

        mae = mean_absolute_error(test_data, fc)
        st.write(f'MAE (Mean Absolute Error): {round(mae, 4)}')
        if mae < 0.05:
            st.info('The MAE is low, which means the average prediction error is small.')
        elif mae < 0.2:
            st.info('The MAE is moderate, so the model generally makes acceptable predictions.')
        else:
            st.info('The MAE is high, indicating the model may have issues with prediction accuracy.')

        rmse = math.sqrt(mean_squared_error(test_data, fc))
        st.write(f'RMSE (Root Mean Squared Error): {round(rmse, 4)}')
        if rmse < 0.05:
            st.info('The RMSE is low, suggesting the model fits the data well.')
        elif rmse < 0.2:
            st.info('The RMSE is moderate, which means the model is fairly reliable.')
        else:
            st.info('The RMSE is high, which could mean the model is unreliable for this data.')

        mape = np.mean(np.abs(fc - test_data) / np.abs(test_data))
        st.write(f'MAPE (Mean Absolute Percentage Error): {round(mape, 4)}')
        if mape < 0.05:
            st.info('The MAPE is below 5%, indicating excellent predictive accuracy.')
        elif mape < 0.2:
            st.info('The MAPE is between 5% and 20%, which is generally considered good.')
        else:
            st.info('The MAPE is above 20%, which means the model may not be very accurate.')

