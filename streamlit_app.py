import datetime
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.models import LinearRegression, RollingAverage, LSTM, Ensemble
from src.utils.features import TimeFeatures
from src.utils.scaler import MinMaxScaler


CONFIG = joblib.load("models/config.pkl")

# Cache the models and scaler


@st.cache_resource
def get_linear_regression_model() -> LinearRegression:
    model = joblib.load("models/lr.pkl")
    return model


@st.cache_resource
def get_scaler() -> MinMaxScaler:
    scaler = joblib.load("models/scaler.pkl")
    return scaler


def get_ensemble_model(input_size: int, hidden_size: int, num_layers: int, output_size: int, **config) -> Ensemble:
    lr_model = get_linear_regression_model()
    nn_model = LSTM(input_size, hidden_size, num_layers, output_size)
    nn_model.load_state_dict(torch.load("models/lstm.pth"))
    nn_model.eval()
    model = Ensemble(lr_model, nn_model, config['window_size'])
    return model


# Prepare the data
def prepare_data(data: np.ndarray, timestamps: List[datetime.datetime]) -> np.ndarray:
    # Create a dataframe of timestamps and data
    df = pd.DataFrame({'date': timestamps, 'count': data})
    # Convert the date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    # Create time features
    df = TimeFeatures.make_time_features(df)
    # Scale the count column
    scaler = get_scaler()
    df['count'] = scaler.transform(df['count'].values.reshape(-1, 1))
    df.drop(['day_of_week', 'day_of_month'], axis=1, inplace=True)
    features = df.values.astype(np.float32)

    return features


def get_rolling_average_predictions(data: np.ndarray, window_size: int, prediction_length: int) -> np.ndarray:
    rolling_avg = RollingAverage(
        window_size=window_size, prediction_size=prediction_length)
    rolling_avg_preds = rolling_avg.predict(data)

    return rolling_avg_preds


def get_linear_regression_predictions(X: np.ndarray, window_size: int, prediction_length: int) -> np.ndarray:
    model = get_linear_regression_model()
    lr_preds = model.predict(X[:, -2]+30)
    # Unscale the predictions
    scaler = get_scaler()
    lr_preds = scaler.inverse_transform(lr_preds.reshape(-1, 1)).reshape(-1)
    return lr_preds


def get_ensemble_predictions(X: np.ndarray) -> np.ndarray:
    model = get_ensemble_model(**CONFIG)
    preds = model.predict(X)
    # Unscale the predictions
    scaler = get_scaler()
    preds = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(-1)
    return preds

# This is a simple demo of the forecast app


def app():
    """
    This function runs the Forecasting App.

    It allows the user to choose a model type, start time for the forecast, and historical data.
    It then prepares the data, generates predictions using the chosen model, and plots the historical data and predictions.
    Finally, it displays the monthly aggregate predictions based on the chosen model.

    Args:
        None

    Returns:
        None
    """

    # Set the title of the app
    st.title('Fetch Receipts Count Forecasting App')
    st.write('This is a simple demo of the forecast app for receipt count.')
    st.write(
        'Project details can be found [here](https://github.com/connectwithprakash/fetch-machine-learning-exercise/blob/main/README.md).')

    # Sidebar
    # Add a sidebar to choose the model type
    st.sidebar.subheader('Choose Model')
    model_name = st.sidebar.selectbox('Model', [
                                      'All', 'Rolling Average', 'Linear Regression', 'Linear Regression + Neural Network'])

    # Choose start time of the forecast
    st.subheader('Choose the Forecast Start Time')
    start_time = st.date_input('Forecast Start Time', value=datetime.datetime(
        2022, 1, 1), min_value=None, max_value=None, key=None)

    # Get an array of floats for the historical data from user input
    st.subheader('Choose Historical Data')
    historical_data = st.text_area('Historical Data',
                                   value="9674146,  9679469, 10060861,  9771507,  9726983, 10152789, 9961637,  9888931, 10016144, 10025271, 10013123, 10144930, 10246870,  9838107,  9845904, 10220516,  9835059,  9572522, 10379305,  9680446, 10124238,  9464659,  9703857, 10045897,10738865, 10350408, 10219445, 10313337, 10310644, 10211187")

    # Process the historical data if it is provided
    if historical_data:
        historical_data = [float(x.strip())
                           for x in historical_data.split(',')]
        historical_data = np.array(historical_data)

    # Proceed if start time and historical data are provided
    if start_time is not None and historical_data is not None:
        # Generate timestamps for historical data and prediction period
        historical_time_stamps = [
            start_time - datetime.timedelta(days=x) for x in reversed(range(1, len(historical_data)+1))]
        pred_time_stamps = [start_time +
                            datetime.timedelta(days=x) for x in range(0, 30)]

        # Prepare the data
        features = prepare_data(historical_data, historical_time_stamps)

        # Plot the historical data
        fig, ax = plt.subplots()
        ax.plot(historical_time_stamps, historical_data,
                color='blue', label='Historical Data')

        # Generate predictions based on the chosen model
        if model_name == 'Rolling Average' or model_name == 'All':
            preds = get_rolling_average_predictions(
                historical_data, len(historical_data), 30)
            ra_monthly_preds = int(preds.sum())
            ax.plot(pred_time_stamps, preds, color='brown',
                    label='Rolling Average Predictions')

        if model_name == 'Linear Regression' or model_name == 'All':
            preds = get_linear_regression_predictions(
                features, len(historical_data), 30)
            lr_monthly_preds = int(preds.sum())
            ax.plot(pred_time_stamps, preds, color='orange',
                    label='Linear Regression Predictions')

        if model_name == 'Linear Regression + Neural Network' or model_name == 'All':
            preds = get_ensemble_predictions(features)
            ensemble_monthly_preds = int(preds.sum())
            ax.plot(pred_time_stamps, preds, color='green',
                    label='Ensemble Predictions')

        # Set the title and labels for the plot
        ax.set_xlabel('Receipt Count')
        plt.xticks(rotation=90)
        ax.set_ylabel('Data')
        ax.set_title('Receipt Count Forecast')
        ax.legend()
        st.pyplot(fig)

        # Show the monthly aggregate predictions
        st.subheader('Monthly Aggregate Predictions')
        if model_name == 'Rolling Average' or model_name == 'All':
            st.write(f'##### Rolling Average Predictions: {ra_monthly_preds}')
        if model_name == 'Linear Regression' or model_name == 'All':
            st.write(
                f'##### Linear Regression Predictions: {lr_monthly_preds}')
        if model_name == 'Linear Regression + Neural Network' or model_name == 'All':
            st.write(f'##### Ensemble Predictions: {ensemble_monthly_preds}')


if __name__ == '__main__':
    app()
