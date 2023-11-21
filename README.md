# Fetch's Receipt Count Forecasting App
This repo is for the machine learning exercise provided by Fetch Rewards.

## Overview

This project is a simple forecasting app built using [Streamlit](https://www.streamlit.io/) for predicting receipt counts based on historical data. It incorporates different forecasting models such as Rolling Average, Linear Regression, and an Ensemble model combining Linear Regression and a Neural Network (LSTM). The demo is hosted on [Streamlit](https://www.streamlit.io/) and can be accessed [here](https://fetch-machine-learning-exercise.streamlit.app).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Docker](#docker)
- [Models](#models)
- [Credits](#credits)
- [License](#license)

## Installation

To run the Streamlit app locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/connectwithprakash/fetch-machine-learning-exercise.git
   ```

2. Navigate to the project directory:

   ```bash
   cd fetch-machine-learning-exercise
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Train the forecasting models:
   
   Run the provided notebook `11-19-2023-forecasting-model.ipynb` to train the forecasting models and save them to disk.

   This will generate the following files in the `models/` directory:

   - `config.pkl`: Stores the configuration for the models.
   - `lr.pkl`: Stores the trained Linear Regression model.
   - `lstm.pkl`: Stores the trained Neural Network (LSTM) model.
   - `scaler.pkl`: Stores the scaler used for scaling the data.

5. Run the Streamlit app:

   ```bash
   streamlit run streamlit_app.py
   ```

   Access the app in your web browser at [http://localhost:8501](http://localhost:8501).

## Usage

1. Choose a forecasting model from the sidebar.
2. Select the start time for the forecast.
3. Input historical data in the provided text area.
4. Click on "Choose Historical Data" to generate predictions.
5. The app will display a plot of historical data and predictions, along with monthly aggregate predictions based on the chosen model.

## Folder Structure

The project folder is structured as follows:

- `data/`: Contains the historical data used for training the models.
- `models/`: Stores trained models and configuration files.
- `notebooks/`: Contains the notebook used for training the models.
- `src/`: Contains utility scripts and modules.
   - `data/`: Contains scripts for loading and preprocessing data.
   - `models/`: Contains scripts for training and evaluating models.
   - `utils/`: Contains utility scripts and modules.
- `requirements.txt`: Lists project dependencies.

## Docker

The project includes a Dockerfile for containerization. To build and run the Docker container, follow these steps:

1. Build the Docker image:

   ```bash
   docker build -t fetch-machine-learning-exercise .
   ```

2. Run the Docker container:

   ```bash
    docker run -p 8080:8080 fetch-machine-learning-exercise
    ```

3. Access the app in your web browser at [http://localhost:8080](http://localhost:8080).

## Models

- **Linear Regression Model:** Trained for predicting receipt counts.
- **Rolling Average Model:** A simple forecasting model using rolling averages.
- **Ensemble Model:** Combines predictions from Linear Regression and a Neural Network (LSTM).

## Authors

This project is maintained by Prakash Chaudhary. You can find me on [LinkedIn](https://www.linkedin.com/in/connectwithprakash/).

---
