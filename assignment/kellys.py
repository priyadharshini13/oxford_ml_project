import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import itertools
# from google.colab import files
import io
from sklearn.model_selection import TimeSeriesSplit
import unittest
import seaborn as sns
import time

def load_data():
    print("Loading data")
    file_path = 'synthetic_retail_data_for_demand_prediction.csv'
    return pd.read_csv(file_path)

def preprocess(data):
    # @title Step 4: Split Data into Training and Testing Sets
    data['Date of Sale'] = pd.to_datetime(data['Date of Sale'])

    # Set the date column as the index
    data.set_index('Date of Sale', inplace=True)

    # Select the 'Units Sold' column for time series analysis
    ts_data = data[['Units Sold']]

    # Add additional parameters (e.g., Country, Store ID, Product ID, Product Category)
    time_series_data = pd.get_dummies(data[['Country', 'Store ID', 'Product ID', 'Product Category']], drop_first=True)

    # Convert boolean columns to integers
    bool_columns = time_series_data.select_dtypes(include=['bool']).columns
    time_series_data[bool_columns] = time_series_data[bool_columns].astype(int)

    # Ensure time_series_data variables are numeric
    time_series_data = time_series_data.apply(pd.to_numeric, errors='coerce')

    # Check the data types of the time_series_data variables
    print("time_series_data variables data types after conversion:\n", time_series_data.dtypes)

    # Split the data into training and testing sets
    split_point = int(len(ts_data) * 0.8)
    train_data, test_data = ts_data[:split_point], ts_data[split_point:]
    train_ts, test_ts = time_series_data[:split_point], time_series_data[split_point:]

    # Ensure the data is numeric
    train_data = train_data.astype(float)
    test_data = test_data.astype(float)

    print(f"Training data: {len(train_data)} records")
    print(f"Testing data: {len(test_data)} records")
    print(f"Training time_series_data data types:\n{train_ts.dtypes}")

    # Ensure no non-numeric columns remain
    non_numeric_columns = train_ts.select_dtypes(exclude=['number']).columns
    if len(non_numeric_columns) > 0:
        raise ValueError(f"Non-numeric columns in training exogenous data: {non_numeric_columns}")
    else:
        print("All training time_series_data data columns are numeric.")

    print(f"Training data: {len(train_data)} records")
    print(f"Testing data: {len(test_data)} records")
    
    return train_data, test_data, train_ts, test_ts


def train(train_data, train_ts):
    # Define the SARIMAX model with exogenous variables
    sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), exog=train_ts, enforce_stationarity=False, enforce_invertibility=False)

    # Fit the model
    sarima_results = sarima_model.fit(disp=False)

    # Print model summary
    print(sarima_results.summary())

    return sarima_results


