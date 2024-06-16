# Instructions to run Train Model and Run Predictions

Switch to `assignment` folder.

## Google Colab Setup
* Import `forecast.ipynb`
* Upload `synthetic_retail_data_for_demand_prediction.csv` file when asked for
* Note: Code from `kellys.py` is copied to this `forecast.ipynb` notebook to make things easy to run in Colab.

## Localhost Setup
### Install all the required packages
In a terminal run
```
pip install -r requirements.txt
```

### Run Unit Tests
In a terminal run, switch to the `assignment` folder and run
```
python test_unit.py
```

## Synthetic Data Setup
* Execute `AI_ML_Project_Synthetic_data_generation.ipynb` file to generate synthetic data for the purpose of this assignment
* A copy of this has already been checked into `synthetic_retail_data_for_demand_prediction.csv` for ready usage.

## Run Predictions
* Code to run predictions have been captured in `forecast.ipynb`
* Input the required values and get the predictions
```py
    # Define future dates for which we want to forecast
    future_dates = pd.date_range(start='2021-06-10', end='2021-06-25', freq='D')

    # Create a DataFrame for the future dates with the same structure as the exogenous data
    future_exog = pd.DataFrame({
        'Country': ['USA'] * len(future_dates),  # Replace with desired countries
        'Store ID': [4] * len(future_dates),    # Replace with desired store IDs
        'Product ID': [15] * len(future_dates), # Replace with desired product IDs
        'Product Category': ['Home & Kitchen'] * len(future_dates)  # Replace with desired product categories
    }, index=future_dates)

    forecast(future_dates, future_exog)
```