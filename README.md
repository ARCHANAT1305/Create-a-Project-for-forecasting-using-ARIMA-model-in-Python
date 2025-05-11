# EX.NO.09 A project on Time series analysis on weather forecasting using ARIMA model
### Date:06/05/25
## AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.

## ALGORITHM:
1. Explore the dataset of weather
2. Check for stationarity of time series time series plot ACF plot and PACF plot ADF test Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
## PROGRAM:
### NAME: ARCHANA T
### REGISTER NUMBER : 212223240013
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("powerconsumption.csv")

# Convert 'Datetime' column to datetime format
data['Datetime'] = pd.to_datetime(data['Datetime'])

# Set 'Datetime' as index
data.set_index('Datetime', inplace=True)

# Combine all zones into a single total power column
data['TotalPower'] = data['PowerConsumption_Zone1'] + data['PowerConsumption_Zone2'] + data['PowerConsumption_Zone3']

# Optionally, resample to daily average or total
# data = data.resample('D').sum()  # Use sum if values are in kWh or total usage
data = data.resample('D').mean()  # Use mean if values represent average power

# Drop missing values if any
data.dropna(inplace=True)

# ARIMA model definition
def arima_model(data, target_variable, order):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=len(test_data))

    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.grid()
    plt.show()

    print("Root Mean Squared Error (RMSE):", rmse)

# Run the ARIMA model
arima_model(data, 'TotalPower', order=(5,1,0))

```
## OUTPUT:
![image](https://github.com/user-attachments/assets/7745c7ea-504d-4e46-9dcc-d8e346b8b485)

## RESULT:
Thus the program run successfully based on the ARIMA model using python.
