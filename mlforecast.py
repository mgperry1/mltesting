import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load your time series data
data = pd.read_csv('your_data.csv')
# Assuming your data has a column named 'date' and a column named 'space_gb'
# Ensure that the 'date' column is in a proper datetime format
data['date'] = pd.to_datetime(data['date'])

# Set the 'date' column as the index of the DataFrame
data.set_index('date', inplace=True)

# Split the data into training and testing sets
train_data = data.iloc[:-12]  # Use all data except the last 12 months for training
test_data = data.iloc[-12:]  # Use the last 12 months for testing

# Create and fit the SARIMA model
model = SARIMAX(train_data['space_gb'], order=(1, 0, 0), seasonal_order=(1, 1, 0, 12))
results = model.fit()

# Perform forecasting
forecast = results.get_forecast(steps=12)  # Forecast for the next 12 months

# Get the forecasted values and confidence intervals
forecast_values = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# Visualize the forecast and actual values
plt.plot(train_data.index, train_data['space_gb'], label='Training data')
plt.plot(test_data.index, test_data['space_gb'], label='Actual data')
plt.plot(forecast_values.index, forecast_values, label='Forecast')
plt.fill_between(confidence_intervals.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='gray', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Space (GB)')
plt.title('Time Series Forecast')
plt.legend()
plt.show()

import pickle

# Assuming you have already trained and obtained the 'results' SARIMA model

# Save the model to a file
model_filename = 'sarima_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(results, file)

# Load the saved model from the file
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# You can now use the loaded model for forecasting or other tasks
forecast = loaded_model.get_forecast(steps=12)
# ...

