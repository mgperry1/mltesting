import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pickle
import matplotlib.pyplot as plt

# Assuming you have the 'capacity_data.csv' file containing your data
data = pd.read_csv('capacity_data.csv')

# Preprocess the data if necessary
# ...

# Split the data into train and test sets
train_data = data[:-30]  # Use all but the last 30 data points for training
test_data = data[-30:]  # Use the last 30 data points for testing

# Fit the ARIMA model
model = ARIMA(train_data['space_gb'], order=(1, 1, 1))
model_fit = model.fit()

# Save the model
with open('arima_model.pkl', 'wb') as f:
    pickle.dump(model_fit, f)

# Load the model
with open('arima_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Forecast using the loaded model
forecast = loaded_model.get_forecast(steps=30)
forecasted_values = forecast.predicted_mean

# Print the forecasted values
print(forecasted_values)


# Generate predictions for the training data
train_predictions = model_fit.predict(start=1, end=len(train_data), typ='levels')

# Plot the actual values and predicted values
plt.plot(train_data.index, train_data['space_gb'], label='Actual')
plt.plot(train_data.index, train_predictions, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Space (GB)')
plt.title('Actual vs Predicted (Training Data)')
plt.legend()
plt.show()
