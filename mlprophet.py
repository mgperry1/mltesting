import pandas as pd
from fbprophet import Prophet

# Load the data
df = pd.read_csv('capacity_data.csv')

# Preprocess the data
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df[['timestamp', 'space_gb']]  # Keep only relevant columns
df = df.rename(columns={'timestamp': 'ds', 'space_gb': 'y'})

# Split the data into train and test sets
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

# Create and train the Prophet model
model = Prophet()
model.fit(train_data)

# Make predictions
future = model.make_future_dataframe(periods=len(test_data), freq='D')
forecast = model.predict(future)

# Visualize the forecast
model.plot(forecast)

# Evaluate the model
predictions = forecast.tail(len(test_data))['yhat']
actual_values = test_data['y']
mae = mean_absolute_error(actual_values, predictions)
print("Mean Absolute Error:", mae)

import pickle

# Save the model
with open('prophet_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the model
with open('prophet_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Retrain the loaded model (optional)
loaded_model.fit(train_data)

