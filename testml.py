import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the dataset
data = pd.read_csv('capacity_data.csv')

# Split the data into features (X) and target variable (y)
X = data[['space_gb']]
y = data['alerted']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Save the trained model
joblib.dump(model, 'trained_model.joblib')


# Load and reuse the trained model
loaded_model = joblib.load('trained_model.joblib')

# Make predictions using the loaded model
new_data = pd.DataFrame({'space_gb': [10, 20, 30]})  # New data for prediction
predictions = loaded_model.predict(new_data)

# Print the predictions
print(predictions)
