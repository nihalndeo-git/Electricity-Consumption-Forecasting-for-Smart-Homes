import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load preprocessed data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Initialize the Gradient Boosting model
model = GradientBoostingRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Gradient Boosting Model Evaluation:\nRMSE: {rmse}\nMAE: {mae}\nR²: {r2}")

# Save the improved model
joblib.dump(model, 'improved_electricity_forecasting_model.pkl')
print("Improved model saved as 'improved_electricity_forecasting_model.pkl'.")