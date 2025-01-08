import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# Load your dataset
data = pd.read_csv('car_data.csv')  # Replace with your dataset file

# Data preprocessing
data['Driven Kilometers'] = data['Driven Kilometers'].str.replace(' KM', '').str.replace(',', '').astype(float)
data['Price (in ₹)'] = data['Price (in ₹)'].str.replace(',', '').astype(float)

# Encoding categorical variables
data = pd.get_dummies(data, columns=['Brand & Model', 'Varient', 'Fuel Type', 'Transmission', 'Owner', 'Location'], drop_first=True)

# Features and target
X = data.drop(['Price (in ₹)', 'Date of Posting Ad'], axis=1)
y = data['Price (in ₹)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Save the model
with open('car_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)
