import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv('CarPrice.csv')  # Replace with your dataset file

# Clean the 'Driven Kilometers' column
data['Driven Kilometers'] = data['Driven Kilometers'] \
    .str.replace(r'\s*[kK][mM]', '', regex=True) \
    .str.replace(',', '', regex=False) \
    .astype(float)

# Perform basic preprocessing (handle missing data if any)
data = data.dropna()

# Feature selection
X = data[['Fuel Type', 'Driven Kilometers', 'Transmission', 'Owner']]
y = data['Price (in ₹)']

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Save the feature names for consistency
feature_names = X.columns.tolist()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save the trained model and feature names
with open('car_price_model.pkl', 'wb') as file:
    joblib.dump((model, feature_names), file)

print("Model training complete. Saved as 'car_price_model.pkl'.")
