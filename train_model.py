"""Train a linear regression model to predict house prices."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

# Load the dataset
df = pd.read_csv('data/housing.csv')

print(f"Dataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}\n")

# Prepare features and target
X = df[['RM', 'LSTAT', 'PTRATIO', 'AGE', 'RAD', 'TAX', 'INDUS']]
y = df['MEDV']

# Split into train/test sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}\n")

# Standardize features (important for linear regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Evaluate
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("=== Training Results ===")
print(f"Train RMSE: ${train_rmse:.2f}K")
print(f"Train MAE:  ${train_mae:.2f}K")
print(f"Train R²:   {train_r2:.4f}\n")

print("=== Test Results ===")
print(f"Test RMSE:  ${test_rmse:.2f}K")
print(f"Test MAE:   ${test_mae:.2f}K")
print(f"Test R²:    {test_r2:.4f}\n")

# Feature importance (coefficients)
feature_names = X.columns
coefficients = model.coef_

print("=== Feature Coefficients ===")
for name, coef in sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True):
    print(f"{name:10s}: {coef:7.3f}")

print(f"\nIntercept: {model.intercept_:.3f}")

# Save the trained model and scaler
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✓ Model saved to models/model.pkl")
print("✓ Scaler saved to models/scaler.pkl")
