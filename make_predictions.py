"""Make predictions on new housing data using the trained model."""
import pickle
import numpy as np
import pandas as pd

# Load the trained model and scaler
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict_price(rm, lstat, ptratio, age, rad, tax, indus):
    """
    Predict house price for given features.
    
    Args:
        rm: Average number of rooms per dwelling
        lstat: % of lower status population
        ptratio: Pupil-teacher ratio
        age: % of buildings built before 1940
        rad: Accessibility to radial highways
        tax: Full-value property tax rate per $10K
        indus: % of non-retail business acres
    
    Returns:
        Predicted price in thousands of dollars
    """
    features = np.array([[rm, lstat, ptratio, age, rad, tax, indus]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    return prediction

if __name__ == "__main__":
    # Example predictions
    print("=== Housing Price Predictions ===")
    print()
    
    # Example 1: Average house
    pred1 = predict_price(rm=6.3, lstat=10, ptratio=16, age=65, rad=5, tax=300, indus=5)
    print(f"Average house (6.3 rooms, 10% lower status):")
    print(f"  Predicted price: ${pred1:.1f}K\n")
    
    # Example 2: Nicer house
    pred2 = predict_price(rm=8.0, lstat=5, ptratio=14, age=40, rad=3, tax=250, indus=3)
    print(f"Nicer house (8 rooms, 5% lower status):")
    print(f"  Predicted price: ${pred2:.1f}K\n")
    
    # Example 3: Cheaper house
    pred3 = predict_price(rm=5.5, lstat=20, ptratio=20, age=85, rad=8, tax=350, indus=8)
    print(f"Cheaper house (5.5 rooms, 20% lower status):")
    print(f"  Predicted price: ${pred3:.1f}K\n")
