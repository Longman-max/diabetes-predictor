# train_model_7fields.py - Train a model using only the first 7 fields

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
DATA_PATH = 'diabetes_dataset.csv'
MODEL_PATH = 'diabetes_rf_model_7fields.pkl'
SCALER_PATH = 'scaler_7fields.pkl'
COLUMNS_PATH = 'columns_7fields.pkl'

# Only the first 7 fields
FEATURE_COLS = [
    'Age', 'Gender', 'BMI', 'Family_History', 'Physical_Activity', 'Diet_Type',
    'Smoking_Status'
]
TARGET_COL = 'Diabetes_Status'

def main():
    data = pd.read_csv(DATA_PATH)
    # Check for missing features
    missing = [col for col in FEATURE_COLS if col not in data.columns]
    if missing:
        print(f"Missing feature columns: {missing}")
        return
    if TARGET_COL not in data.columns:
        print(f"Target column '{TARGET_COL}' not found. Using last column as target.")
        target = data.columns[-1]
    else:
        target = TARGET_COL
    X = data[FEATURE_COLS]
    y = data[target]
    # Fill missing values with 0
    X = X.fillna(0)
    # Save columns for prediction
    with open(COLUMNS_PATH, 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    # Save model and scaler
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"7-field model, scaler, and columns saved.")

if __name__ == '__main__':
    main()
