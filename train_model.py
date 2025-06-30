# train_model.py - Rewritten from scratch for diabetes prediction model

# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
DATA_PATH = 'diabetes_dataset.csv'
MODEL_PATH = 'diabetes_rf_model.pkl'
SCALER_PATH = 'scaler.pkl'

# Define feature columns (update these to match your dataset exactly)
FEATURE_COLS = [
    'Age', 'Gender', 'BMI', 'Family_History', 'Physical_Activity', 'Diet_Type',
    'Smoking_Status', 'Alcohol_Intake', 'Stress_Level', 'Hypertension', 'Cholesterol_Level',
    'Fasting_Blood_Sugar', 'Postprandial_Blood_Sugar', 'HBA1C', 'Heart_Rate', 'Waist_Hip_Ratio',
    'Urban_Rural', 'Health_Insurance', 'Regular_Checkups', 'Medication_For_Chronic_Conditions',
    'Pregnancies', 'Polycystic_Ovary_Syndrome', 'Glucose_Tolerance_Test_Result', 'Vitamin_D_Level',
    'C_Protein_Level', 'Thyroid_Condition'
]
TARGET_COL = 'Diabetes_Status'

def main():
    # Load data
    data = pd.read_csv(DATA_PATH)
    print('Columns in dataset:', list(data.columns))

    # Check for missing features
    missing = [col for col in FEATURE_COLS if col not in data.columns]
    if missing:
        print(f"Missing feature columns: {missing}")
        print("Please update FEATURE_COLS to match your dataset.")
        return
    if TARGET_COL not in data.columns:
        print(f"Target column '{TARGET_COL}' not found. Using last column as target.")
        target = data.columns[-1]
    else:
        target = TARGET_COL

    # Prepare features and target
    X = data[FEATURE_COLS]
    y = data[target]
    print(f"Feature shape: {X.shape}, Target shape: {y.shape}")

    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    print(f"Features after encoding: {X.shape}")

    # Save the columns after encoding for use in prediction
    with open('columns.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features scaled.")

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    print("Model trained.")

    # Save model and scaler
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Model saved to {MODEL_PATH}, scaler saved to {SCALER_PATH}.")

if __name__ == '__main__':
    main()