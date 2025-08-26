# train_model.py - Diabetes Prediction with Risk Stratification

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# File paths
DATA_PATH = 'data/diabetes_dataset.csv'
MODEL_PATH = 'diabetes_rf_model.pkl'
SCALER_PATH = 'scaler.pkl'
COLUMNS_PATH = 'columns.pkl'

def classify_risk(prob):
    """Convert predicted probability to risk level with more granular thresholds."""
    if prob < 0.3:  # Very low probability
        return 'Low'
    elif prob < 0.5:  # Moderate probability
        return 'Moderate'
    else:  # High probability
        return 'High'

def main():
    print("Starting model training process...")

    # Load and preprocess data
    try:
        data = pd.read_csv('data/diabetes_dataset.csv')

        # Create derived features from the dataset that match our form fields
        data['Blood Glucose'] = data['Fasting_Blood_Sugar']
        data['Blood Pressure'] = data['Heart_Rate']  # This might need adjustment
        data['HbA1c'] = data['HBA1C']
        data['Insulin Level'] = data['Glucose_Tolerance_Test_Result']  # Approximation
        data['Skin thickness'] = data['Waist_Hip_Ratio'] * 50  # Rough approximation
        data['Family history'] = data['Family_History']
        data['Physical Activity'] = data['Physical_Activity']
        data['Smoking status'] = data['Smoking_Status']
        data['Alcohol Intake'] = data['Alcohol_Intake']
        # 'Diet_Type' is used directly
        data['Cholesterol'] = data['Cholesterol_Level']
        data['Waist ratio'] = data['Waist_Hip_Ratio'] * 100  # Convert to cm
        data['Triglycerides'] = data['Cholesterol_Level'] * 0.8  # Rough approximation

        # List of features that match our form
        feature_cols = [
            'Age', 'BMI', 'Blood Glucose', 'Blood Pressure', 'HbA1c',
            'Insulin Level', 'Skin thickness', 'Pregnancies', 'Family history',
            'Physical Activity', 'Smoking status', 'Alcohol Intake', 'Diet_Type',
            'Cholesterol', 'Triglycerides', 'Waist ratio'
        ]

        print('Prepared dataset with columns:', list(data.columns))

        # Verify all required features are present
        missing = [col for col in feature_cols if col not in data.columns]
        if missing:
            print(f"Missing feature columns: {missing}")
            return

        # Prepare features and target
        print("Preparing features and target...")
        X = data[feature_cols]
        y = data['Diabetes_Status']
        print(f"Feature shape: {X.shape}, Target shape: {y.shape}")

        # Encode categorical features
        print("Encoding categorical features...")
        X_encoded = pd.get_dummies(X, drop_first=True)
        print(f"Features after encoding: {X_encoded.shape}")

        # Save encoded column names
        print("Saving column information...")
        with open(COLUMNS_PATH, 'wb') as f:
            pickle.dump(X_encoded.columns.tolist(), f)

        # Scale features
        print("Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        # Train model with balanced class weights
        print("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_scaled, y)
        print("Model trained successfully.")

        # Save model and scaler
        print("Saving model and scaler...")
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Saved model to {MODEL_PATH} and scaler to {SCALER_PATH}")

        # Generate and evaluate predictions
        print("Evaluating model performance...")
        probs = model.predict_proba(X_scaled)[:, 1]
        predictions = model.predict(X_scaled)
        risk_levels = np.array([classify_risk(p) for p in probs])

        # Calculate basic metrics
        accuracy = (predictions == y).mean()
        print(f"\nModel Accuracy: {accuracy:.2%}")

        # Print risk level distribution
        risk_dist = pd.Series(risk_levels).value_counts()
        print("\nRisk Level Distribution:")
        print(risk_dist)

        # Save predictions to CSV
        print("\nSaving predictions...")
        output_df = data.copy()
        output_df['Predicted_Probability'] = probs
        output_df['Predicted_Status'] = predictions
        output_df['Diabetes_Risk_Level'] = risk_levels
        output_df.to_csv('diabetes_predictions_with_risk.csv', index=False)
        print("Predictions saved to 'diabetes_predictions_with_risk.csv'")

    except Exception as e:
        print(f"Error during model training: {e}")
        raise
    output_df['Diabetes_Risk_Level'] = risk_levels
    output_df.to_csv('diabetes_predictions_with_risk.csv', index=False)
    print("Predictions saved to 'diabetes_predictions_with_risk.csv'")
    print("Training process completed successfully!")

if __name__ == '__main__':
    main()