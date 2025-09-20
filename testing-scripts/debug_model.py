# debug_model.py - Test the trained model with different profiles
import pandas as pd
import numpy as np
import pickle
import os

def load_model():
    """Load the trained model components."""
    try:
        models_dir = 'models'
        with open(os.path.join(models_dir, 'diabetes_rf_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(models_dir, 'columns.pkl'), 'rb') as f:
            columns = pickle.load(f)
        return model, scaler, columns
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def classify_risk(prob):
    """Risk classification function."""
    if prob >= 0.70:
        return "High Risk"
    elif prob >= 0.45:
        return "Moderate Risk"
    elif prob >= 0.20:
        return "Fair Risk"
    else:
        return "Non-Diabetic"

def create_test_profiles():
    """Create test profiles for each risk category."""
    profiles = {
        'Non-Diabetic': {
            'Age': 25, 'BMI': 22.0, 'Blood Glucose': 85, 'Blood Pressure': 110,
            'HbA1c': 5.0, 'Insulin Level': 15, 'Skin thickness': 15, 'Pregnancies': 0,
            'Family history': 0, 'Physical Activity': 'High', 'Smoking status': 'Non-Smoker',
            'Alcohol Intake': 1, 'Diet_Type': 'Vegetarian', 'Cholesterol': 150,
            'Triglycerides': 100, 'Waist ratio': 75
        },
        'Fair Risk': {
            'Age': 35, 'BMI': 26.0, 'Blood Glucose': 105, 'Blood Pressure': 125,
            'HbA1c': 5.8, 'Insulin Level': 45, 'Skin thickness': 25, 'Pregnancies': 2,
            'Family history': 0, 'Physical Activity': 'Medium', 'Smoking status': 'Non-Smoker',
            'Alcohol Intake': 5, 'Diet_Type': 'Non-Vegetarian', 'Cholesterol': 200,
            'Triglycerides': 160, 'Waist ratio': 90
        },
        'Moderate Risk': {
            'Age': 50, 'BMI': 30.0, 'Blood Glucose': 125, 'Blood Pressure': 140,
            'HbA1c': 6.2, 'Insulin Level': 80, 'Skin thickness': 35, 'Pregnancies': 3,
            'Family history': 1, 'Physical Activity': 'Low', 'Smoking status': 'Smoker',
            'Alcohol Intake': 10, 'Diet_Type': 'Non-Vegetarian', 'Cholesterol': 240,
            'Triglycerides': 220, 'Waist ratio': 105
        },
        'High Risk': {
            'Age': 65, 'BMI': 35.0, 'Blood Glucose': 180, 'Blood Pressure': 160,
            'HbA1c': 8.5, 'Insulin Level': 150, 'Skin thickness': 50, 'Pregnancies': 5,
            'Family history': 1, 'Physical Activity': 'Low', 'Smoking status': 'Smoker',
            'Alcohol Intake': 15, 'Diet_Type': 'Non-Vegetarian', 'Cholesterol': 280,
            'Triglycerides': 350, 'Waist ratio': 120
        }
    }
    return profiles

def test_model():
    """Test the model with different profiles."""
    print("=== DIABETES MODEL DEBUG TEST ===\n")
    
    # Load model
    model, scaler, columns = load_model()
    if model is None:
        print("Failed to load model. Please run train_model.py first.")
        return
    
    print(f"Model loaded successfully!")
    print(f"Expected columns: {len(columns)}")
    print(f"Columns: {columns}\n")
    
    # Create test profiles
    profiles = create_test_profiles()
    
    for profile_name, profile_data in profiles.items():
        print(f"=== TESTING {profile_name.upper()} PROFILE ===")
        print("Input data:")
        for key, value in profile_data.items():
            print(f"  {key}: {value}")
        
        # Create DataFrame
        X_input = pd.DataFrame([profile_data])
        
        # One-hot encode
        categorical_cols = ['Physical Activity', 'Smoking status', 'Diet_Type']
        X_encoded = pd.get_dummies(X_input, columns=categorical_cols, drop_first=True)
        
        print(f"\nAfter encoding: {X_encoded.shape}")
        print(f"Encoded columns: {list(X_encoded.columns)}")
        
        # Align columns
        for col in columns:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        
        # Remove extra columns
        extra_cols = [col for col in X_encoded.columns if col not in columns]
        if extra_cols:
            X_encoded = X_encoded.drop(columns=extra_cols)
        
        # Reorder columns
        X_final = X_encoded[columns]
        
        print(f"Final shape: {X_final.shape}")
        print("Final values:")
        for i, (col, val) in enumerate(X_final.iloc[0].items()):
            print(f"  {col}: {val}")
        
        # Scale and predict
        X_scaled = scaler.transform(X_final)
        probabilities = model.predict_proba(X_scaled)[0]
        prob_diabetes = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        
        # Get risk classification
        risk_level = classify_risk(prob_diabetes)
        
        print(f"\nRESULTS:")
        print(f"  Raw probabilities: {probabilities}")
        print(f"  Diabetes probability: {prob_diabetes:.4f} ({prob_diabetes*100:.1f}%)")
        print(f"  Risk classification: {risk_level}")
        print(f"  Expected: {profile_name}")
        print(f"  Match: {'✓' if risk_level == profile_name else '✗'}")
        print("\n" + "="*50 + "\n")

if __name__ == '__main__':
    test_model()