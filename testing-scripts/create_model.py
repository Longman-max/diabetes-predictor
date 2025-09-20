# create_mock_model.py - Create a mock model that gives predictable results
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

class MockDiabetesModel:
    """A mock model that gives predictable results based on input values."""
    
    def __init__(self):
        self.feature_weights = {
            'Age': 0.02,
            'BMI': 0.03,
            'Blood Glucose': 0.008,
            'Blood Pressure': 0.005,
            'HbA1c': 0.12,
            'Insulin Level': 0.002,
            'Skin thickness': 0.01,
            'Pregnancies': 0.05,
            'Family history': 0.15,
            'Alcohol Intake': 0.02,
            'Cholesterol': 0.001,
            'Triglycerides': 0.001,
            'Waist ratio': 0.003,
            # Categorical features (will be added after one-hot encoding)
        }
        
        # Add categorical feature weights
        self.categorical_weights = {
            'Physical Activity_Medium': -0.05,
            'Physical Activity_High': -0.1,
            'Smoking status_Smoker': 0.15,
            'Diet_Type_Vegetarian': -0.05,
            'Diet_Type_Vegan': -0.08
        }
        
        self.feature_weights.update(self.categorical_weights)
    
    def predict_proba(self, X):
        """Predict probabilities based on weighted features."""
        probabilities = []
        
        for i in range(X.shape[0]):
            # Calculate risk score
            risk_score = 0
            
            # Base risk
            base_risk = 0.3
            
            # Add weighted contributions
            for j, col_name in enumerate(self.columns):
                if col_name in self.feature_weights:
                    weight = self.feature_weights[col_name]
                    value = X[i, j]
                    risk_score += weight * value
            
            # Convert to probability (sigmoid-like function)
            prob_diabetes = base_risk + (risk_score * 0.5)
            prob_diabetes = max(0.01, min(0.99, prob_diabetes))  # Clamp between 0.01 and 0.99
            
            prob_no_diabetes = 1 - prob_diabetes
            probabilities.append([prob_no_diabetes, prob_diabetes])
        
        return np.array(probabilities)
    
    def predict(self, X):
        """Binary prediction based on probability threshold."""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)
    
    def set_columns(self, columns):
        """Set the expected column names."""
        self.columns = columns

def create_mock_model():
    """Create and save a mock model with predictable results."""
    print("Creating mock diabetes prediction model...")
    
    # Create models directory
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Define expected columns (same as your training script)
    feature_cols = [
        'Age', 'BMI', 'Blood Glucose', 'Blood Pressure', 'HbA1c',
        'Insulin Level', 'Skin thickness', 'Pregnancies', 'Family history',
        'Physical Activity', 'Smoking status', 'Alcohol Intake', 'Diet_Type',
        'Cholesterol', 'Triglycerides', 'Waist ratio'
    ]
    
    # Create sample data for encoding
    sample_data = pd.DataFrame({
        'Age': [30, 50, 70],
        'BMI': [25, 30, 35],
        'Blood Glucose': [90, 120, 150],
        'Blood Pressure': [120, 140, 160],
        'HbA1c': [5.5, 6.0, 7.0],
        'Insulin Level': [50, 100, 150],
        'Skin thickness': [25, 35, 45],
        'Pregnancies': [1, 3, 5],
        'Family history': [0, 1, 1],
        'Physical Activity': ['High', 'Medium', 'Low'],
        'Smoking status': ['Non-Smoker', 'Non-Smoker', 'Smoker'],
        'Alcohol Intake': [2, 8, 15],
        'Diet_Type': ['Vegetarian', 'Non-Vegetarian', 'Non-Vegetarian'],
        'Cholesterol': [180, 220, 280],
        'Triglycerides': [120, 180, 300],
        'Waist ratio': [85, 100, 120]
    })
    
    # One-hot encode to get column names
    categorical_cols = ['Physical Activity', 'Smoking status', 'Diet_Type']
    encoded_data = pd.get_dummies(sample_data, columns=categorical_cols, drop_first=True)
    columns = encoded_data.columns.tolist()
    
    print(f"Expected columns ({len(columns)}): {columns}")
    
    # Create mock model
    model = MockDiabetesModel()
    model.set_columns(columns)
    
    # Create a simple scaler (identity scaler for simplicity)
    scaler = StandardScaler()
    scaler.fit(encoded_data)
    
    # Save model components
    model_path = os.path.join(models_dir, 'diabetes_rf_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    columns_path = os.path.join(models_dir, 'columns.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(columns_path, 'wb') as f:
        pickle.dump(columns, f)
    
    print(f"Mock model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Columns saved to: {columns_path}")
    
    # Test the mock model
    print("\n=== TESTING MOCK MODEL ===")
    test_profiles = {
        'Non-Diabetic': [25, 22.0, 85, 110, 5.0, 15, 15, 0, 0, 1, 150, 100, 75],
        'High Risk': [65, 35.0, 180, 160, 8.5, 150, 50, 5, 1, 15, 280, 350, 120]
    }
    
    for profile_name, values in test_profiles.items():
        # Create test input matching the expected format
        test_input = pd.DataFrame([{
            'Age': values[0], 'BMI': values[1], 'Blood Glucose': values[2], 
            'Blood Pressure': values[3], 'HbA1c': values[4], 'Insulin Level': values[5],
            'Skin thickness': values[6], 'Pregnancies': values[7], 'Family history': values[8],
            'Physical Activity': 'High' if profile_name == 'Non-Diabetic' else 'Low',
            'Smoking status': 'Non-Smoker' if profile_name == 'Non-Diabetic' else 'Smoker',
            'Alcohol Intake': values[9], 'Diet_Type': 'Vegetarian' if profile_name == 'Non-Diabetic' else 'Non-Vegetarian',
            'Cholesterol': values[10], 'Triglycerides': values[11], 'Waist ratio': values[12]
        }])
        
        # Encode
        test_encoded = pd.get_dummies(test_input, columns=categorical_cols, drop_first=True)
        
        # Align columns
        for col in columns:
            if col not in test_encoded.columns:
                test_encoded[col] = 0
        test_encoded = test_encoded[columns]
        
        # Scale and predict
        test_scaled = scaler.transform(test_encoded)
        prob = model.predict_proba(test_scaled)[0][1]
        
        print(f"{profile_name}: {prob:.3f} ({prob*100:.1f}%)")
    
    print("\nMock model creation completed!")

if __name__ == '__main__':
    create_mock_model()