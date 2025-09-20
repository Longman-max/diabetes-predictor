# quick_test.py - Quick test to see what's happening

import requests
import json

def test_predictions():
    """Test the Flask app with different profiles."""
    
    # Test profiles
    profiles = {
        "Non-Diabetic": {
            'Age': '25',
            'BMI': '22.0',
            'Blood Glucose': '85',
            'Blood Pressure': '110',
            'HbA1c': '5.0',
            'Insulin Level': '15',
            'Skin thickness': '15',
            'Pregnancies': '0',
            'Family history': '0',
            'Physical Activity': 'High',
            'Smoking status': 'Non-Smoker',
            'Alcohol Intake': '1',
            'Diet_Type': 'Vegetarian',
            'Cholesterol': '150',
            'Triglycerides': '100',
            'Waist ratio': '75'
        },
        "Fair Risk": {
            'Age': '35',
            'BMI': '26.0',
            'Blood Glucose': '105',
            'Blood Pressure': '125',
            'HbA1c': '5.8',
            'Insulin Level': '45',
            'Skin thickness': '25',
            'Pregnancies': '2',
            'Family history': '0',
            'Physical Activity': 'Medium',
            'Smoking status': 'Non-Smoker',
            'Alcohol Intake': '5',
            'Diet_Type': 'Non-Vegetarian',
            'Cholesterol': '200',
            'Triglycerides': '160',
            'Waist ratio': '90'
        },
        "Moderate Risk": {
            'Age': '50',
            'BMI': '30.0',
            'Blood Glucose': '125',
            'Blood Pressure': '140',
            'HbA1c': '6.2',
            'Insulin Level': '80',
            'Skin thickness': '35',
            'Pregnancies': '3',
            'Family history': '1',
            'Physical Activity': 'Low',
            'Smoking status': 'Smoker',
            'Alcohol Intake': '10',
            'Diet_Type': 'Non-Vegetarian',
            'Cholesterol': '240',
            'Triglycerides': '220',
            'Waist ratio': '105'
        },
        "High Risk": {
            'Age': '65',
            'BMI': '35.0',
            'Blood Glucose': '180',
            'Blood Pressure': '160',
            'HbA1c': '8.5',
            'Insulin Level': '150',
            'Skin thickness': '50',
            'Pregnancies': '5',
            'Family history': '1',
            'Physical Activity': 'Low',
            'Smoking status': 'Smoker',
            'Alcohol Intake': '15',
            'Diet_Type': 'Non-Vegetarian',
            'Cholesterol': '280',
            'Triglycerides': '350',
            'Waist ratio': '120'
        }
    }
    
    print("=== TESTING FLASK APP PREDICTIONS ===\n")
    
    for profile_name, data in profiles.items():
        print(f"Testing {profile_name}...")
        
        try:
            response = requests.post('http://localhost:5000/predict', data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"  Expected: {profile_name}")
                print(f"  Got: {result['message']}")
                print(f"  Class: {result['class_name']}")
                print(f"  Match: {'✓' if profile_name.lower().replace(' ', '-') in result['class_name'] else '✗'}")
            else:
                print(f"  Error: {response.status_code}")
                try:
                    error_info = response.json()
                    print(f"  Error details: {error_info}")
                except:
                    print(f"  Error text: {response.text}")
                    
        except Exception as e:
            print(f"  Request failed: {e}")
        
        print()

if __name__ == '__main__':
    test_predictions()