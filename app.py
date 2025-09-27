from flask import Flask, render_template, request, jsonify, send_from_directory
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

# Ensure the instance folder exists
os.makedirs(app.instance_path, exist_ok=True)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'your_secret_key_here'

# Load model, scaler, and columns
try:
    print("Loading model files...")
    models_dir = 'models'
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found. Please run train_model.py first.")
        raise FileNotFoundError(f"Models directory '{models_dir}' not found")
    
    model_path = os.path.join(models_dir, 'diabetes_rf_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    columns_path = os.path.join(models_dir, 'columns.pkl')
    
    # Check if all required files exist
    required_files = [model_path, scaler_path, columns_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Required model file '{file_path}' not found. Please run train_model.py first.")
            raise FileNotFoundError(f"Model file '{file_path}' not found")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(columns_path, 'rb') as f:
        columns = pickle.load(f)
    print("Model files loaded successfully.")
    print(f"Expected columns: {columns}")
except Exception as e:
    print(f"Error loading model files: {e}")
    raise

def get_risk_result(probabilities):
    """
    Get risk classification result based on predicted class probabilities.
    probabilities: array of probabilities for each class [Fair, Moderate, High]
    """
    risk_levels = ['Fair', 'Moderate', 'High']
    highest_prob_idx = np.argmax(probabilities)
    risk_level = risk_levels[highest_prob_idx]
    prob_percent = probabilities[highest_prob_idx] * 100
    
    print(f"=== RISK CLASSIFICATION ===")
    print(f"Probabilities: Fair={probabilities[0]:.3f}, Moderate={probabilities[1]:.3f}, High={probabilities[2]:.3f}")
    print(f"Highest probability: {prob_percent:.1f}% ({risk_level} Risk)")
    
    if risk_level == 'High':
        result = {
            'message': "High Risk",
            # 'message': f"{prob_percent:.1f}% High Risk",
            'advice': 'Your results indicate a high risk. Please consult a healthcare professional immediately for diagnosis and guidance.',
            'class_name': 'high-risk'
        }
    elif risk_level == 'Moderate':
        result = {
            'message': "Moderate Risk",
            # 'message': f"{prob_percent:.1f}% Moderate Risk",
            'advice': 'You are at moderate risk. It is advisable to monitor your health and consult a doctor for preventative measures.',
            'class_name': 'moderate-risk'
        }
    else:  # Fair Risk
        result = {
            'message': "Fair Risk",
            # 'message': f"{prob_percent:.1f}% Fair Risk",
            'advice': 'Your risk is fair, but there is room for improvement. Focus on a balanced diet and regular physical activity.',
            'class_name': 'fair-risk'
        }
    
    print(f"Classification result: {result}")
    return result


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("\n" + "="*50)
        print("NEW PREDICTION REQUEST")
        print("="*50)
        
        # Collect input from form
        input_dict = {}
        expected_fields = [
            'Age', 'BMI', 'Blood Glucose', 'Blood Pressure', 'HbA1c',
            'Insulin Level', 'Skin thickness', 'Pregnancies', 'Family history',
            'Physical Activity', 'Smoking status', 'Alcohol Intake', 'Diet_Type',
            'Cholesterol', 'Triglycerides', 'Waist ratio'
        ]
        
        print("RAW FORM DATA:")
        for field in expected_fields:
            value = request.form.get(field, '')
            input_dict[field] = value
            print(f"  {field}: '{value}'")
        
        # Convert and validate numeric fields
        numeric_fields = [
            'Age', 'BMI', 'Blood Glucose', 'Blood Pressure', 'HbA1c',
            'Insulin Level', 'Skin thickness', 'Pregnancies', 'Alcohol Intake',
            'Cholesterol', 'Triglycerides', 'Waist ratio'
        ]
        
        print("\nCONVERTED NUMERIC VALUES:")
        for field in numeric_fields:
            try:
                if input_dict[field] in ['', None]:
                    input_dict[field] = 0.0
                else:
                    input_dict[field] = float(input_dict[field])
                print(f"  {field}: {input_dict[field]}")
            except (ValueError, TypeError) as e:
                print(f"  ERROR converting {field}: {e}")
                input_dict[field] = 0.0
        
        # Handle Family history - convert to numeric
        if input_dict['Family history'] in ['', None]:
            input_dict['Family history'] = 0
        else:
            input_dict['Family history'] = int(float(input_dict['Family history']))
        
        print(f"\nFamily history converted: {input_dict['Family history']}")
        
        # Validate categorical fields have values
        categorical_fields = ['Physical Activity', 'Smoking status', 'Diet_Type']
        print("\nCATEGORICAL VALUES:")
        for field in categorical_fields:
            print(f"  {field}: '{input_dict[field]}'")
            if input_dict[field] in ['', None]:
                return jsonify({
                    'error': f'Please select a value for {field}'
                }), 400
        
        # Create DataFrame
        X_input = pd.DataFrame([input_dict])
        print(f"\nDATAFRAME CREATED:")
        print(f"Shape: {X_input.shape}")
        print(f"Columns: {X_input.columns.tolist()}")
        print(f"Values:\n{X_input.iloc[0].to_dict()}")
        
        # One-hot encode categorical features (excluding Family history which is already numeric)
        categorical_to_encode = ['Physical Activity', 'Smoking status', 'Diet_Type']
        X_encoded = pd.get_dummies(X_input, columns=categorical_to_encode, drop_first=True)
        
        print(f"\nAFTER ONE-HOT ENCODING:")
        print(f"Shape: {X_encoded.shape}")
        print(f"Columns: {X_encoded.columns.tolist()}")
        
        # Align columns with training data
        print(f"\nEXPECTED COLUMNS ({len(columns)}): {columns}")
        print(f"CURRENT COLUMNS ({len(X_encoded.columns)}): {X_encoded.columns.tolist()}")
        
        # Add missing columns
        missing_cols = []
        for col in columns:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
                missing_cols.append(col)
        
        if missing_cols:
            print(f"ADDED MISSING COLUMNS: {missing_cols}")
        
        # Remove extra columns not in training data
        extra_cols = [col for col in X_encoded.columns if col not in columns]
        if extra_cols:
            print(f"REMOVING EXTRA COLUMNS: {extra_cols}")
            X_encoded = X_encoded.drop(columns=extra_cols)
        
        # Reorder columns to match training data exactly
        X_final = X_encoded[columns]
        
        print(f"\nFINAL DATAFRAME FOR PREDICTION:")
        print(f"Shape: {X_final.shape}")
        print(f"Columns match training data: {list(X_final.columns) == list(columns)}")
        print("Final values:")
        for i, (col, val) in enumerate(X_final.iloc[0].items()):
            print(f"  {i:2d}. {col}: {val}")
        
        # Scale features
        X_scaled = scaler.transform(X_final)
        print(f"\nSCALED FEATURES:")
        print(f"Shape: {X_scaled.shape}")
        print(f"Min value: {X_scaled.min():.3f}")
        print(f"Max value: {X_scaled.max():.3f}")
        print(f"Mean: {X_scaled.mean():.3f}")
        
        # Make prediction
        print("\nMAKING PREDICTION...")
        risk_probs = model.predict_proba(X_scaled)[0]
        print(f"Risk probabilities: {risk_probs}")
        
        # Get risk classification result
        risk_data = get_risk_result(risk_probs)
        
        print(f"\nFINAL RESULT:")
        print(f"Message: {risk_data['message']}")
        print(f"Class: {risk_data['class_name']}")
        print(f"Advice: {risk_data['advice'][:50]}...")
        
        return jsonify(risk_data)

    except Exception as e:
        print(f"\n=== PREDICTION ERROR ===")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_result = {'error': f'An unexpected error occurred: {e}'}
        return jsonify(error_result), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html'), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)