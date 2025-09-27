from flask import Flask, render_template, request, jsonify, send_from_directory
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

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
    models_dir = 'xgboostmodel'
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found. Please run train_model.py first.")
        raise FileNotFoundError(f"Models directory '{models_dir}' not found")
    
    model_risk_path = os.path.join(models_dir, 'xgb_risk_model.pkl')
    model_prog_path = os.path.join(models_dir, 'xgb_prog_model.pkl')
    pipeline_path = os.path.join(models_dir, 'pipeline.pkl')
    columns_path = os.path.join(models_dir, 'columns.pkl')
    
    # Check if all required files exist
    required_files = [model_risk_path, model_prog_path, pipeline_path, columns_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Required model file '{file_path}' not found. Please run train_model.py first.")
            raise FileNotFoundError(f"Model file '{file_path}' not found")
    
    with open(model_risk_path, 'rb') as f:
        risk_model = pickle.load(f)
        print(f"\nRisk Model type: {type(risk_model)}")
        if hasattr(risk_model, 'steps'):
            print("Risk Model pipeline steps:", [s[0] for s in risk_model.steps])
    
    with open(model_prog_path, 'rb') as f:
        prog_model = pickle.load(f)
        print(f"\nProgression Model type: {type(prog_model)}")
        if hasattr(prog_model, 'steps'):
            print("Progression Model pipeline steps:", [s[0] for s in prog_model.steps])
    
    with open(columns_path, 'rb') as f:
        features = pickle.load(f)
    print("\nModel files loaded successfully.")
    print(f"Expected features: {features}")
    print(f"Number of features: {len(features)}")
except Exception as e:
    print(f"Error loading model files: {e}")
    raise

def calculate_risk_score(data):
    """Calculate risk score based on input data"""
    score = 0
    
    # HbA1c risk (highest weight)
    if data.get('A1Cresult') == '>8':
        score += 40
    elif data.get('A1Cresult') == '>7':
        score += 30
    elif data.get('A1Cresult') == 'Norm':
        score += 10
        
    # Blood glucose risk
    if data.get('max_glu_serum') == '>300':
        score += 30
    elif data.get('max_glu_serum') == '>200':
        score += 20
    elif data.get('max_glu_serum') == 'Norm':
        score += 5
        
    # Hospital visits risk
    emergency_visits = float(data.get('number_emergency', 0))
    inpatient_visits = float(data.get('number_inpatient', 0))
    if emergency_visits + inpatient_visits >= 3:
        score += 20
    elif emergency_visits + inpatient_visits >= 1:
        score += 10
        
    # Medications risk
    num_meds = float(data.get('num_medications', 0))
    if num_meds >= 5:
        score += 10
    elif num_meds >= 3:
        score += 5
        
    # Age impact
    age = float(data.get('age', 0))
    if age >= 60:
        score += 10
    elif age >= 45:
        score += 5
        
    return score

def get_risk_result(probabilities, input_data):
    """
    Get risk classification result based on both model probabilities and direct risk calculation
    """
    # Calculate direct risk score
    risk_score = calculate_risk_score(input_data)
    print(f"\nDirect Risk Score Calculation:")
    print(f"Total Risk Score: {risk_score}")
    
    # Determine risk level based on absolute score
    if risk_score >= 70:
        risk_level = 'High'
    elif risk_score >= 40:
        risk_level = 'Moderate'
    else:
        risk_level = 'Fair'
        
    print(f"\n=== RISK CLASSIFICATION ===")
    print(f"Model Probabilities: Fair={probabilities[0]:.3f}, Moderate={probabilities[1]:.3f}, High={probabilities[2]:.3f}")
    print(f"Direct Risk Score: {risk_score}")
    print(f"Final Risk Level: {risk_level}")
    
    if risk_level == 'High':
        result = {
            'message': "High Risk",
            'risk_score': risk_score,
            'advice': 'Your results indicate a high risk. Please consult a healthcare professional immediately for diagnosis and guidance.',
            'class_name': 'high-risk'
        }
    elif risk_level == 'Moderate':
        result = {
            'message': "Moderate Risk",
            'risk_score': risk_score,
            'advice': 'You are at moderate risk. It is advisable to monitor your health and consult a doctor for preventative measures.',
            'class_name': 'moderate-risk'
        }
    else:  # Fair Risk
        result = {
            'message': "Fair Risk",
            'risk_score': risk_score,
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
        
        # Get form data with defaults
        form_data = request.form.to_dict()
        print("Received form data:", form_data)
        
        # Initialize input dictionary with mandatory fields
        input_dict = {
            'race': form_data.get('race', 'Unknown'),
            'gender': form_data.get('gender', 'Unknown'),
            'age': form_data.get('Age', '0'),
            'weight': form_data.get('Weight', '0'),
            'time_in_hospital': '1',  # Default value
            'num_lab_procedures': '0',  # Default value
            'num_procedures': '0',  # Default value
            'num_medications': form_data.get('num_medications', '0'),
            'number_outpatient': form_data.get('number_outpatient', '0'),
            'number_emergency': form_data.get('number_emergency', '0'),
            'number_inpatient': form_data.get('number_inpatient', '0'),
            'number_diagnoses': '1',  # Default value
            'max_glu_serum': 'None',  # Will be mapped from Blood Glucose
            'A1Cresult': 'None'      # Will be mapped from HbA1c
        }

        try:
            # Convert glucose to max_glu_serum category
            glucose = float(form_data.get('Blood Glucose', 0))
            if glucose > 300:
                input_dict['max_glu_serum'] = '>300'
            elif glucose > 200:
                input_dict['max_glu_serum'] = '>200'
            elif glucose >= 70:
                input_dict['max_glu_serum'] = 'Norm'
            else:
                input_dict['max_glu_serum'] = 'None'

            # Convert HbA1c to A1Cresult category
            hba1c = float(form_data.get('HbA1c', 0))
            if hba1c > 8:
                input_dict['A1Cresult'] = '>8'
            elif hba1c > 7:
                input_dict['A1Cresult'] = '>7'
            elif hba1c >= 5:
                input_dict['A1Cresult'] = 'Norm'
            else:
                input_dict['A1Cresult'] = 'None'

        except ValueError as e:
            return jsonify({'error': f'Invalid numeric value: {str(e)}'}), 400

        print("Processed input data:")
        for field, value in input_dict.items():
            print(f"  {field}: '{value}'")
            
        # Convert numeric fields
        numeric_fields = [
            'age', 'weight', 'time_in_hospital', 'num_lab_procedures',
            'num_procedures', 'num_medications', 'number_outpatient',
            'number_emergency', 'number_inpatient', 'number_diagnoses'
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
        
        # Validate race and gender have values
        categorical_fields = ['race', 'gender']
        print("\nCATEGORICAL VALUES:")
        for field in categorical_fields:
            print(f"  {field}: '{input_dict[field]}'")
            if input_dict[field] in ['', None, 'Unknown']:
                return jsonify({
                    'error': f'Please select a value for {field}'
                }), 400
        
        # Create DataFrame
        X_input = pd.DataFrame([input_dict])
        print(f"\nDATAFRAME CREATED:")
        print(f"Shape: {X_input.shape}")
        print(f"Columns: {X_input.columns.tolist()}")
        print(f"Values:\n{X_input.iloc[0].to_dict()}")
        
        # Extract required features
        print(f"\nEXPECTED FEATURES ({len(features)}): {features}")
        
        # Keep only features used during training
        missing_features = [f for f in features if f not in X_input.columns]
        if missing_features:
            print(f"\nMISSING FEATURES: {missing_features}")
            for f in missing_features:
                X_input[f] = 0 if f not in ['A1Cresult', 'max_glu_serum'] else 'None'
        
        # Remove extra columns not used in training
        extra_cols = [c for c in X_input.columns if c not in features]
        if extra_cols:
            print(f"\nREMOVING EXTRA COLUMNS: {extra_cols}")
            X_input = X_input.drop(columns=extra_cols)
        
        # Ensure all features are present in correct order
        X_final = X_input[features]
        
        print(f"\nFINAL DATAFRAME FOR PREDICTION:")
        print(f"Shape: {X_final.shape}")
        print(f"Features match training data: {list(X_final.columns) == features}")
        print("Final values:")
        for i, (col, val) in enumerate(X_final.iloc[0].items()):
            print(f"  {i:2d}. {col}: {val}")
        
        # Make predictions using pipelines
        print("\nMAKING PREDICTIONS...")
        
        try:
            # Risk prediction
            print("Making risk prediction...")
            risk_probs = risk_model.predict_proba(X_final)
            print(f"Risk model output shape: {risk_probs.shape}")
            print(f"Risk probabilities (raw): {risk_probs}")
            risk_probs = risk_probs[0]  # Get first (only) prediction
            print(f"Final risk probabilities: {risk_probs}")
            
            # Progression prediction
            print("\nMaking progression prediction...")
            prog_probs = prog_model.predict_proba(X_final)
            print(f"Progression model output shape: {prog_probs.shape}")
            print(f"Progression probabilities (raw): {prog_probs}")
            prog_prob = prog_probs[0][1]  # Probability of progression
            print(f"Final progression probability: {prog_prob:.3f}")
            
            # Get risk classification result and add progression info
            risk_data = get_risk_result(risk_probs, input_dict)
            risk_data['progression_probability'] = float(prog_prob)
            
            if prog_prob > 0.5:
                risk_data['progression_warning'] = ('Warning: There is a high chance your condition may worsen. '
                                                 'Please consult your healthcare provider for preventive measures.')
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            print("Model information:")
            print(f"Risk model type: {type(risk_model)}")
            print(f"Progression model type: {type(prog_model)}")
            print(f"Input data shape: {X_final.shape}")
            print(f"Input data types:\n{X_final.dtypes}")
            raise
        
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