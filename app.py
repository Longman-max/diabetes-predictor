from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__, 
            static_folder='static',  # specify the static folder
            template_folder='templates')  # specify the template folder

# Ensure the instance folder exists
os.makedirs(app.instance_path, exist_ok=True)

# Configure static file serving
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching during development
app.secret_key = 'your_secret_key_here'

# Load model, scaler, and columns
try:
    print("Loading model files...")
    with open('diabetes_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    print("Model files loaded successfully")
except Exception as e:
    print(f"Error loading model files: {e}")
    raise

def classify_risk(prob, features):
    """Classify risk based on probability and feature values"""
    # Check for definite high risk indicators first
    if (features.get('Blood Glucose', 0) >= 126 or  # Diabetes threshold
        features.get('HbA1c', 0) >= 6.5):          # Diabetes threshold
        return 'High'
    
    # Count risk factors
    risk_factors = 0
    
    # Check key indicators with clinical thresholds
    if features.get('Blood Glucose', 0) >= 100:  # Prediabetes threshold
        risk_factors += 2
    if features.get('HbA1c', 0) >= 5.7:         # Prediabetes threshold
        risk_factors += 2
    if features.get('BMI', 0) >= 30:            # Obesity threshold
        risk_factors += 1
    if features.get('Family history') == '1':    # Family history
        risk_factors += 1
    if features.get('Blood Pressure', 0) >= 140: # Hypertension threshold
        risk_factors += 1
    
    # Determine risk level based on both probability and risk factors
    if prob >= 0.5 or risk_factors >= 4:
        return 'High'
    elif prob >= 0.3 or risk_factors >= 2:
        return 'Moderate'
    return 'Low'
    
    # Adjust risk level based on risk factors
    if risk_factors >= 3:
        return 'High'
    elif risk_factors >= 2:
        return 'Moderate' if base_risk == 'Low' else 'High'
    elif risk_factors >= 1:
        return 'Moderate' if base_risk == 'Low' else base_risk
        
    return base_risk

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gather input from form fields
        input_dict = {col: request.form.get(col, '') for col in [
            'Age', 'BMI', 'Blood Glucose', 'Blood Pressure', 'HbA1c', 
            'Insulin Level', 'Skin thickness', 'Pregnancies', 'Family history',
            'Physical Activity', 'Smoking status', 'Alcohol Intake', 'Diet Qualtiy',
            'Cholesterol', 'Triglycerides', 'Waiste ratio'
        ]}

        # Convert numeric features
        for key in input_dict:
            try:
                input_dict[key] = float(input_dict[key])
            except ValueError:
                pass  # Keep categorical as string

        # Create input DataFrame
        X_input = pd.DataFrame([input_dict])

        # One-hot encode and align columns
        X_input = pd.get_dummies(X_input, drop_first=True)
        for col in columns:
            if col not in X_input:
                X_input[col] = 0
        X_input = X_input[columns]

        # Scale features
        X_scaled = scaler.transform(X_input)

        # Predict
        prediction = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]  # Probability of class 1
        risk_level = classify_risk(prob, input_dict)
        
        # Format probability as percentage
        prob_percentage = f"{prob * 100:.1f}%"
        
        # Build detailed result message based on probability and prediction
        if prob >= 0.7:  # High probability
            result = f"High Risk of Diabetes (Probability: {prob_percentage})"
        elif prob >= 0.5:  # Moderate to high probability
            result = f"High Risk of Diabetes (Probability: {prob_percentage})"
        elif prob >= 0.3:  # Moderate probability
            result = f"Moderate Risk of Diabetes (Probability: {prob_percentage})"
        else:  # Low probability
            result = f"Low Risk of Diabetes (Probability: {prob_percentage})"

        # Add risk factor assessment for more context
        if prediction == 1:
            # Check for critical indicators
            critical_factors = []
            if float(input_dict.get('Blood Glucose', 0)) > 125:
                critical_factors.append("elevated blood glucose")
            if float(input_dict.get('HbA1c', 0)) > 6.5:
                critical_factors.append("high HbA1c")
            if critical_factors:
                factors_text = " and ".join(critical_factors)
                result += f" - Critical indicators: {factors_text}"

    except Exception as e:
        result = f'Prediction error: {e}'
        print(f"Prediction error details: {str(e)}")

    flash(result)
    return redirect(url_for('home'))

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html'), 500

# Route to serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
