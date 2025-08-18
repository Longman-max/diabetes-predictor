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
    with open('models/diabetes_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    print("Model files loaded successfully")
except Exception as e:
    print(f"Error loading model files: {e}")
    raise

def classify_risk(prob, features):
    """Classify risk based on probability and feature values, using clinical guidelines"""
    # Definite diabetes: clinical diagnosis
    if (features.get('Blood Glucose', 0) >= 126 or features.get('HbA1c', 0) >= 6.5):
        return f"High Risk ({prob*100:.1f}%)"

    # Count risk factors
    risk_factors = 0
    if features.get('Blood Glucose', 0) >= 100:  # Prediabetes threshold
        risk_factors += 2
    if features.get('HbA1c', 0) >= 5.7:
        risk_factors += 2
    if features.get('BMI', 0) >= 30:
        risk_factors += 1
    if features.get('Family history') == '1':
        risk_factors += 1
    if features.get('Blood Pressure', 0) >= 140:
        risk_factors += 1

    # Model-based thresholds (adjust as needed)
    if prob >= 0.7 or risk_factors >= 4:
        return f"High Risk ({prob*100:.1f}%)"
    elif prob >= 0.3 or risk_factors >= 2:
        return f"Medium Risk ({prob*100:.1f}%)"
    else:
        return f"No Risk ({prob*100:.1f}%)"

@app.route('/')
def home():
    # Pass previous form values if available
    prev_values = None
    if 'prev_values' in request.args:
        import json
        try:
            prev_values = json.loads(request.args['prev_values'])
        except Exception:
            prev_values = None
    return render_template('index.html', prev_values=prev_values)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():

    import json
    try:
        # Gather input from form fields
        input_dict = {col: request.form.get(col, '') for col in [
            'Age', 'BMI', 'Blood Glucose', 'Blood Pressure', 'HbA1c', 
            'Insulin Level', 'Skin thickness', 'Pregnancies', 'Family history',
            'Physical Activity', 'Smoking status', 'Alcohol Intake', 'Diet Qualtiy',
            'Cholesterol', 'Triglycerides', 'Waiste ratio'
        ]}

        # Save original values for repopulation
        prev_values = input_dict.copy()

        # Convert numeric features for prediction
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

        # Custom messages for each risk level
        if 'High Risk' in risk_level:
            advice = 'Your results show a high risk. Please consult a healthcare professional immediately for diagnosis and lifestyle guidance.'
        elif 'Medium Risk' in risk_level:
            advice = 'You’re at moderate risk. Start making small healthy changes — eat balanced meals, stay active, and monitor your health regularly.'
        else:
            advice = 'Your risk is low — keep it that way! Maintain a healthy lifestyle with regular exercise and a nutritious diet.'

        result = f"{risk_level}<br><span class='advice'>{advice}</span>"

    except Exception as e:
        result = f'Prediction error: {e}'
        prev_values = None
        print(f"Prediction error details: {str(e)}")

    flash(result)
    # Pass previous values as JSON in query string
    return redirect(url_for('home', prev_values=json.dumps(prev_values)))

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
