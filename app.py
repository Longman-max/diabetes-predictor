from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

os.makedirs(app.instance_path, exist_ok=True)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'your_secret_key_here'

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
    """Classify diabetes risk by combining model probability with clinical risk factors."""

    risk_score = 0

    if prob >= 0.7:
        risk_score += 3
    elif prob >= 0.3:
        risk_score += 2
    else:
        risk_score += 0

    if features.get('Blood Glucose', 0) >= 126:
        risk_score += 2
    elif features.get('Blood Glucose', 0) >= 100:
        risk_score += 1

    if features.get('HbA1c', 0) >= 6.5:
        risk_score += 2
    elif features.get('HbA1c', 0) >= 5.7:
        risk_score += 1

    if features.get('BMI', 0) >= 30:
        risk_score += 1

    if features.get('Blood Pressure', 0) >= 140:
        risk_score += 1

    if features.get('Family history') == '1':
        risk_score += 1

    if risk_score >= 5:
        return {
            'message': f"{prob*100:.1f}% High Risk",
            'advice': 'Your results show a high risk. Please consult a healthcare professional immediately for diagnosis and lifestyle guidance.',
            'class_name': 'high-risk'
        }
    elif risk_score >= 3:
        return {
            'message': f"{prob*100:.1f}% Medium Risk",
            'advice': 'You’re at moderate risk. Start making small healthy changes — eat balanced meals, stay active, and monitor your health regularly.',
            'class_name': 'medium-risk'
        }
    else:
        return {
            'message': f"{prob*100:.1f}% No Risk",
            'advice': 'Your risk is low — keep it that way! Maintain a healthy lifestyle with regular exercise and a nutritious diet.',
            'class_name': 'no-risk'
        }

@app.route('/')
def home():
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
        input_dict = {col: request.form.get(col, '') for col in [
            'Age', 'BMI', 'Blood Glucose', 'Blood Pressure', 'HbA1c',
            'Insulin Level', 'Skin thickness', 'Pregnancies', 'Family history',
            'Physical Activity', 'Smoking status', 'Alcohol Intake', 'Diet_Type',
            'Cholesterol', 'Triglycerides', 'Waiste ratio'
        ]}

        prev_values = input_dict.copy()

        for key in input_dict:
            try:
                input_dict[key] = float(input_dict[key])
            except ValueError:
                pass

        X_input = pd.DataFrame([input_dict])

        X_input = pd.get_dummies(X_input, drop_first=True)
        for col in columns:
            if col not in X_input:
                X_input[col] = 0
        X_input = X_input[columns]

        X_scaled = scaler.transform(X_input)

        prediction = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]
        risk_data = classify_risk(prob, input_dict)

        return jsonify(risk_data)

    except Exception as e:
        result = {'error': f'Prediction error: {e}'}
        print(f"Prediction error details: {str(e)}")
        return jsonify(result), 500

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