from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import numpy as np
import pandas as pd

# Load model, scaler, and columns
with open('diabetes_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for flash messages

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect all features as per FEATURE_COLS in train_model.py
        input_dict = {col: request.form.get(col, '') for col in [
            'Age', 'Gender', 'BMI', 'Family_History', 'Physical_Activity', 'Diet_Type',
            'Smoking_Status', 'Alcohol_Intake', 'Stress_Level', 'Hypertension', 'Cholesterol_Level',
            'Fasting_Blood_Sugar', 'Postprandial_Blood_Sugar', 'HBA1C', 'Heart_Rate', 'Waist_Hip_Ratio',
            'Urban_Rural', 'Health_Insurance', 'Regular_Checkups', 'Medication_For_Chronic_Conditions',
            'Pregnancies', 'Polycystic_Ovary_Syndrome', 'Glucose_Tolerance_Test_Result', 'Vitamin_D_Level',
            'C_Protein_Level', 'Thyroid_Condition']}
        # Convert numerics
        for key in input_dict:
            try:
                input_dict[key] = float(input_dict[key])
            except ValueError:
                pass  # keep as string for categorical
        # Create DataFrame
        X_input = pd.DataFrame([input_dict])
        # One-hot encode and align columns
        X_input = pd.get_dummies(X_input, drop_first=True)
        for col in columns:
            if col not in X_input:
                X_input[col] = 0
        X_input = X_input[columns]
        # Scale
        X_scaled = scaler.transform(X_input)
        prediction = model.predict(X_scaled)
        result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
    except Exception as e:
        result = f'Prediction error: {e}'
        print(result)
    flash(result)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)