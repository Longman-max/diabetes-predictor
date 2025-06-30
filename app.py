from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model and scaler
with open('diabetes_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Map categorical values to numbers
    pa_map = {'Low': 0, 'Medium': 1, 'High': 2}
    smoke_map = {'Non-Smoker': 0, 'Smoker': 1}
    fam_map = {'0': 0, '1': 1}

    features = [
        float(request.form['Age']),
        float(request.form['BMI']),
        float(request.form['Blood Glucose']),
        float(request.form['Blood Pressure']),
        float(request.form['HbA1c']),
        float(request.form['Insulin Level']),
        float(request.form['Skin thickness']),
        float(request.form['Pregnancies']),
        fam_map.get(request.form['Family history'], 0),
        pa_map.get(request.form['Physical Activity'], 0),
        smoke_map.get(request.form['Smoking status'], 0),
        float(request.form['Alcohol Intake']),
        float(request.form['Diet Qualtiy']),
        float(request.form['Cholesterol']),
        float(request.form['Triglycerides']),
        float(request.form['Waiste ratio'])
    ]

    # Preprocess the input data
    features_scaled = scaler.transform([features])

    # Make prediction
    prediction = model.predict(features_scaled)
    result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)