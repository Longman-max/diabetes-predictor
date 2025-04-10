from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model and scaler
with open('diabetes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [
        float(request.form['Pregnancies']),
        float(request.form['Glucose']),
        float(request.form['BloodPressure']),
        float(request.form['SkinThickness']),
        float(request.form['Insulin']),
        float(request.form['BMI']),
        float(request.form['DiabetesPedigreeFunction']),
        float(request.form['Age'])
    ]

    # Preprocess the input data
    features_scaled = scaler.transform([features])

    # Make prediction
    prediction = model.predict(features_scaled)
    result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)