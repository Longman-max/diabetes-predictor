# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load the updated dataset with 16 features
# Make sure the filename and column names match your actual dataset
data = pd.read_csv('diabetes_dataset.csv')

# Print columns to help with debugging
print('Columns in dataset:', list(data.columns))

# Define features (X) and target (y) - update these to match your dataset
feature_cols = ['Age', 'BMI', 'Blood Glucose', 'Blood Pressure', 'HbA1c', 'Insulin Level', 'Skin thickness', 'Pregnancies', 'Family history', 'Physical Activity', 'Smoking status', 'Alcohol Intake', 'Diet Qualtiy', 'Cholesterol', 'Triglycerides', 'Waiste ratio']
target_col = 'Outcome'  # Change if your target column is named differently

try:
    X = data[feature_cols]
    y = data[target_col]
    print("Features (X) and Target (y) defined.")
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
except KeyError as e:
    print(f"KeyError: {e}. Please check that all feature and target column names match your dataset exactly.")
    raise

# Preprocess the data: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save the trained model and scaler using pickle
with open('diabetes_rf_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler have been saved successfully as 'diabetes_rf_model.pkl' and 'scaler.pkl'.")