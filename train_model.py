import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# File paths
DATA_PATH = 'data/diabetes_dataset.csv'
MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'diabetes_rf_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
COLUMNS_PATH = os.path.join(MODELS_DIR, 'columns.pkl')

def create_risk_labels(features):
    """Create multi-class risk labels based on medical criteria"""
    risk_levels = []
    for _, row in features.iterrows():
        # Initialize risk score
        risk_score = 0
        
        # Blood Glucose criteria (0-30 points)
        if row['Blood Glucose'] >= 126:  # Diabetes range
            risk_score += 30
        elif row['Blood Glucose'] >= 100:  # Pre-diabetes range
            risk_score += 15
        
        # HbA1c criteria (0-30 points)
        if row['HbA1c'] >= 6.5:  # Diabetes range
            risk_score += 30
        elif row['HbA1c'] >= 5.7:  # Pre-diabetes range
            risk_score += 15
        
        # BMI criteria (0-15 points)
        if row['BMI'] >= 30:  # Obese
            risk_score += 15
        elif row['BMI'] >= 25:  # Overweight
            risk_score += 7
        
        # Blood Pressure criteria (0-15 points)
        if row['Blood Pressure'] >= 140:  # Hypertension
            risk_score += 15
        elif row['Blood Pressure'] >= 120:  # Pre-hypertension
            risk_score += 7
        
        # Family history (0-10 points)
        if row['Family history'] == 1:
            risk_score += 10
            
        # Determine risk level based on total score
        # Maximum possible score is 100
        # Fair Risk: 0-40
        # Moderate Risk: 41-70
        # High Risk: 71-100
        if risk_score >= 71:
            risk_levels.append(2)  # High Risk
        elif risk_score >= 41:
            risk_levels.append(1)  # Moderate Risk
        else:
            risk_levels.append(0)  # Fair Risk
            
    return np.array(risk_levels)

def main():
    print("Starting model training process...")
    
    # Create models directory if it doesn't exist
    print(f"Creating models directory: {MODELS_DIR}")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load and preprocess data
    try:
        print("Loading dataset...")
        data = pd.read_csv(DATA_PATH)
        print(f"Dataset loaded successfully. Shape: {data.shape}")
        print(f"Columns in dataset: {list(data.columns)}")

        # Map dataset columns to our form fields
        # Note: This mapping depends on your actual dataset structure
        print("Mapping dataset columns to form fields...")
        
        # Create a mapping dictionary based on common diabetes dataset structures
        column_mapping = {}
        
        # Try to map common column variations
        for col in data.columns:
            col_lower = col.lower()
            if 'age' in col_lower:
                column_mapping['Age'] = col
            elif 'bmi' in col_lower:
                column_mapping['BMI'] = col
            elif 'glucose' in col_lower or 'blood_glucose' in col_lower or 'fasting' in col_lower:
                column_mapping['Blood Glucose'] = col
            elif 'blood_pressure' in col_lower or 'bp' in col_lower or 'systolic' in col_lower or 'pressure' in col_lower:
                column_mapping['Blood Pressure'] = col
            elif 'hba1c' in col_lower or 'hb' in col_lower or 'a1c' in col_lower:
                column_mapping['HbA1c'] = col
            elif 'insulin' in col_lower:
                column_mapping['Insulin Level'] = col
            elif 'skin' in col_lower or 'thickness' in col_lower:
                column_mapping['Skin thickness'] = col
            elif 'pregnanc' in col_lower:
                column_mapping['Pregnancies'] = col
            elif 'family' in col_lower or 'history' in col_lower:
                column_mapping['Family history'] = col
            elif 'physical' in col_lower or 'activity' in col_lower:
                column_mapping['Physical Activity'] = col
            elif 'smoking' in col_lower:
                column_mapping['Smoking status'] = col
            elif 'alcohol' in col_lower:
                column_mapping['Alcohol Intake'] = col
            elif 'diet' in col_lower:
                column_mapping['Diet_Type'] = col
            elif 'cholesterol' in col_lower:
                column_mapping['Cholesterol'] = col
            elif 'triglyceride' in col_lower:
                column_mapping['Triglycerides'] = col
            elif 'waist' in col_lower:
                column_mapping['Waist ratio'] = col

        print(f"Column mapping found: {column_mapping}")

        # Create new dataframe with mapped columns and proper data conversion
        mapped_data = pd.DataFrame()
        
        # Map existing columns with proper data conversion
        for form_field, dataset_col in column_mapping.items():
            print(f"Processing {form_field} from {dataset_col}")
            original_values = data[dataset_col]
            print(f"  Original values sample: {original_values.head()}")
            print(f"  Data type: {original_values.dtype}")
            print(f"  Unique values: {original_values.unique()[:10]}")
            
            # Handle specific conversions based on field type
            if form_field == 'Family history':
                # Convert Yes/No, True/False, 1/0 to binary
                if original_values.dtype == 'object':
                    mapped_values = original_values.str.lower().map({
                        'yes': 1, 'no': 0, 'true': 1, 'false': 0,
                        '1': 1, '0': 0, 1: 1, 0: 0
                    }).fillna(0).astype(int)
                else:
                    mapped_values = original_values.astype(int)
            elif form_field == 'Physical Activity':
                # Standardize physical activity values
                if original_values.dtype == 'object':
                    activity_map = {}
                    for val in original_values.unique():
                        if pd.isna(val):
                            continue
                        val_lower = str(val).lower()
                        if any(word in val_lower for word in ['high', 'vigorous', 'intense', '3', 'active']):
                            activity_map[val] = 'High'
                        elif any(word in val_lower for word in ['medium', 'moderate', '2', 'regular']):
                            activity_map[val] = 'Medium'
                        else:
                            activity_map[val] = 'Low'
                    mapped_values = original_values.map(activity_map).fillna('Medium')
                else:
                    # If numeric, assume 0=Low, 1=Medium, 2+=High
                    mapped_values = pd.cut(original_values, bins=[-1, 0.5, 1.5, float('inf')], 
                                         labels=['Low', 'Medium', 'High']).astype(str)
            elif form_field == 'Smoking status':
                # Convert smoking status
                if original_values.dtype == 'object':
                    smoking_map = {}
                    for val in original_values.unique():
                        if pd.isna(val):
                            continue
                        val_lower = str(val).lower()
                        if any(word in val_lower for word in ['yes', 'smoker', 'smoke', 'true', '1']):
                            smoking_map[val] = 'Smoker'
                        else:
                            smoking_map[val] = 'Non-Smoker'
                    mapped_values = original_values.map(smoking_map).fillna('Non-Smoker')
                else:
                    mapped_values = original_values.map({1: 'Smoker', 0: 'Non-Smoker'}).fillna('Non-Smoker')
            elif form_field == 'Diet_Type':
                # Handle diet type
                if original_values.dtype == 'object':
                    diet_map = {}
                    for val in original_values.unique():
                        if pd.isna(val):
                            continue
                        val_lower = str(val).lower()
                        if 'vegan' in val_lower:
                            diet_map[val] = 'Vegan'
                        elif 'vegetarian' in val_lower:
                            diet_map[val] = 'Vegetarian'
                        else:
                            diet_map[val] = 'Non-Vegetarian'
                    mapped_values = original_values.map(diet_map).fillna('Non-Vegetarian')
                else:
                    # If numeric, map to diet types
                    mapped_values = original_values.map({
                        0: 'Non-Vegetarian', 1: 'Vegetarian', 2: 'Vegan'
                    }).fillna('Non-Vegetarian')
            else:
                # For numeric fields, convert to float
                try:
                    if original_values.dtype == 'object':
                        # Try to convert strings to numeric
                        mapped_values = pd.to_numeric(original_values, errors='coerce')
                    else:
                        mapped_values = original_values.astype(float)
                    # Fill NaN values with reasonable defaults
                    if mapped_values.isna().any():
                        fill_value = mapped_values.median() if not mapped_values.isna().all() else 0
                        mapped_values = mapped_values.fillna(fill_value)
                except:
                    print(f"  Warning: Could not convert {form_field} to numeric, using median of dataset")
                    mapped_values = pd.Series([mapped_data.select_dtypes(include=[np.number]).median().median()] * len(original_values))
            
            mapped_data[form_field] = mapped_values
            print(f"  Converted values sample: {mapped_values.head()}")
            print(f"  Final data type: {mapped_values.dtype}")
            print(f"  Final unique values: {mapped_values.unique()[:10]}")
            print()
        
        # Create missing columns with reasonable defaults or derived values
        feature_cols = [
            'Age', 'BMI', 'Blood Glucose', 'Blood Pressure', 'HbA1c',
            'Insulin Level', 'Skin thickness', 'Pregnancies', 'Family history',
            'Physical Activity', 'Smoking status', 'Alcohol Intake', 'Diet_Type',
            'Cholesterol', 'Triglycerides', 'Waist ratio'
        ]

        # Fill missing columns with synthetic data or reasonable approximations
        for col in feature_cols:
            if col not in mapped_data.columns:
                print(f"Creating synthetic data for missing column: {col}")
                if col == 'Age':
                    mapped_data[col] = np.random.randint(20, 80, len(data))
                elif col == 'BMI':
                    mapped_data[col] = np.random.uniform(18, 40, len(data))
                elif col == 'Blood Glucose':
                    mapped_data[col] = np.random.uniform(70, 200, len(data))
                elif col == 'Blood Pressure':
                    mapped_data[col] = np.random.uniform(90, 160, len(data))
                elif col == 'HbA1c':
                    mapped_data[col] = np.random.uniform(4.0, 10.0, len(data))
                elif col == 'Insulin Level':
                    mapped_data[col] = np.random.uniform(10, 200, len(data))
                elif col == 'Skin thickness':
                    mapped_data[col] = np.random.uniform(15, 50, len(data))
                elif col == 'Pregnancies':
                    mapped_data[col] = np.random.randint(0, 8, len(data))
                elif col == 'Family history':
                    mapped_data[col] = np.random.choice([0, 1], len(data))
                elif col == 'Physical Activity':
                    mapped_data[col] = np.random.choice(['Low', 'Medium', 'High'], len(data))
                elif col == 'Smoking status':
                    mapped_data[col] = np.random.choice(['Smoker', 'Non-Smoker'], len(data))
                elif col == 'Alcohol Intake':
                    mapped_data[col] = np.random.randint(0, 15, len(data))
                elif col == 'Diet_Type':
                    mapped_data[col] = np.random.choice(['Non-Vegetarian', 'Vegetarian', 'Vegan'], len(data))
                elif col == 'Cholesterol':
                    mapped_data[col] = np.random.uniform(150, 300, len(data))
                elif col == 'Triglycerides':
                    mapped_data[col] = np.random.uniform(80, 400, len(data))
                elif col == 'Waist ratio':
                    mapped_data[col] = np.random.uniform(70, 120, len(data))

        # Create multi-class target based on risk factors
        print("Creating multi-class risk labels...")
        risk_labels = create_risk_labels(mapped_data)
        print("Risk labels created successfully")
        
        print("Risk level distribution:")
        risk_dist = pd.Series(risk_labels).value_counts()
        print(risk_dist)
        
        # Map numerical labels to risk categories for display
        risk_mapping = {0: 'Fair Risk', 1: 'Moderate Risk', 2: 'High Risk'}
        print("\nRisk categories distribution:")
        for label, count in risk_dist.items():
            print(f"{risk_mapping[label]}: {count} samples")

        # Prepare features
        X = mapped_data[feature_cols].copy()
        y = pd.Series(risk_labels).copy()

        print(f"\nFinal feature shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Features: {list(X.columns)}")
        
        # Verify data types before encoding
        print("\nData types before encoding:")
        for col in X.columns:
            print(f"  {col}: {X[col].dtype}")
            if X[col].dtype == 'object':
                print(f"    Unique values: {X[col].unique()}")

        # Encode categorical features
        print("\nEncoding categorical features...")
        categorical_cols = ['Physical Activity', 'Smoking status', 'Diet_Type']
        
        # Ensure categorical columns have the expected values
        for cat_col in categorical_cols:
            if cat_col in X.columns:
                print(f"Encoding {cat_col}: {X[cat_col].unique()}")
        
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        print(f"Features after encoding: {X_encoded.shape}")
        print(f"Encoded columns: {list(X_encoded.columns)}")
        
        # Ensure all columns are numeric
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object':
                print(f"Warning: Column {col} is still object type. Converting to numeric.")
                X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0)

        # Save column information
        print("Saving column information...")
        with open(COLUMNS_PATH, 'wb') as f:
            pickle.dump(X_encoded.columns.tolist(), f)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        print("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        print("Model trained successfully.")

        # Evaluate model
        print("Evaluating model...")
        train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
        test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
        
        print(f"Training Accuracy: {train_accuracy:.3f}")
        print(f"Testing Accuracy: {test_accuracy:.3f}")

        # Print classification report
        y_pred = model.predict(X_test_scaled)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Test risk distribution in predictions
        y_pred_test = model.predict(X_test_scaled)
        risk_mapping = {0: 'Fair Risk', 1: 'Moderate Risk', 2: 'High Risk'}
        risk_levels = [risk_mapping[pred] for pred in y_pred_test]
        risk_dist = pd.Series(risk_levels).value_counts()
        print(f"\nRisk Level Distribution in Test Set:")
        print(risk_dist)

        # Save model and scaler
        print("Saving model and scaler...")
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)

        print(f"Model saved to {MODEL_PATH}")
        print(f"Scaler saved to {SCALER_PATH}")
        print(f"Columns saved to {COLUMNS_PATH}")

        # Save sample predictions
        sample_df = pd.DataFrame(X_test)
        sample_df['Actual'] = y_test.values
        sample_df['Predicted_Class'] = y_pred
        sample_df['Risk_Level'] = risk_levels
        
        # Get probabilities for each class
        y_proba = model.predict_proba(X_test_scaled)
        for i, risk in enumerate(['Fair Risk', 'Moderate Risk', 'High Risk']):
            sample_df[f'Probability_{risk}'] = y_proba[:, i]
            
        sample_df.to_csv('sample_predictions.csv', index=False)
        print("Sample predictions saved to 'sample_predictions.csv'")

        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()