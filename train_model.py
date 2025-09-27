import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier
import pickle

# Constants and file paths
CSV_PATH = "data/new_diabetic_data.csv"   # Source data file
MODELS_DIR = 'xgboostmodel'               # Directory for trained models
MODEL_RISK_PATH = os.path.join(MODELS_DIR, 'xgb_risk_model.pkl')
MODEL_PROG_PATH = os.path.join(MODELS_DIR, 'xgb_prog_model.pkl')
PIPELINE_PATH = os.path.join(MODELS_DIR, 'pipeline.pkl')
COLUMNS_PATH = os.path.join(MODELS_DIR, 'columns.pkl')
RANDOM_STATE = 42
TEST_SIZE = 0.2

def calculate_base_risk(row):
    """Calculate base risk from critical factors"""
    risk = 0
    
    # HbA1c risk (highest weight)
    if row['A1Cresult'] == '>8':
        risk += 40
    elif row['A1Cresult'] == '>7':
        risk += 30
    elif row['A1Cresult'] == 'Norm':
        risk += 10
        
    # Blood glucose risk
    if row['max_glu_serum'] == '>300':
        risk += 30
    elif row['max_glu_serum'] == '>200':
        risk += 20
    elif row['max_glu_serum'] == 'Norm':
        risk += 5
        
    # Hospital visits risk
    emergency_visits = float(row['number_emergency'] or 0)
    inpatient_visits = float(row['number_inpatient'] or 0)
    if emergency_visits + inpatient_visits >= 3:
        risk += 20
    elif emergency_visits + inpatient_visits >= 1:
        risk += 10
        
    # Medications risk
    num_meds = float(row['num_medications'] or 0)
    if num_meds >= 5:
        risk += 10
    elif num_meds >= 3:
        risk += 5
        
    return risk

def risk_class(score):
    """Determine risk class based on absolute risk score"""
    if score >= 70:  # High risk if score is 70 or above
        return 2  # High
    elif score >= 40:  # Moderate risk if score is between 40 and 69
        return 1  # Moderate
    else:  # Fair risk if score is below 40
        return 0  # Fair

def create_interaction_features(df):
    """Create interaction features for better prediction"""
    # Convert columns to float explicitly
    df['A1C_numeric'] = pd.to_numeric(df['A1C_numeric'], errors='coerce').fillna(0.0)
    df['max_glu_numeric'] = pd.to_numeric(df['max_glu_numeric'], errors='coerce').fillna(0.0)
    df['time_in_hospital'] = pd.to_numeric(df['time_in_hospital'], errors='coerce').fillna(0.0)
    df['number_diagnoses'] = pd.to_numeric(df['number_diagnoses'], errors='coerce').fillna(0.0)
    df['number_emergency'] = pd.to_numeric(df['number_emergency'], errors='coerce').fillna(0.0)
    
    # Create interaction features
    df['glucose_a1c'] = df['max_glu_numeric'] * df['A1C_numeric']
    df['hospital_severity'] = df['time_in_hospital'] * df['number_diagnoses']
    df['emergency_impact'] = df['number_emergency'] * df['number_diagnoses']
    
    return df

def main():
    """Enhanced training function with improved risk calculation and feature engineering"""
    print("Starting model training process...")
    
    # Create models directory if it doesn't exist
    print(f"Creating models directory: {MODELS_DIR}")
    os.makedirs(MODELS_DIR, exist_ok=True)

    try:
        # Load & basic cleaning
        print("Loading dataset...")
        df = pd.read_csv(CSV_PATH)
        df = df.replace("?", np.nan)
        print(f"Dataset loaded successfully. Shape: {df.shape}")

        # ensure encounter_id and patient_nbr are usable for sorting
        df['encounter_id'] = pd.to_numeric(df['encounter_id'], errors='coerce')
        df['patient_nbr'] = pd.to_numeric(df['patient_nbr'], errors='coerce')

        # Map A1Cresult & max_glu_serum to numeric proxies
        a1c_map = {
            'None': 0.0, 'Norm': 1.0, '>7': 2.0, '>8': 3.0
        }
        glu_map = {
            'None': 0.0, 'Norm': 1.0, '>200': 2.0, '>300': 3.0
        }
        df['A1C_numeric'] = df['A1Cresult'].astype(str).map(a1c_map)
        df['max_glu_numeric'] = df['max_glu_serum'].astype(str).map(glu_map)

        # Ensure numeric columns are numeric
        num_cols = [
            'num_lab_procedures', 'num_procedures', 'num_medications',
            'number_outpatient', 'number_emergency', 'number_inpatient',
            'number_diagnoses', 'time_in_hospital'
        ]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Calculate risk scores using the new method
        print("Calculating risk scores...")
        
        # Initialize numeric columns
        numeric_cols = ['number_inpatient', 'number_diagnoses', 'num_medications', 
                       'number_emergency', 'max_glu_numeric']
        for c in numeric_cols:
            if c not in df.columns:
                df[c] = 0
            df[c] = df[c].fillna(0)
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
        # Calculate base risk scores
        print("Calculating base risk scores...")
        df['risk_score_raw'] = df.apply(calculate_base_risk, axis=1)
        
        # Add age factor
        df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(50)
        age_factor = (df['age'] - 40) / 20  # Normalize age impact
        age_factor = age_factor.clip(-1, 1)  # Limit impact
        df['risk_score_raw'] = df['risk_score_raw'] * (1 + 0.1 * age_factor)  # Add 10% age impact
        
        # Create interaction features for model training
        df = create_interaction_features(df)

        # normalize 0-100
        minv = df['risk_score_raw'].min()
        maxv = df['risk_score_raw'].max()
        if pd.isna(minv) or pd.isna(maxv) or maxv == minv:
            df['risk_score'] = 0.0
        else:
            df['risk_score'] = 100 * (df['risk_score_raw'] - minv) / (maxv - minv)

        df['risk_class'] = df['risk_score'].apply(risk_class)

        # Build progression label using patient_nbr
        df = df.sort_values(['patient_nbr', 'encounter_id'])
        df['next_risk_class'] = df.groupby('patient_nbr')['risk_class'].shift(-1)
        df['progression'] = (df['next_risk_class'] > df['risk_class']).astype('Int64')
        
        # drop last encounters (no next visit)
        df = df.dropna(subset=['next_risk_class']).copy()
        df['progression'] = df['progression'].astype(int)

        # Feature list: remove identifiers and targets
        exclude = {
            'encounter_id', 'patient_nbr', 'risk_score_raw', 'risk_score',
            'risk_class', 'next_risk_class', 'progression'
        }
        features = [c for c in df.columns if c not in exclude]
        features = [c for c in features if df[c].notna().any()]

        X = df[features]
        y_risk = df['risk_class']
        y_prog = df['progression']
        # Preprocessing pipeline
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='drop')

        # Train-test split
        X_train, X_test, y_risk_train, y_risk_test, y_prog_train, y_prog_test = train_test_split(
            X, y_risk, y_prog, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_risk
        )

        # Enhanced XGBoost pipelines with better parameters
        # Risk classification (multiclass)
        xgb_risk = Pipeline(steps=[
            ('preproc', preprocessor),
            ('model', XGBClassifier(
                objective='multi:softprob',
                eval_metric='mlogloss',
                use_label_encoder=False,
                random_state=RANDOM_STATE,
                tree_method='hist',
                verbosity=0,
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1
            ))
        ])
        xgb_risk.fit(X_train, y_risk_train)
        y_risk_pred = xgb_risk.predict(X_test)

        # Progression (binary)
        xgb_prog = Pipeline(steps=[
            ('preproc', preprocessor),
            ('model', XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=RANDOM_STATE,
                tree_method='hist',
                verbosity=0
            ))
        ])
        xgb_prog.fit(X_train, y_prog_train)
        y_prog_pred = xgb_prog.predict(X_test)

        # Evaluation
        print("==== XGBoost - Risk Stratification (multiclass) ====")
        print(classification_report(y_risk_test, y_risk_pred))
        print("Accuracy:", accuracy_score(y_risk_test, y_risk_pred))
        print("Macro F1:", f1_score(y_risk_test, y_risk_pred, average='macro'))

        print("\n==== XGBoost - Progression (binary) ====")
        print(classification_report(y_prog_test, y_prog_pred))
        print("Accuracy:", accuracy_score(y_prog_test, y_prog_pred))
        print("F1 (binary):", f1_score(y_prog_test, y_prog_pred, average='binary'))

        # Save models and pipeline
        print("\nSaving models and preprocessing pipeline...")
        with open(MODEL_RISK_PATH, 'wb') as f:
            pickle.dump(xgb_risk, f)
        with open(MODEL_PROG_PATH, 'wb') as f:
            pickle.dump(xgb_prog, f)
        with open(PIPELINE_PATH, 'wb') as f:
            pickle.dump(preprocessor, f)
            
        # Save feature columns for later reference
        with open(COLUMNS_PATH, 'wb') as f:
            pickle.dump(features, f)

        print(f"Models saved to {MODELS_DIR}/")
        print(f"Risk Model: {os.path.basename(MODEL_RISK_PATH)}")
        print(f"Progression Model: {os.path.basename(MODEL_PROG_PATH)}")
        print(f"Pipeline: {os.path.basename(PIPELINE_PATH)}")
        print(f"Feature columns: {os.path.basename(COLUMNS_PATH)}")

        # Save metrics for comparison
        metrics = {
            'model': 'xgboost',
            'risk_accuracy': accuracy_score(y_risk_test, y_risk_pred),
            'risk_macro_f1': f1_score(y_risk_test, y_risk_pred, average='macro'),
            'prog_accuracy': accuracy_score(y_prog_test, y_prog_pred),
            'prog_f1': f1_score(y_prog_test, y_prog_pred, average='binary')
        }
        pd.DataFrame([metrics]).to_csv("metrics_xgboost.csv", index=False)
        print("\nSaved metrics_xgboost.csv")

        # Save sample predictions with risk levels for analysis
        predictions_df = df[df.index.isin(X_test.index)].copy()
        predictions_df['Actual_Risk'] = y_risk_test
        predictions_df['Predicted_Risk'] = y_risk_pred
        predictions_df['Actual_Progression'] = y_prog_test 
        predictions_df['Predicted_Progression'] = y_prog_pred
        
        # Get probabilities for risk classes
        risk_probs = xgb_risk.predict_proba(X_test)
        for i, level in enumerate(['Fair', 'Moderate', 'High']):
            predictions_df[f'Risk_{level}_Probability'] = risk_probs[:, i]
        
        # Save predictions
        predictions_df.to_csv('diabetes_predictions_with_risk.csv', index=False)
        print("\nSaved predictions to diabetes_predictions_with_risk.csv")

        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()