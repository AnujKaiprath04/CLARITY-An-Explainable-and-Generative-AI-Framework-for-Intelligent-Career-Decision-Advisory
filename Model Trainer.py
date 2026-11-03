"""
Model Training Module
Trains a Random Forest Classifier and saves the model and scaler.
"""

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def train_and_save_model():
    data_path = 'data/student_career_data.csv'
    
    if not os.path.exists(data_path):
        print("Data file not found. Please run generate_data.py first.")
        return

    # Load data
    df = pd.read_csv(data_path)
    
    X = df.drop('Career_Choice', axis=1)
    y = df['Career_Choice']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest (Good for SHAP explainability)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/career_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Model and scaler saved to 'models/' directory.")

if __name__ == "__main__":
    train_and_save_model()