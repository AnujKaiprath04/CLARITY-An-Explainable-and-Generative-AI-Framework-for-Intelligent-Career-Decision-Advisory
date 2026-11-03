"""
Data Generation Module
Generates a synthetic dataset for the Career Decision Advisory System.
Creates logical correlations (e.g., High Risk + High Entrepreneurial = Startup).
"""

import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=2000):
    np.random.seed(42)
    
    # 1. Generate Base Features
    tech_skill = np.random.randint(20, 100, num_samples)
    comm_skill = np.random.randint(20, 100, num_samples)
    
    # Financial Stability: 0 (Low), 1 (Medium), 2 (High)
    fin_stability = np.random.choice([0, 1, 2], num_samples, p=[0.3, 0.5, 0.2])
    
    risk_tolerance = np.random.randint(10, 100, num_samples)
    higher_studies_interest = np.random.randint(10, 100, num_samples)
    entrepreneurial_mindset = np.random.randint(10, 100, num_samples)
    
    # 2. Logic to determine the target variable (Career Choice)
    # 0: Job, 1: Higher Studies, 2: Startup
    target = []
    
    for i in range(num_samples):
        score_job = (tech_skill[i] * 1.5) + comm_skill[i] + (fin_stability[i] * 20)
        score_studies = higher_studies_interest[i] * 2 + (tech_skill[i] * 0.5) + (fin_stability[i] * 30)
        score_startup = entrepreneurial_mindset[i] * 2.5 + risk_tolerance[i] * 2 - (fin_stability[i] * -10)
        
        # Add some noise for realism
        score_job += np.random.normal(0, 20)
        score_studies += np.random.normal(0, 20)
        score_startup += np.random.normal(0, 20)
        
        scores = [score_job, score_studies, score_startup]
        target.append(np.argmax(scores))
        
    # 3. Create DataFrame
    data = pd.DataFrame({
        'Technical_Skill': tech_skill,
        'Communication_Skill': comm_skill,
        'Financial_Stability': fin_stability,
        'Risk_Tolerance': risk_tolerance,
        'Interest_Higher_Studies': higher_studies_interest,
        'Entrepreneurial_Mindset': entrepreneurial_mindset,
        'Career_Choice': target
    })
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/student_career_data.csv', index=False)
    print(f"Successfully generated {num_samples} records in 'data/student_career_data.csv'")

if __name__ == "__main__":
    generate_synthetic_data()