"""
Data processing utilities for federated learning with differential privacy.
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_breast_cancer_data():
    """Load and return the breast cancer dataset with target mapping documentation."""
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    
    # Get dataset info for target mapping confirmation
    data = load_breast_cancer()
    print("=== BREAST CANCER DATASET INFO ===")
    print(f"Dataset shape: {X.shape}")
    print(f"Feature names: {len(data.feature_names)} features")
    print(f"Target names: {data.target_names}")
    print("\n=== TARGET MAPPING ===")
    print("0 = malignant (cancer)")
    print("1 = benign (no cancer)")
    print("\nThis mapping is CONFIRMED and will be used throughout the analysis.")
    
    return X, y


def create_hospital_splits(X, y, test_size=0.2, hospital_A_malignant_ratio=0.6, random_state=42):
    """Create non-IID hospital splits and external test set."""
    print("=== CREATING HOSPITAL SPLITS ===")
    print(f"Test set size: {test_size*100}%")
    print(f"Hospital A will get {hospital_A_malignant_ratio*100}% of malignant cases")
    print(f"Hospital A will get {(1-hospital_A_malignant_ratio)*100}% of benign cases")
    print("This creates non-IID distribution between hospitals.\n")
    
    # Create output directory
    os.makedirs("../data", exist_ok=True)
    
    # First, split off external test set (stratified)
    X_hospitals, X_test, y_hospitals, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"After test split:")
    print(f"  Hospitals data: {len(X_hospitals)} samples")
    print(f"  Test data: {len(X_test)} samples")
    
    # Create non-IID split between hospitals
    malignant_mask = y_hospitals == 0
    benign_mask = y_hospitals == 1
    
    malignant_indices = y_hospitals[malignant_mask].index
    benign_indices = y_hospitals[benign_mask].index
    
    # Shuffle indices
    np.random.seed(random_state)
    malignant_indices = np.random.permutation(malignant_indices)
    benign_indices = np.random.permutation(benign_indices)
    
    # Split malignant cases
    n_malignant_A = int(len(malignant_indices) * hospital_A_malignant_ratio)
    malignant_A_indices = malignant_indices[:n_malignant_A]
    malignant_B_indices = malignant_indices[n_malignant_A:]
    
    # Split benign cases (opposite ratio to create non-IID distribution)
    hospital_A_benign_ratio = 1 - hospital_A_malignant_ratio
    n_benign_A = int(len(benign_indices) * hospital_A_benign_ratio)
    benign_A_indices = benign_indices[:n_benign_A]
    benign_B_indices = benign_indices[n_benign_A:]
    
    # Combine indices for each hospital
    hospital_A_indices = np.concatenate([malignant_A_indices, benign_A_indices])
    hospital_B_indices = np.concatenate([malignant_B_indices, benign_B_indices])
    
    # Create hospital dataframes
    hospital_A_X = X_hospitals.loc[hospital_A_indices]
    hospital_A_y = y_hospitals.loc[hospital_A_indices]
    hospital_A_data = pd.concat([hospital_A_X, hospital_A_y], axis=1)
    
    hospital_B_X = X_hospitals.loc[hospital_B_indices]
    hospital_B_y = y_hospitals.loc[hospital_B_indices]
    hospital_B_data = pd.concat([hospital_B_X, hospital_B_y], axis=1)
    
    # Create test dataframe
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # Save to CSV files
    hospital_A_data.to_csv("../data/hospital_A.csv", index=False)
    hospital_B_data.to_csv("../data/hospital_B.csv", index=False)
    test_data.to_csv("../data/test_set.csv", index=False)
    
    return hospital_A_data, hospital_B_data, test_data


def prepare_centralized_data(hospital_A, hospital_B):
    """Combine hospital data for centralized training."""
    print("=== PREPARING CENTRALIZED TRAINING DATA ===")
    
    # Combine both hospitals
    combined_data = pd.concat([hospital_A, hospital_B], axis=0, ignore_index=True)
    
    # Separate features and target
    X = combined_data.iloc[:, :-1]  # All columns except last
    y = combined_data.iloc[:, -1]   # Last column (target)
    
    print(f"Combined training data: {len(X)} samples, {X.shape[1]} features")
    print(f"Class distribution in combined data:")
    print(f"  Malignant: {sum(y == 0)} ({sum(y == 0)/len(y):.2%})")
    print(f"  Benign: {sum(y == 1)} ({sum(y == 1)/len(y):.2%})")
    
    return X, y


def analyze_hospital_splits(hospital_A, hospital_B, test_data):
    """Analyze and display hospital split statistics."""
    
    print("=== HOSPITAL SPLIT ANALYSIS ===")
    
    # Extract targets
    y_A = hospital_A.iloc[:, -1]
    y_B = hospital_B.iloc[:, -1]
    y_test = test_data.iloc[:, -1]
    
    # Calculate statistics
    datasets = {
        'Hospital A': y_A,
        'Hospital B': y_B,
        'Test Set': y_test
    }
    
    summary_data = []
    
    for name, target in datasets.items():
        total = len(target)
        malignant = sum(target == 0)
        benign = sum(target == 1)
        malignant_pct = malignant / total * 100
        benign_pct = benign / total * 100
        
        summary_data.append({
            'Dataset': name,
            'Total Samples': total,
            'Malignant': malignant,
            'Malignant %': f"{malignant_pct:.1f}%",
            'Benign': benign,
            'Benign %': f"{benign_pct:.1f}%"
        })
        
        print(f"\n{name}:")
        print(f"  Total: {total} samples")
        print(f"  Malignant: {malignant} ({malignant_pct:.1f}%)")
        print(f"  Benign: {benign} ({benign_pct:.1f}%)")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    print("\n=== SUMMARY TABLE ===")
    
    # Save summary
    os.makedirs("../results", exist_ok=True)
    summary_df.to_csv('../results/hospital_split_summary.csv', index=False)
    
    return summary_df