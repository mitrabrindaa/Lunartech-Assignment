"""
Training functions for centralized and federated learning approaches.
"""

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .models import FederatedNet
from .federated_client import FlowerClient
from .dp_client import DPFlowerClient
from .utils import calculate_metrics, print_metrics, evaluate_global_model


def train_centralized_models(X_train, y_train, X_test, y_test, random_state=42):
    """Train and evaluate centralized models."""
    
    print("=== TRAINING CENTRALIZED MODELS ===")
    results = {}
    
    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
    )
    
    print(f"Training split: {len(X_train_split)} samples")
    print(f"Validation split: {len(X_val)} samples")
    
    # 1. Logistic Regression
    print("\n--- Training Logistic Regression ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(random_state=random_state, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train_split)
    
    # Evaluate LR
    lr_test_pred = lr_model.predict(X_test_scaled)
    lr_test_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_test_metrics = calculate_metrics(y_test, lr_test_pred, lr_test_proba)
    
    print_metrics(lr_test_metrics, "Logistic Regression", "Test")
    
    results['logistic_regression'] = {
        'model': lr_model,
        'scaler': scaler,
        'test_metrics': lr_test_metrics,
        'test_proba': lr_test_proba
    }
    
    # 2. Random Forest
    print("\n--- Training Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10)
    rf_model.fit(X_train_split, y_train_split)
    
    # Evaluate RF
    rf_test_pred = rf_model.predict(X_test)
    rf_test_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_test_metrics = calculate_metrics(y_test, rf_test_pred, rf_test_proba)
    
    print_metrics(rf_test_metrics, "Random Forest", "Test")
    
    results['random_forest'] = {
        'model': rf_model,
        'test_metrics': rf_test_metrics,
        'test_proba': rf_test_proba
    }
    
    return results


def run_federated_learning(hospital_A, hospital_B, test_data, num_rounds=10):
    """Run federated learning simulation."""
    
    print("=== STARTING FEDERATED LEARNING ===")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    input_size = hospital_A.shape[1] - 1  # -1 for target column
    global_model = FederatedNet(input_size).to(device)
    
    # Standardize features
    scaler = StandardScaler()
    
    # Fit scaler on combined data
    combined_features = pd.concat([
        hospital_A.iloc[:, :-1], 
        hospital_B.iloc[:, :-1]
    ], axis=0)
    scaler.fit(combined_features)
    
    # Scale hospital data
    hospital_A_scaled = hospital_A.copy()
    hospital_A_scaled.iloc[:, :-1] = scaler.transform(hospital_A.iloc[:, :-1])
    
    hospital_B_scaled = hospital_B.copy()
    hospital_B_scaled.iloc[:, :-1] = scaler.transform(hospital_B.iloc[:, :-1])
    
    # Create clients
    client_A = FlowerClient(hospital_A_scaled, global_model, device)
    client_B = FlowerClient(hospital_B_scaled, global_model, device)
    
    # Federated learning simulation
    print(f"\n--- Starting {num_rounds} rounds of federated learning ---")
    
    # Initialize results tracking
    fl_results = {
        'round': [],
        'client_A_loss': [],
        'client_A_accuracy': [],
        'client_B_loss': [],
        'client_B_accuracy': [],
        'global_test_accuracy': []
    }
    
    # Initial parameters
    global_params = client_A.get_parameters({})
    
    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1}/{num_rounds} ===")
        
        # Client training
        config = {"epochs": 3, "batch_size": 16}
        
        # Train client A
        print("Training Hospital A...")
        params_A, samples_A, _ = client_A.fit(global_params, config)
        loss_A, _, metrics_A = client_A.evaluate(params_A, {})
        
        # Train client B
        print("Training Hospital B...")
        params_B, samples_B, _ = client_B.fit(global_params, config)
        loss_B, _, metrics_B = client_B.evaluate(params_B, {})
        
        # Federated averaging (simple weighted average)
        total_samples = samples_A + samples_B
        global_params = []
        
        for param_A, param_B in zip(params_A, params_B):
            weighted_param = (param_A * samples_A + param_B * samples_B) / total_samples
            global_params.append(weighted_param)
        
        # Evaluate global model on test set
        test_accuracy = evaluate_global_model(global_model, global_params, test_data, scaler, device)
        
        # Store results
        fl_results['round'].append(round_num + 1)
        fl_results['client_A_loss'].append(loss_A)
        fl_results['client_A_accuracy'].append(metrics_A['accuracy'])
        fl_results['client_B_loss'].append(loss_B)
        fl_results['client_B_accuracy'].append(metrics_B['accuracy'])
        fl_results['global_test_accuracy'].append(test_accuracy)
        
        print(f"Client A - Loss: {loss_A:.4f}, Accuracy: {metrics_A['accuracy']:.4f}")
        print(f"Client B - Loss: {loss_B:.4f}, Accuracy: {metrics_B['accuracy']:.4f}")
        print(f"Global Test Accuracy: {test_accuracy:.4f}")
    
    return fl_results, global_model, global_params, scaler


def run_dp_federated_learning(hospital_A, hospital_B, test_data, num_rounds=10, epsilon=1.0):
    """Run federated learning with differential privacy."""
    
    print("=== STARTING DP FEDERATED LEARNING ===")
    print(f"Privacy budget per client: ε = {epsilon}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create fresh model for DP training
    input_size = hospital_A.shape[1] - 1  # -1 for target column
    global_model = FederatedNet(input_size).to(device)
    
    # Standardize features
    scaler = StandardScaler()
    combined_features = pd.concat([
        hospital_A.iloc[:, :-1], 
        hospital_B.iloc[:, :-1]
    ], axis=0)
    scaler.fit(combined_features)
    
    # Scale hospital data
    hospital_A_scaled = hospital_A.copy()
    hospital_A_scaled.iloc[:, :-1] = scaler.transform(hospital_A.iloc[:, :-1])
    
    hospital_B_scaled = hospital_B.copy()
    hospital_B_scaled.iloc[:, :-1] = scaler.transform(hospital_B.iloc[:, :-1])
    
    # Create DP clients with fresh models
    model_A = FederatedNet(input_size).to(device)
    model_B = FederatedNet(input_size).to(device)
    
    dp_client_A = DPFlowerClient(hospital_A_scaled, model_A, device, epsilon=epsilon)
    dp_client_B = DPFlowerClient(hospital_B_scaled, model_B, device, epsilon=epsilon)
    
    # Initialize results tracking
    dp_fl_results = {
        'round': [],
        'client_A_loss': [],
        'client_A_accuracy': [],
        'client_A_epsilon': [],
        'client_B_loss': [],
        'client_B_accuracy': [],
        'client_B_epsilon': [],
        'global_test_accuracy': []
    }
    
    # Initial parameters
    global_params = dp_client_A.get_parameters({})
    
    print(f"\n--- Starting {num_rounds} rounds of DP federated learning ---")
    
    for round_num in range(num_rounds):
        print(f"\n=== DP Round {round_num + 1}/{num_rounds} ===")
        
        # Client training with DP
        config = {"epochs": 3, "batch_size": 16}
        
        try:
            # Train client A with DP
            print("Training Hospital A (with DP)...")
            params_A, samples_A, metrics_A_train = dp_client_A.fit(global_params, config)
            loss_A, _, metrics_A = dp_client_A.evaluate(params_A, {})
            
            # Train client B with DP
            print("Training Hospital B (with DP)...")
            params_B, samples_B, metrics_B_train = dp_client_B.fit(global_params, config)
            loss_B, _, metrics_B = dp_client_B.evaluate(params_B, {})
            
        except Exception as e:
            print(f"Error during DP training: {e}")
            print("Continuing with available results...")
            break
        
        # Federated averaging
        total_samples = samples_A + samples_B
        global_params = []
        
        for param_A, param_B in zip(params_A, params_B):
            weighted_param = (param_A * samples_A + param_B * samples_B) / total_samples
            global_params.append(weighted_param)
        
        # Evaluate global model on test set
        test_accuracy = evaluate_global_model(global_model, global_params, test_data, scaler, device)
        
        # Store results
        dp_fl_results['round'].append(round_num + 1)
        dp_fl_results['client_A_loss'].append(loss_A)
        dp_fl_results['client_A_accuracy'].append(metrics_A['accuracy'])
        dp_fl_results['client_A_epsilon'].append(metrics_A_train.get('epsilon', 0))
        dp_fl_results['client_B_loss'].append(loss_B)
        dp_fl_results['client_B_accuracy'].append(metrics_B['accuracy'])
        dp_fl_results['client_B_epsilon'].append(metrics_B_train.get('epsilon', 0))
        dp_fl_results['global_test_accuracy'].append(test_accuracy)
        
        print(f"Client A - Loss: {loss_A:.4f}, Accuracy: {metrics_A['accuracy']:.4f}, ε: {metrics_A_train.get('epsilon', 0):.2f}")
        print(f"Client B - Loss: {loss_B:.4f}, Accuracy: {metrics_B['accuracy']:.4f}, ε: {metrics_B_train.get('epsilon', 0):.2f}")
        print(f"Global Test Accuracy: {test_accuracy:.4f}")
    
    return dp_fl_results, global_model, global_params, scaler