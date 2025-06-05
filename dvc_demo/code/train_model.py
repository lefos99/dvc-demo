#!/usr/bin/env python3
"""
Model training script for breast cancer classification.
Trains a Random Forest classifier without evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import argparse
import os
import json
import joblib
import yaml


def load_params():
    """Load parameters from params.yaml file."""
    try:
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        return params
    except FileNotFoundError:
        print("Warning: params.yaml not found. Using default parameters.")
        return {}


def train_model(train_path, test_path, output_dir, n_estimators=100, max_depth=None, 
                min_samples_split=2, min_samples_leaf=1, max_features='sqrt', 
                bootstrap=True, random_state=42, n_jobs=-1, pos_label='M',
                use_feature_scaling=True):
    """
    Train a Random Forest model.
    
    Args:
        train_path (str): Path to training data CSV
        test_path (str): Path to test data CSV (used for feature consistency)
        output_dir (str): Directory to save model and results
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of the trees
        min_samples_split (int): Minimum samples required to split an internal node
        min_samples_leaf (int): Minimum samples required to be at a leaf node
        max_features (str/int): Number of features to consider for best split
        bootstrap (bool): Whether bootstrap samples are used when building trees
        random_state (int): Random state for reproducibility
        n_jobs (int): Number of jobs to run in parallel
        pos_label (str): Positive class label for binary classification metrics
        use_feature_scaling (bool): Whether to apply StandardScaler
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Prepare features and targets
    feature_cols = [col for col in train_df.columns if col not in ['id', 'diagnosis']]
    
    X_train = train_df[feature_cols]
    y_train = train_df['diagnosis']
    X_test = test_df[feature_cols]  # Only for feature consistency check
    
    # Feature scaling
    if use_feature_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Save the scaler
        scaler_path = os.path.join(output_dir, 'scaler.joblib')
        joblib.dump(scaler, scaler_path)
        print(f"Feature scaling applied and scaler saved")
    else:
        X_train_scaled = X_train.values
        print(f"Feature scaling skipped")
    
    # Convert max_features if it's a string representing None
    if max_features == 'None' or max_features == 'null':
        max_features = None
    
    # Train Random Forest model
    print(f"Training Random Forest with the following parameters:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  min_samples_split: {min_samples_split}")
    print(f"  min_samples_leaf: {min_samples_leaf}")
    print(f"  max_features: {max_features}")
    print(f"  bootstrap: {bootstrap}")
    print(f"  random_state: {random_state}")
    print(f"  n_jobs: {n_jobs}")
    
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Save the model
    model_path = os.path.join(output_dir, 'random_forest_model.joblib')
    joblib.dump(rf_model, model_path)
    
    # Calculate training accuracy for basic validation
    y_train_pred = rf_model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Save basic training info
    training_info = {
        'train_accuracy': train_accuracy,
        'model_params': {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': str(max_features),
            'bootstrap': bootstrap,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'pos_label': pos_label,
            'use_feature_scaling': use_feature_scaling
        },
        'data_info': {
            'n_features': len(feature_cols),
            'n_train_samples': len(train_df),
            'feature_names': feature_cols
        }
    }
    
    # Save feature importance
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance_path = os.path.join(output_dir, 'feature_importance.csv')
    feature_importance_df.to_csv(feature_importance_path, index=False)
    
    # Save training info
    training_info_path = os.path.join(output_dir, 'training_info.json')
    with open(training_info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"Model Training Completed!")
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Number of Features: {len(feature_cols)}")
    print(f"  Training Samples: {len(train_df)}")
    
    print(f"\nFiles saved to {output_dir}:")
    print(f"  - random_forest_model.joblib")
    if use_feature_scaling:
        print(f"  - scaler.joblib")
    print(f"  - feature_importance.csv")
    print(f"  - training_info.json")
    
    return training_info


if __name__ == "__main__":
    # Load parameters from params.yaml
    params = load_params()
    train_params = params.get('train_model', {})
    
    parser = argparse.ArgumentParser(description="Train Random Forest model")
    parser.add_argument("--train", required=True, help="Training data CSV file path")
    parser.add_argument("--test", required=True, help="Test data CSV file path")
    parser.add_argument("--output", required=True, help="Output directory for model and results")
    
    args = parser.parse_args()
    
    # Get all parameters from params.yaml
    n_estimators = train_params.get('n_estimators', 100)
    max_depth = train_params.get('max_depth', None)
    min_samples_split = train_params.get('min_samples_split', 2)
    min_samples_leaf = train_params.get('min_samples_leaf', 1)
    max_features = train_params.get('max_features', 'sqrt')
    bootstrap = train_params.get('bootstrap', True)
    random_state = train_params.get('random_state', 42)
    n_jobs = train_params.get('n_jobs', -1)
    pos_label = train_params.get('pos_label', 'M')
    use_feature_scaling = train_params.get('use_feature_scaling', True)
    
    print(f"Using parameters from params.yaml:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  min_samples_split: {min_samples_split}")
    print(f"  min_samples_leaf: {min_samples_leaf}")
    print(f"  max_features: {max_features}")
    print(f"  bootstrap: {bootstrap}")
    print(f"  random_state: {random_state}")
    print(f"  n_jobs: {n_jobs}")
    print(f"  pos_label: {pos_label}")
    print(f"  use_feature_scaling: {use_feature_scaling}")
    
    train_model(
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=random_state,
        n_jobs=n_jobs,
        pos_label=pos_label,
        use_feature_scaling=use_feature_scaling
    ) 