#!/usr/bin/env python3
"""
Model training script for breast cancer classification.
Trains a Random Forest classifier and evaluates performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
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
                use_feature_scaling=True, top_features_to_plot=20, viz_config=None):
    """
    Train a Random Forest model and evaluate its performance.
    
    Args:
        train_path (str): Path to training data CSV
        test_path (str): Path to test data CSV
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
        top_features_to_plot (int): Number of top features to show in importance plot
        viz_config (dict): Visualization configuration
    """
    if viz_config is None:
        viz_config = {}
    
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
    X_test = test_df[feature_cols]
    y_test = test_df['diagnosis']
    
    # Feature scaling
    if use_feature_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the scaler
        scaler_path = os.path.join(output_dir, 'scaler.joblib')
        joblib.dump(scaler, scaler_path)
        print(f"Feature scaling applied and scaler saved")
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values
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
    
    # Make predictions
    y_train_pred = rf_model.predict(X_train_scaled)
    y_test_pred = rf_model.predict(X_test_scaled)
    y_test_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = {}
    
    # Training metrics
    metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
    
    # Test metrics
    metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)
    metrics['test_precision'] = precision_score(y_test, y_test_pred, pos_label=pos_label)
    metrics['test_recall'] = recall_score(y_test, y_test_pred, pos_label=pos_label)
    metrics['test_f1'] = f1_score(y_test, y_test_pred, pos_label=pos_label)
    metrics['test_roc_auc'] = roc_auc_score(y_test == pos_label, y_test_proba)
    
    # Model parameters
    metrics['model_params'] = {
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
    }
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model Metrics:")
    print(f"  Training Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  Test Precision: {metrics['test_precision']:.4f}")
    print(f"  Test Recall: {metrics['test_recall']:.4f}")
    print(f"  Test F1-Score: {metrics['test_f1']:.4f}")
    print(f"  Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Visualization parameters from config
    cm_config = viz_config.get('confusion_matrix', {})
    roc_config = viz_config.get('roc_curve', {})
    fi_config = viz_config.get('feature_importance', {})
    pa_config = viz_config.get('prediction_analysis', {})
    
    # Set default matplotlib style
    plt.style.use('seaborn-v0_8')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    figsize = cm_config.get('figsize', [8, 6])
    cmap = cm_config.get('cmap', 'Blues')
    class_labels = cm_config.get('class_labels', ['Benign', 'Malignant'])
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=class_labels, 
                yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test == pos_label, y_test_proba)
    figsize = roc_config.get('figsize', [8, 6])
    line_color = roc_config.get('line_color', 'darkorange')
    line_width = roc_config.get('line_width', 2)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color=line_color, lw=line_width, 
             label=f'ROC curve (AUC = {metrics["test_roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # Plot top N feature importances
    figsize = fi_config.get('figsize', [12, 8])
    
    plt.figure(figsize=figsize)
    top_features = feature_importance.head(top_features_to_plot)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_features_to_plot} Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Prediction distribution
    figsize = pa_config.get('figsize', [12, 5])
    hist_bins = pa_config.get('hist_bins', 30)
    hist_alpha = pa_config.get('hist_alpha', 0.7)
    
    plt.figure(figsize=figsize)
    
    # Subplot 1: Prediction probabilities
    plt.subplot(1, 2, 1)
    plt.hist(y_test_proba[y_test == 'B'], bins=hist_bins, alpha=hist_alpha, label='Benign', color='blue')
    plt.hist(y_test_proba[y_test == pos_label], bins=hist_bins, alpha=hist_alpha, label='Malignant', color='red')
    plt.xlabel(f'Prediction Probability ({pos_label})')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Probabilities')
    plt.legend()
    
    # Subplot 2: Model confidence
    plt.subplot(1, 2, 2)
    confidence = np.max(rf_model.predict_proba(X_test_scaled), axis=1)
    plt.hist(confidence, bins=hist_bins, alpha=hist_alpha, color='green')
    plt.xlabel('Model Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Model Confidence')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nModel training completed!")
    print(f"Files saved to {output_dir}:")
    print(f"  - random_forest_model.joblib")
    if use_feature_scaling:
        print(f"  - scaler.joblib")
    print(f"  - metrics.json")
    print(f"  - feature_importance.csv")
    print(f"  - confusion_matrix.png")
    print(f"  - roc_curve.png")
    print(f"  - feature_importance.png")
    print(f"  - prediction_analysis.png")
    
    return metrics


if __name__ == "__main__":
    # Load parameters from params.yaml
    params = load_params()
    train_params = params.get('train_model', {})
    viz_config = train_params.get('visualization', {})
    
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
    top_features_to_plot = train_params.get('top_features_to_plot', 20)
    
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
    print(f"  top_features_to_plot: {top_features_to_plot}")
    
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
        use_feature_scaling=use_feature_scaling,
        top_features_to_plot=top_features_to_plot,
        viz_config=viz_config
    ) 