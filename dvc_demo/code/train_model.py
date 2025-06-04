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


def train_model(train_path, test_path, output_dir, n_estimators=100, max_depth=None, random_state=42):
    """
    Train a Random Forest model and evaluate its performance.
    
    Args:
        train_path (str): Path to training data CSV
        test_path (str): Path to test data CSV
        output_dir (str): Directory to save model and results
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of the trees
        random_state (int): Random state for reproducibility
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
    feature_cols = [col for col in train_df.columns if col not in ['patient_id', 'diagnosis']]
    
    X_train = train_df[feature_cols]
    y_train = train_df['diagnosis']
    X_test = test_df[feature_cols]
    y_test = test_df['diagnosis']
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    # Train Random Forest model
    print(f"Training Random Forest with {n_estimators} estimators, max_depth={max_depth}")
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
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
    metrics['test_precision'] = precision_score(y_test, y_test_pred, pos_label='M')
    metrics['test_recall'] = recall_score(y_test, y_test_pred, pos_label='M')
    metrics['test_f1'] = f1_score(y_test, y_test_pred, pos_label='M')
    metrics['test_roc_auc'] = roc_auc_score(y_test == 'M', y_test_proba)
    
    # Model parameters
    metrics['model_params'] = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'random_state': random_state
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
    
    # Generate detailed classification report
    class_report = classification_report(y_test, y_test_pred, output_dict=True)
    class_report_path = os.path.join(output_dir, 'classification_report.json')
    with open(class_report_path, 'w') as f:
        json.dump(class_report, f, indent=2)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Visualization
    plt.style.use('seaborn-v0_8')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'], 
                yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test == 'M', y_test_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
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
    
    # Plot top 20 feature importances
    plt.figure(figsize=(12, 8))
    top_20_features = feature_importance.head(20)
    plt.barh(range(len(top_20_features)), top_20_features['importance'])
    plt.yticks(range(len(top_20_features)), top_20_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Prediction distribution
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Prediction probabilities
    plt.subplot(1, 2, 1)
    plt.hist(y_test_proba[y_test == 'B'], bins=30, alpha=0.7, label='Benign', color='blue')
    plt.hist(y_test_proba[y_test == 'M'], bins=30, alpha=0.7, label='Malignant', color='red')
    plt.xlabel('Prediction Probability (Malignant)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Probabilities')
    plt.legend()
    
    # Subplot 2: Model confidence
    plt.subplot(1, 2, 2)
    confidence = np.max(rf_model.predict_proba(X_test_scaled), axis=1)
    plt.hist(confidence, bins=30, alpha=0.7, color='green')
    plt.xlabel('Model Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Model Confidence')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nModel training completed!")
    print(f"Files saved to {output_dir}:")
    print(f"  - random_forest_model.joblib")
    print(f"  - scaler.joblib")
    print(f"  - metrics.json")
    print(f"  - classification_report.json")
    print(f"  - feature_importance.csv")
    print(f"  - confusion_matrix.png")
    print(f"  - roc_curve.png")
    print(f"  - feature_importance.png")
    print(f"  - prediction_analysis.png")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest model")
    parser.add_argument("--train", required=True, help="Training data CSV file path")
    parser.add_argument("--test", required=True, help="Test data CSV file path")
    parser.add_argument("--output", required=True, help="Output directory for model and results")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum tree depth")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    
    args = parser.parse_args()
    
    train_model(
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    ) 