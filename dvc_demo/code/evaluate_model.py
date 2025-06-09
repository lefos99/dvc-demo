#!/usr/bin/env python3
"""
Model evaluation script for breast cancer classification.
Evaluates a trained Random Forest classifier and generates visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
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


def evaluate_model(model_path, scaler_path, test_path, output_dir, pos_label='M',
                  top_features_to_plot=20, viz_config=None):
    """
    Evaluate a trained model and generate performance metrics and visualizations.
    
    Args:
        model_path (str): Path to the trained model file
        scaler_path (str): Path to the scaler file (can be None)
        test_path (str): Path to test data CSV
        output_dir (str): Directory to save evaluation results
        pos_label (str): Positive class label for binary classification metrics
        top_features_to_plot (int): Number of top features to show in importance plot
        viz_config (dict): Visualization configuration
    """
    if viz_config is None:
        viz_config = {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Load scaler if provided
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        print(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
    
    # Load test data
    print(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Test data shape: {test_df.shape}")
    
    # Prepare features and targets
    feature_cols = [col for col in test_df.columns if col not in ['id', 'diagnosis']]
    
    X_test = test_df[feature_cols]
    y_test = test_df['diagnosis']
    
    # Apply scaling if scaler is available
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        print("Feature scaling applied")
    else:
        X_test_scaled = X_test.values
        print("No feature scaling applied")
    
    # Make predictions
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = {}
    
    # Test metrics
    metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)
    metrics['test_precision'] = precision_score(y_test, y_test_pred, pos_label=pos_label)
    metrics['test_recall'] = recall_score(y_test, y_test_pred, pos_label=pos_label)
    metrics['test_f1'] = f1_score(y_test, y_test_pred, pos_label=pos_label)
    metrics['test_roc_auc'] = roc_auc_score(y_test == pos_label, y_test_proba)
    
    # Model evaluation parameters
    metrics['evaluation_params'] = {
        'pos_label': pos_label,
        'top_features_to_plot': top_features_to_plot
    }
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model Evaluation Metrics:")
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
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
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
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    
    # 3. Feature Importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
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
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
        plt.close()
    else:
        print("Model does not support feature importances")
    
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
    confidence = np.max(model.predict_proba(X_test_scaled), axis=1)
    plt.hist(confidence, bins=hist_bins, alpha=hist_alpha, color='green')
    plt.xlabel('Model Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Model Confidence')
    
    plt.savefig(os.path.join(output_dir, 'prediction_analysis.png'), dpi=300)
    plt.close()
    
    print(f"\nModel evaluation completed!")
    print(f"Files saved to {output_dir}:")
    print(f"  - metrics.json")
    print(f"  - confusion_matrix.png")
    print(f"  - roc_curve.png")
    if hasattr(model, 'feature_importances_'):
        print(f"  - feature_importance.csv")
        print(f"  - feature_importance.png")
    print(f"  - prediction_analysis.png")
    
    return metrics


if __name__ == "__main__":
    # Load parameters from params.yaml
    params = load_params()
    eval_params = params.get('evaluate_model', {})
    viz_config = eval_params.get('visualization', {})
    
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", required=True, help="Trained model file path")
    parser.add_argument("--scaler", help="Scaler file path (optional)")
    parser.add_argument("--test", required=True, help="Test data CSV file path")
    parser.add_argument("--output", required=True, help="Output directory for evaluation results")
    
    args = parser.parse_args()
    
    # Get parameters from params.yaml
    pos_label = eval_params.get('pos_label', 'M')
    top_features_to_plot = eval_params.get('top_features_to_plot', 20)
    
    print(f"Using parameters from params.yaml:")
    print(f"  pos_label: {pos_label}")
    print(f"  top_features_to_plot: {top_features_to_plot}")
    
    evaluate_model(
        model_path=args.model,
        scaler_path=args.scaler,
        test_path=args.test,
        output_dir=args.output,
        pos_label=pos_label,
        top_features_to_plot=top_features_to_plot,
        viz_config=viz_config
    ) 