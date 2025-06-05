#!/usr/bin/env python3
"""
Data statistics and visualization script for breast cancer dataset.
Generates comprehensive statistics and plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json
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


def generate_statistics(train_path, output_dir):
    """
    Generate comprehensive statistics and visualizations for the training data.
    
    Args:
        train_path (str): Path to the training CSV file
        output_dir (str): Directory to save statistics and plots
    """
    # Load parameters
    params = load_params()
    stats_params = params.get('data_statistics', {})
    viz_params = stats_params.get('visualization', {})
    
    # Configuration from params.yaml
    important_features = stats_params.get('important_features', [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
        'smoothness_mean', 'compactness_mean', 'concavity_mean', 
        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
    ])
    
    pairplot_features = stats_params.get('pairplot_features', [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean'
    ])
    
    top_variance_features = stats_params.get('top_variance_features', 20)
    
    # Visualization settings
    fig_size_large = viz_params.get('figure_size_large', [20, 16])
    fig_size_medium = viz_params.get('figure_size_medium', [12, 10])
    fig_size_small = viz_params.get('figure_size_small', [12, 8])
    dpi = viz_params.get('dpi', 300)
    style = viz_params.get('style', 'seaborn-v0_8')
    
    print(f"Using configuration:")
    print(f"  Important features: {len(important_features)} features")
    print(f"  Pairplot features: {len(pairplot_features)} features")
    print(f"  Top variance features: {top_variance_features}")
    print(f"  Plot style: {style}")
    print(f"  DPI: {dpi}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the training data
    print(f"Loading training data from {train_path}")
    df = pd.read_csv(train_path)
    
    print(f"Training data shape: {df.shape}")
    
    # Basic statistics
    stats = {}
    
    # Class distribution
    class_dist = df['diagnosis'].value_counts()
    stats['class_distribution'] = class_dist.to_dict()
    stats['class_percentages'] = (class_dist / len(df) * 100).to_dict()
    
    # Feature statistics
    feature_cols = [col for col in df.columns if col not in ['id', 'diagnosis']]
    feature_stats = df[feature_cols].describe()
    
    # Save detailed statistics to JSON
    stats['feature_statistics'] = feature_stats.to_dict()
    stats['total_samples'] = len(df)
    stats['total_features'] = len(feature_cols)
    stats['configuration'] = {
        'important_features': important_features,
        'pairplot_features': pairplot_features,
        'top_variance_features': top_variance_features
    }
    
    with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to {output_dir}/statistics.json")
    
    # Set style for plots
    plt.style.use(style)
    sns.set_palette("husl")
    
    # 1. Class distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    sns.countplot(data=df, x='diagnosis', ax=ax1)
    ax1.set_title('Class Distribution')
    ax1.set_xlabel('Diagnosis')
    ax1.set_ylabel('Count')
    
    # Pie chart
    ax2.pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Class Distribution (Percentage)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 2. Feature correlation heatmap
    plt.figure(figsize=fig_size_large)
    correlation_matrix = df[feature_cols].corr()
    
    # Create heatmap with hierarchical clustering
    sns.clustermap(correlation_matrix, 
                   annot=False, 
                   cmap='coolwarm', 
                   center=0,
                   figsize=fig_size_large,
                   dendrogram_ratio=0.1)
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 3. Feature distributions by diagnosis
    # Use configurable important features
    available_features = [f for f in important_features if f in df.columns]
    if not available_features:
        available_features = feature_cols[:10]  # Fallback to first 10 features
    
    # Adjust plot layout based on number of features
    n_features = len(available_features)
    n_cols = min(5, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 5*n_rows))
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
    
    for i, feature in enumerate(available_features):
        sns.boxplot(data=df, x='diagnosis', y=feature, ax=axes[i])
        axes[i].set_title(f'{feature}')
        axes[i].set_xlabel('Diagnosis')
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 4. Pairplot for key features
    pairplot_features_available = [f for f in pairplot_features if f in df.columns]
    if pairplot_features_available:
        key_features = pairplot_features_available + ['diagnosis']
        
        plt.figure(figsize=fig_size_medium)
        pairplot = sns.pairplot(df[key_features], hue='diagnosis', diag_kind='hist')
        pairplot.savefig(os.path.join(output_dir, 'pairplot_key_features.png'), dpi=dpi, bbox_inches='tight')
        plt.close()
    
    # 5. Feature importance through variance
    feature_variance = df[feature_cols].var().sort_values(ascending=False)
    
    plt.figure(figsize=fig_size_small)
    top_var_features = feature_variance.head(top_variance_features)
    plt.barh(range(len(top_var_features)), top_var_features.values)
    plt.yticks(range(len(top_var_features)), top_var_features.index)
    plt.xlabel('Variance')
    plt.title(f'Top {top_variance_features} Features by Variance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_variance.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 6. Missing values analysis
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        plt.figure(figsize=fig_size_small)
        missing_data[missing_data > 0].plot(kind='bar')
        plt.title('Missing Values by Feature')
        plt.xlabel('Features')
        plt.ylabel('Missing Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'missing_values.png'), dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        print("No missing values found in the dataset")
    
    # 7. Statistical summary table
    summary_features = available_features if available_features else important_features
    available_summary_features = [f for f in summary_features if f in df.columns]
    
    if available_summary_features:
        summary_stats = df.groupby('diagnosis')[available_summary_features].agg(['mean', 'std', 'min', 'max'])
        summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
    
    print(f"Generated plots and statistics in {output_dir}")
    print(f"Files created:")
    print(f"  - statistics.json")
    print(f"  - class_distribution.png")
    print(f"  - correlation_heatmap.png")
    print(f"  - feature_distributions.png")
    print(f"  - pairplot_key_features.png")
    print(f"  - feature_variance.png")
    print(f"  - summary_statistics.csv")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data statistics and plots")
    parser.add_argument("--train", required=True, help="Training data CSV file path")
    parser.add_argument("--output", required=True, help="Output directory for statistics and plots")
    
    args = parser.parse_args()
    
    generate_statistics(
        train_path=args.train,
        output_dir=args.output
    ) 