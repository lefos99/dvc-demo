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


def generate_statistics(train_path, output_dir):
    """
    Generate comprehensive statistics and visualizations for the training data.
    
    Args:
        train_path (str): Path to the training CSV file
        output_dir (str): Directory to save statistics and plots
    """
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
    feature_cols = [col for col in df.columns if col not in ['patient_id', 'diagnosis']]
    feature_stats = df[feature_cols].describe()
    
    # Save detailed statistics to JSON
    stats['feature_statistics'] = feature_stats.to_dict()
    stats['total_samples'] = len(df)
    stats['total_features'] = len(feature_cols)
    
    with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to {output_dir}/statistics.json")
    
    # Set style for plots
    plt.style.use('seaborn-v0_8')
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
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature correlation heatmap
    plt.figure(figsize=(20, 16))
    correlation_matrix = df[feature_cols].corr()
    
    # Create heatmap with hierarchical clustering
    sns.clustermap(correlation_matrix, 
                   annot=False, 
                   cmap='coolwarm', 
                   center=0,
                   figsize=(20, 16),
                   dendrogram_ratio=0.1)
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature distributions by diagnosis
    # Select a subset of important features for visualization
    important_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                         'smoothness_mean', 'compactness_mean', 'concavity_mean', 
                         'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(important_features):
        sns.boxplot(data=df, x='diagnosis', y=feature, ax=axes[i])
        axes[i].set_title(f'{feature}')
        axes[i].set_xlabel('Diagnosis')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Pairplot for key features
    key_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'diagnosis']
    
    plt.figure(figsize=(12, 10))
    pairplot = sns.pairplot(df[key_features], hue='diagnosis', diag_kind='hist')
    pairplot.savefig(os.path.join(output_dir, 'pairplot_key_features.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Feature importance through variance
    feature_variance = df[feature_cols].var().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    top_20_var = feature_variance.head(20)
    plt.barh(range(len(top_20_var)), top_20_var.values)
    plt.yticks(range(len(top_20_var)), top_20_var.index)
    plt.xlabel('Variance')
    plt.title('Top 20 Features by Variance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_variance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Missing values analysis
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        plt.figure(figsize=(12, 6))
        missing_data[missing_data > 0].plot(kind='bar')
        plt.title('Missing Values by Feature')
        plt.xlabel('Features')
        plt.ylabel('Missing Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'missing_values.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("No missing values found in the dataset")
    
    # 7. Statistical summary table
    summary_stats = df.groupby('diagnosis')[important_features].agg(['mean', 'std', 'min', 'max'])
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