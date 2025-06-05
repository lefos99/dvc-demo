#!/usr/bin/env python3
"""
Data splitting script for breast cancer dataset.
Splits the data into train/test sets with stratification.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os
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


def split_data(input_path, output_dir, test_size=0.2, random_state=42):
    """
    Split the dataset into train and test sets.
    
    Args:
        input_path (str): Path to the input CSV file
        output_dir (str): Directory to save the split datasets
        test_size (float): Proportion of data for testing
        random_state (int): Random state for reproducibility
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Diagnosis distribution:\n{df['diagnosis'].value_counts()}")
    
    # Separate features and target
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']
    
    # Split the data with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    # Create train and test dataframes
    train_df = pd.concat([
        df.loc[X_train.index, ['id']],
        y_train,
        X_train
    ], axis=1)
    
    test_df = pd.concat([
        df.loc[X_test.index, ['id']],
        y_test,
        X_test
    ], axis=1)
    
    # Save the splits
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train set saved to {train_path}: {train_df.shape}")
    print(f"Test set saved to {test_path}: {test_df.shape}")
    
    print(f"Train diagnosis distribution:\n{train_df['diagnosis'].value_counts()}")
    print(f"Test diagnosis distribution:\n{test_df['diagnosis'].value_counts()}")


if __name__ == "__main__":
    # Load parameters from params.yaml
    params = load_params()
    split_params = params.get('split_data', {})
    
    parser = argparse.ArgumentParser(description="Split dataset into train/test")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output directory for splits")
    parser.add_argument("--test-size", type=float, 
                       default=split_params.get('test_size', 0.2), 
                       help="Test set proportion")
    parser.add_argument("--random-state", type=int, 
                       default=split_params.get('random_state', 42), 
                       help="Random state")
    
    args = parser.parse_args()
    
    print(f"Using parameters:")
    print(f"  Test size: {args.test_size}")
    print(f"  Random state: {args.random_state}")
    
    split_data(
        input_path=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.random_state
    ) 