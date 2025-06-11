# DVC Demo: Breast Cancer Classification Pipeline

This project demonstrates the capabilities of **Data Version Control (DVC)** using the Wisconsin Breast Cancer Diagnostic dataset. It showcases a complete machine learning pipeline with data versioning, experiment tracking, and reproducible workflows.

## 🎯 Project Overview

The pipeline predicts breast cancer diagnosis (Benign/Malignant) using cell nucleus characteristics from fine needle aspirate (FNA) images. The dataset contains 30 numerical features computed from digitized images.

### Dataset Information
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- **Samples**: 569 instances
- **Features**: 30 real-valued features (mean, standard error, and worst values for 10 characteristics)
- **Target**: Binary classification (M = Malignant, B = Benign)
- **Distribution**: 357 benign, 212 malignant cases

## 🏗️ Project Structure

```
dvc-demo/
├── dvc_demo/
│   ├── data/
│   │   ├── raw/
│   │   │   ├── data.csv              # Original dataset
│   │   │   └── data.csv.dvc          # DVC tracking file
│   │   ├── processed/                # Train/test splits (generated)
│   │   │   ├── train.csv             # Training data
│   │   │   └── test.csv              # Test data
│   │   └── model/                    # Model artifacts and evaluation outputs
│   │       ├── random_forest_model.joblib  # Trained model
│   │       ├── scaler.joblib         # Feature scaler
│   │       ├── feature_importance.csv      # Feature importance rankings
│   │       ├── training_info.json    # Training metadata
│   │       └── evaluation/           # Model evaluation outputs
│   │           ├── confusion_matrix.png    # Confusion matrix visualization
│   │           ├── roc_curve.png     # ROC curve analysis
│   │           ├── feature_importance.png  # Feature importance plot
│   │           ├── prediction_analysis.png # Prediction distribution analysis
│   │           ├── feature_importance.csv  # Feature importance rankings
│   │           └── metrics.json      # Performance metrics
│   ├── code/                         # Source code
│   │   ├── split_data.py             # Data splitting script
│   │   ├── train_model.py            # Model training script
│   │   └── evaluate_model.py         # Model evaluation script
│   ├── dvc.yaml                      # DVC pipeline definition
│   ├── params.yaml                   # Pipeline parameters configuration
│   └── dvc.lock                      # DVC lock file (auto-generated)
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── .gitignore                        # Git ignore patterns
├── .dvcignore                        # DVC ignore patterns
└── .dvc/                             # DVC configuration
```

## 🚀 DVC Pipeline Stages

### Stage 1: Data Splitting (`split_data`)
- **Input**: Original dataset (`dvc_demo/data/raw/data.csv`)
- **Output**: Train/test splits with stratification
- **Configuration**: Controlled via `params.yaml`
- **Features**:
  - 80/20 train/test split (configurable)
  - Stratified sampling to maintain class balance
  - Reproducible with fixed random seed

### Stage 2: Model Training (`train_model`)
- **Input**: Training data (`dvc_demo/data/processed/train.csv`)
- **Output**: Trained Random Forest model and related artifacts
- **Configuration**: Hyperparameters defined in `params.yaml`
- **Generated Files**:
  - `random_forest_model.joblib`: Trained Random Forest model
  - `scaler.joblib`: Feature scaler (StandardScaler)
  - `feature_importance.csv`: Feature importance rankings
  - `training_info.json`: Training metadata and configuration

### Stage 3: Model Evaluation (`evaluate_model`)
- **Input**: Trained model, scaler, and test data
- **Output**: Comprehensive evaluation metrics and visualizations
- **Configuration**: Evaluation parameters in `params.yaml`
- **Generated Files**:
  - `metrics.json`: Performance metrics (accuracy, precision, recall, F1, ROC-AUC)
  - `confusion_matrix.png`: Confusion matrix visualization
  - `roc_curve.png`: ROC curve analysis
  - `feature_importance.png`: Feature importance plot
  - `prediction_analysis.png`: Prediction distribution analysis
  - `feature_importance.csv`: Feature importance rankings

## ⚙️ Configuration Management

The pipeline uses a centralized `params.yaml` file for parameter management.

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.8+
- Git

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd dvc-demo
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize DVC** (if not already done):
   ```bash
   dvc init
   ```

## 🎮 Running the Pipeline

### Execute the Complete Pipeline
```bash
cd dvc_demo
dvc exp run
```

This command will:
1. Check for changes in dependencies and parameters
2. Run only the stages that need to be updated
3. Generate all outputs in the correct order

### Run Individual Stages
```bash
# Run only data splitting
dvc exp run split_data

# Run only model training
dvc exp run train_model

# Run only model evaluation
dvc exp run evaluate_model
```

### Force Re-run All Stages
```bash
dvc exp run --force
```

## 📊 Viewing Results

### Check Pipeline Status
```bash
dvc status
```

### View Metrics
```bash
dvc exp show
```

### Compare Experiments (after parameter changes)
```bash
dvc metrics diff
```

### Visualize Pipeline DAG
```bash
dvc dag
```

## 🔧 Experimenting with Parameters

You can easily experiment with different parameters by modifying the `params.yaml` file:

```yaml
train_model:
  n_estimators: 200     # Increase number of trees
  max_depth: 10         # Limit tree depth
  min_samples_split: 5  # Require more samples to split
```

After modifying parameters, simply run:
```bash
dvc exp run -n new_experiment
```

DVC will automatically detect the parameter changes and re-run the affected stages.

### Alternative: Command-line Experiments
```bash
# Run an experiment with different parameters
dvc exp run -S train_model.n_estimators=200 -S train_model.max_depth=10
```

## 📈 Key DVC Features Demonstrated

### 1. **Data Versioning**
- Original dataset is tracked with `dvc add`
- Changes to data are automatically detected
- Data integrity is maintained with checksums

### 2. **Pipeline Management**
- Declarative pipeline definition in `dvc.yaml`
- Automatic dependency tracking
- Incremental execution (only runs changed stages)

### 3. **Parameter Management**
- Centralized parameter configuration in `params.yaml`
- Automatic detection of parameter changes
- Easy experimentation and comparison

### 4. **Experiment Tracking**
- Metrics are automatically tracked in `metrics.json`
- Easy comparison between experiments
- Reproducible results with version control

### 5. **Artifacts Management**
- Models, plots, and evaluation results are versioned
- Large files are efficiently stored
- Easy sharing and collaboration

### 6. **Reproducibility**
- Fixed random seeds ensure reproducible results
- Dependencies are explicitly tracked
- Environment captured in `requirements.txt`
- Parameters version-controlled in `params.yaml`

## 🎯 Expected Results

A successful pipeline run should achieve:
- **Accuracy**: ~95-97%
- **ROC-AUC**: ~0.98-0.99
- **Precision/Recall**: High performance for both classes

Key insights from the analysis:
- **Most important features**: Usually radius, perimeter, and area measurements
- **Feature correlations**: Strong correlations between size-related features
- **Class separation**: Clear distinction between benign and malignant cases

## 🔄 Version Control Integration

### Track Changes with Git
```bash
# Add DVC files to git
git add dvc_demo/dvc.yaml dvc_demo/params.yaml dvc_demo/data/raw/data.csv.dvc requirements.txt README.md

# Commit the pipeline
git commit -m "Add breast cancer classification pipeline"
```

### Share the Project
```bash
# Push to remote repository
git push origin main

# Others can reproduce by:
git clone <repository-url>
cd dvc-demo
pip install -r requirements.txt
cd dvc_demo
dvc exp run
```

## 📚 Learn More

- [DVC Documentation](https://dvc.org/doc)
- [DVC Tutorials](https://dvc.org/doc/start)
- [Data Science Pipeline Best Practices](https://dvc.org/doc/use-cases)

## 📝 License

This project is for educational purposes and demonstrates DVC capabilities using the publicly available Wisconsin Breast Cancer dataset.

---

**Built with ❤️ using DVC - Making ML experiments reproducible and shareable!**