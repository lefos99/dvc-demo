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
│   │   ├── data.csv              # Original dataset
│   │   ├── data.csv.dvc          # DVC tracking file
│   │   └── processed/            # Train/test splits (generated)
│   ├── code/                     # Source code
│   │   ├── split_data.py         # Data splitting script
│   │   ├── data_statistics.py    # Statistical analysis script
│   │   └── train_model.py        # Model training script
│   ├── reports/                  # Generated reports (created by pipeline)
│   │   └── statistics/           # Data analysis outputs
│   └── models/                   # Trained models (created by pipeline)
├── dvc.yaml                      # DVC pipeline definition
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .dvc/                         # DVC configuration
```

## 🚀 DVC Pipeline Stages

### Stage 1: Data Splitting (`split_data`)
- **Input**: Original dataset (`dvc_demo/data/data.csv`)
- **Output**: Train/test splits with stratification
- **Features**:
  - 80/20 train/test split
  - Stratified sampling to maintain class balance
  - Reproducible with fixed random seed

### Stage 2: Data Statistics (`data_statistics`)
- **Input**: Training data
- **Output**: Comprehensive statistical analysis and visualizations
- **Generated Files**:
  - `statistics.json`: Numerical statistics
  - `class_distribution.png`: Class balance visualization
  - `correlation_heatmap.png`: Feature correlation analysis
  - `feature_distributions.png`: Feature distributions by diagnosis
  - `pairplot_key_features.png`: Pairwise feature relationships
  - `feature_variance.png`: Feature variance ranking
  - `summary_statistics.csv`: Statistical summary by class

### Stage 3: Model Training (`train_model`)
- **Input**: Train/test splits
- **Output**: Trained Random Forest model and evaluation metrics
- **Generated Files**:
  - `random_forest_model.joblib`: Trained model
  - `scaler.joblib`: Feature scaler
  - `metrics.json`: Performance metrics
  - `feature_importance.csv`: Feature importance rankings
  - `confusion_matrix.png`: Confusion matrix visualization
  - `roc_curve.png`: ROC curve analysis
  - `feature_importance.png`: Feature importance plot
  - `prediction_analysis.png`: Prediction distribution analysis

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

2. **Create a virtual environment** (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate
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
dvc repro
```

This command will:
1. Check for changes in dependencies
2. Run only the stages that need to be updated
3. Generate all outputs in the correct order

### Run Individual Stages
```bash
# Run only data splitting
dvc repro split_data

# Run only statistics generation
dvc repro data_statistics

# Run only model training
dvc repro train_model
```

### Force Re-run All Stages
```bash
dvc repro --force
```

## 📊 Viewing Results

### Check Pipeline Status
```bash
dvc status
```

### View Metrics
```bash
dvc metrics show
```

### Compare Experiments (if you modify parameters)
```bash
dvc metrics diff
```

### Visualize Pipeline
```bash
dvc dag
```

## 🔧 Experimenting with Parameters

You can easily experiment with different model parameters by modifying the `dvc.yaml` file:

```yaml
train_model:
  cmd: python dvc_demo/code/train_model.py --train ... --n-estimators 200 --max-depth 10
```

Or run experiments directly:
```bash
dvc exp run -S train_model.cmd="python dvc_demo/code/train_model.py --train dvc_demo/data/processed/train.csv --test dvc_demo/data/processed/test.csv --output dvc_demo/models --n-estimators 200 --max-depth 10 --random-state 42"
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

### 3. **Experiment Tracking**
- Metrics are automatically tracked
- Easy comparison between experiments
- Reproducible results with version control

### 4. **Artifacts Management**
- Models, plots, and reports are versioned
- Large files are efficiently stored
- Easy sharing and collaboration

### 5. **Reproducibility**
- Fixed random seeds ensure reproducible results
- Dependencies are explicitly tracked
- Environment is captured in requirements.txt

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
git add dvc.yaml dvc_demo/data/data.csv.dvc requirements.txt README.md

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
dvc repro
```

## 🤝 Contributing

To extend this demo:

1. **Add new stages**: Modify `dvc.yaml` to include preprocessing, feature selection, or model evaluation stages
2. **Try different algorithms**: Modify `train_model.py` to use SVM, XGBoost, or neural networks
3. **Add hyperparameter tuning**: Integrate grid search or Bayesian optimization
4. **Include model validation**: Add cross-validation and statistical testing

## 📚 Learn More

- [DVC Documentation](https://dvc.org/doc)
- [DVC Tutorials](https://dvc.org/doc/start)
- [Data Science Pipeline Best Practices](https://dvc.org/doc/use-cases)

## 📝 License

This project is for educational purposes and demonstrates DVC capabilities using the publicly available Wisconsin Breast Cancer dataset.

---

**Built with ❤️ using DVC - Making ML experiments reproducible and shareable!**