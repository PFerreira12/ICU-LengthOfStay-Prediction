"""
Training and evaluation pipeline for ICU Length of Stay prediction.

Main responsibilities of this stage:
1. Load prepared train/test datasets
2. Separate features and target (LOS)
3. Remove traceability identifiers from model inputs
4. Define baseline regression models
5. Perform grouped cross-validation using SUBJECT_ID
6. Train models and evaluate performance
7. Compare experiments across different feature sets
8. Save metrics, predictions, and trained models

Expected inputs:
- data/processed/train.csv
- data/processed/test.csv

Expected outputs:
- reports/metrics/
- reports/predictions/
- models/

Important considerations:
- Prevent patient leakage with grouped CV
- Use the same CV strategy across experiments
- Keep experiments reproducible
- Compare models fairly using the same splits
"""

# ============================================================
# Imports
# ============================================================

# ============================================================
# Configuration
# ============================================================

# ============================================================
# Load Preparation Pipeline
# ============================================================

# Load train/test datasets
# Load the fitted preprocessing pipeline saved during
# the preparation stage:
# models/preprocessor.joblib

# This guarantees:
# - same imputers
# - same scaling
# - same categorical encoding
# - consistent feature transformations

# The preprocessing pipeline must NEVER be refitted on test data.

# ============================================================
# Feature / Target Separation
# ============================================================

# Remove identifiers:
# SUBJECT_ID
# HADM_ID
# ICUSTAY_ID

# Keep LOS as target variable

# ============================================================
# Cross-Validation Strategy
# ============================================================

# Use grouped CV to prevent patient leakage
# Recommended:
# - GroupKFold
# - groups = SUBJECT_ID

# ============================================================
# Baseline Models
# ============================================================

# Possible starting models:
# - Linear Regression
# - Random Forest Regressor
# - XGBoost / LightGBM (later)

# ============================================================
# Training
# ============================================================

# Train models on training folds only

# ============================================================
# Evaluation
# ============================================================

# Suggested metrics:
# - MAE
# - RMSE
# - R2
# - Median Absolute Error

# ============================================================
# Experiment Tracking
# ============================================================

# Store:
# - model name
# - feature set
# - window size
# - metrics
# - runtime

# ============================================================
# Saving Outputs
# ============================================================

# Save:
# - trained models
# - metrics CSVs
# - predictions