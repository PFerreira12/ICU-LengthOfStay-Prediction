# ICU Length of Stay Prediction - Data Preprocessing

## Project Overview

This project aims to analyse ICU patient data and build a machine learning pipeline to **predict Length of Stay (LOS)** using early-stage clinical information.

The dataset is based on MIMIC-style ICU data. The first implemented version focuses on `CHARTEVENTS`, `ICUSTAYS`, and `D_ITEMS`, but the project direction is to use the full set of relevant clinical tables and then run controlled experiments to understand which tables actually improve prediction performance.

The repository currently focuses on the **data preprocessing and feature engineering pipeline**, which will support model training, evaluation, and feature comparison.

---

## Objectives

- Process large-scale ICU data efficiently
- Extract meaningful features from multiple clinical data sources
- Prevent data leakage through proper temporal design
- Build clean, structured datasets for ML models
- Compare different table groups through experiments
- Identify whether extra clinical tables improve LOS prediction
- Ensure scalability using cloud-based tools such as GCP and BigQuery

---

## Data Sources

The goal is to use all relevant MIMIC-style tables discussed for LOS prediction, grouped by clinical meaning.

### ICU Stay and Patient Context

- `ICUSTAYS` -> ICU admission/discharge information, ICU timestamps, and LOS target
- `PATIENTS` -> patient-level demographics, such as age and gender
- `ADMISSIONS` -> hospital admission context, admission type, insurance, ethnicity, admission/discharge times
- `TRANSFERS` -> patient movement across hospital units, if available

### Time-Series ICU Measurements

- `CHARTEVENTS` -> bedside charted measurements, vital signs, nursing observations
- `D_ITEMS` -> metadata for `ITEMID`s used in `CHARTEVENTS`, `INPUTEVENTS`, and related ICU tables

### Laboratory Data

- `LABEVENTS` -> laboratory measurements
- `D_LABITEMS` -> metadata for laboratory item IDs

### Treatments, Medication, and Fluid Balance

- `INPUTEVENTS_CV` / `INPUTEVENTS_MV` -> fluids, medications, and other inputs, depending on MIMIC version
- `OUTPUTEVENTS` -> urine output and other outputs
- `PRESCRIPTIONS` -> medication orders

### Diagnoses and Procedures

- `DIAGNOSES_ICD` -> diagnosis codes
- `PROCEDURES_ICD` -> procedure codes
- `D_ICD_DIAGNOSES` -> diagnosis code descriptions
- `D_ICD_PROCEDURES` -> procedure code descriptions

### Optional Clinical Events

- `MICROBIOLOGYEVENTS` -> cultures and microbiology results
- `CPTEVENTS` -> billing/procedure events, if useful
- `SERVICES` -> hospital service transfers, if available

Not every table is guaranteed to improve the model. The plan is to include them carefully and compare their impact through experiments.

---

## Problem Definition
- **Unit of analysis:** `ICUSTAY_ID`
- **Target:** `Length of Stay (LOS)`
- **Prediction setting:** early prediction (first 24h ICU data)
- **Critical constraint:** strict prevention of temporal + patient leakage

## Tech Stack

- **Google Cloud Platform (GCP)**
  - Cloud Storage
  - BigQuery
- **Python**
  - pandas / numpy
  - scikit-learn
- **VS Code + Codex**
- Optional:
  - PySpark / Dataproc
  - Vertex AI

---

## 1. Preprocessing Pipeline (BigQuery / Heavy ETL)
- Raw MIMIC tables (330M+ rows)
- Filtering temporal (first 24h ICU window)
- Feature aggregation (ITEMIDs, labs, inputs, etc.)
- Output: 1 row per ICUSTAY_ID

**Output:**
> data/processed/features_24h.csv

---

## 2. Preparation (Scikit-learn ML pipeline)
- Train/test split (by SUBJECT_ID)
- Missing value handling
- Scaling + encoding
- Fit-only-on-train transformation
- Export ML-ready datasets
- Save fitted preprocessing pipeline

**Output:**
> data/processed/train.csv

> data/processed/test.csv

> models/preprocessor.joblib

---

## 3. Training & Experimentation (Future phase)
- ML models
- Cross-validation (GroupKFold)
- Hyperparameter tuning
- Evaluation & interpretation

---

## PREPROCESSING (BigQuery + Feature Engineering)

**Goal:**
Transform raw ICU event tables into a structured ML dataset.

**Key idea:**
Convert millions of clinical events → 1 structured row per ICU stay. Steps:

---

**1. Data ingestion**
- Load raw CSVs or BigQuery tables
- Store in GCS / local staging

---

**2. Temporal filtering**

Only keep events such that:
> INTIME ≤ event_time ≤ INTIME + 24h

- Prevents future leakage
- Simulates real ICU prediction scenario

---

**3. Feature aggregation**

For each (ICUSTAY_ID, ITEMID):
- mean
- min
- max
- std
- count
- last value

---

**4. Context features**

Add static features:
- age
- gender
- admission type
- ethnicity
- insurance
- care unit info

---

**5. Output dataset**

Final structure:
- 1 row per ICUSTAY_ID
- LOS target included
- no post-discharge information

Output artifacts:
> icu_los.features_24h

> icu_los.quality_report_24h

---

## PREPARATION

**Goal:**
Convert engineered dataset into ML-ready format.

**Responsibilities:**

**1. Train/test split (patient-safe)**

Split by SUBJECT_ID (it prevents same patient in train and test)

---

**2. Feature separation**

- ID columns → excluded from features
- LOS → target
- remaining → features

---

**3. Type detection**

Automatically detect:
- numeric columns
- categorical columns

---

**4. Numeric pipeline**

Median imputation
+ missing indicators
+ standard scaling

---

**5. Categorical pipeline**

Most frequent imputation
+ one-hot encoding

---

**6. Combined preprocessing**

ColumnTransformer = numeric + categorical pipelines

---

**7. Fit only on training data**

It prevents data leakage
- fit(train)
- transform(train/test)

---

**8. Persist preprocessing pipeline**

- models/preprocessor.joblib

Used later for:
- training consistency
- inference
- reproducibility

---

**9. Output datasets**

> data/processed/train.csv

> data/processed/test.csv

---

## Key guarantee

This stage ensures:
- no patient leakage
- no feature leakage
- reproducible transformations
- ML-ready structured dataset

---

## EXPERIMENTATION PLAN (NEXT PHASE)

**Baseline experiments**

Progressive feature inclusion:
- ICUSTAYS only
- PATIENTS + ADMISSIONS
- CHARTEVENTS
- LABEVENTS
- CHARTEVENTS + LABEVENTS
- INPUTEVENTS
- OUTPUTEVENTS
- PRESCRIPTIONS
- DIAGNOSES + PROCEDURES
- Full feature set

---

**Window experiments**
- 6h
- 12h
- 24h
- 48h

---

**Feature selection experiments**
- top 25 `ITEMIDs`
- top 50 `ITEMIDs`
- top 100 `ITEMIDs`
- clinically selected features
- table-specific feature groups

---

**Models**
- Linear Regression (baseline)
- Random Forest
- Gradient Boosting / XGBoost

---

**CV strategy**

It prevents patient leakage in validation
- GroupKFold (by SUBJECT_ID)

---

**Metrics**
- MAE
- RMSE
- R²
- Median Absolute Error

The objective is to understand the trade-off between predictive gain, complexity, processing cost, and interpretability.

---

## PERFORMANCE TRACKING

**Track:**
- preprocessing time
- feature count
- model training time
- memory usage
- BigQuery execution time

---

## FUTURE WORK

- expand beyond CHARTEVENTS
- feature importance analysis
- SHAP explanations
- hyperparameter tuning
- experiment tracking (MLflow)
- deployable inference pipeline