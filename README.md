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
- **Target variable:** Length of Stay (`LOS`)
- **Prediction setting:** early prediction using only the first *X hours* of ICU data, for example 24h or 48h
- **Main risk to avoid:** data leakage from events that happen after the prediction window

---

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

## Preprocessing Pipeline

### 1. Data Ingestion

- Place raw `.csv.gz` files in `data/raw/`
- Later, upload large raw files to Cloud Storage
- Load large tables into BigQuery when local processing becomes too slow
- Validate schemas, column names, and data types

---

### 2. Initial Filtering

For event tables, keep only records that can be linked to an ICU stay and used safely:

- valid patient/admission/stay identifiers
- valid event timestamp
- valid numeric or categorical value, depending on the table
- no events after the selected prediction window

For the first local implementation, `CHARTEVENTS` keeps:

- `SUBJECT_ID`
- `HADM_ID`
- `ICUSTAY_ID`
- `ITEMID`
- `CHARTTIME`
- `VALUENUM`

---

### 3. Data Integration

Join all useful information around the ICU stay:

- `ICUSTAYS` -> LOS, `INTIME`, `OUTTIME`
- `PATIENTS` -> demographics
- `ADMISSIONS` -> hospital admission context
- `CHARTEVENTS` -> early ICU measurements
- `LABEVENTS` -> early lab results
- `INPUTEVENTS_*` -> early treatments and administered inputs
- `OUTPUTEVENTS` -> early outputs and fluid balance
- `PRESCRIPTIONS` -> medication information available before or during the early window
- diagnosis/procedure tables -> comorbidity and clinical context, handled carefully to avoid leakage

---

### 4. Temporal Windowing

Restrict time-dependent data to the early ICU stay:

- first 24h for a simple baseline
- first 48h for a richer baseline
- possible later experiments with 6h, 12h, 24h, and 48h windows

This is essential to prevent the model from learning from future information.

---

### 5. Feature Engineering

For numeric time-series variables, aggregate per `(ICUSTAY_ID, ITEMID)`:

- mean
- min
- max
- std
- count
- last value

For other tables, possible feature groups include:

- demographics: age, gender
- admissions: admission type, ethnicity, insurance, emergency/elective status
- labs: aggregated lab values in the prediction window
- inputs: total volume, medication/input counts, rates where available
- outputs: total output, urine output summaries
- diagnoses: grouped ICD categories or comorbidity indicators
- procedures: procedure category indicators
- medications: medication classes or selected high-level drug indicators

---

### 6. Missing Data Handling

- Impute missing numeric values using training data only
- Add missing indicators where useful
- Treat categorical missingness explicitly when meaningful
- Avoid fitting imputers or encoders on the full dataset before train/test split

---

### 7. Scaling and Encoding

- Scale numeric features when needed
- Encode categorical features with train-only fitted encoders
- Keep identifiers for traceability, but do not use them as model features

---

### 8. Dataset Construction

Final dataset:

- 1 row per `ICUSTAY_ID`
- identifiers for traceability
- engineered features from selected table groups
- target variable `LOS`

---

### 9. Train/Test Split

- Split by `SUBJECT_ID`, not random rows
- Prevents the same patient appearing in both train and test
- Later experiments may use validation splits or cross-validation grouped by patient

---

### 10. Export

Save processed datasets:

- CSV for simple inspection
- Parquet for larger feature tables
- experiment-specific train/test outputs

---

## Experiment Strategy

We will not assume that using every table automatically improves the model. Instead, we will build several datasets and compare them.

### Baseline Experiments

1. `ICUSTAYS` only
2. `ICUSTAYS` + `PATIENTS` + `ADMISSIONS`
3. Baseline context + `CHARTEVENTS`
4. Baseline context + `LABEVENTS`
5. Baseline context + `CHARTEVENTS` + `LABEVENTS`
6. Add `INPUTEVENTS_*`
7. Add `OUTPUTEVENTS`
8. Add `PRESCRIPTIONS`
9. Add diagnoses and procedures
10. Full dataset with all selected table groups

### Window Experiments

Compare prediction windows:

- 6h
- 12h
- 24h
- 48h

### Feature Selection Experiments

Compare:

- top 25 `ITEMID`s
- top 50 `ITEMID`s
- top 100 `ITEMID`s
- clinically selected features
- table-specific feature groups

### Evaluation

Use consistent metrics across experiments:

- MAE
- RMSE
- R2
- median absolute error

The objective is to understand the trade-off between predictive gain, complexity, processing cost, and interpretability.

---

## Performance Considerations

- Avoid loading full large tables into memory
- Use chunked local processing for early prototypes
- Use BigQuery for heavier joins and aggregations
- Track:
  - query execution time
  - preprocessing time
  - memory usage
  - final feature count
  - model performance per table group

---

## Key Challenges

- Large dataset size
- Many heterogeneous tables
- High dimensionality from many `ITEMID`s and clinical codes
- Missing and irregular time-series data
- Preventing temporal leakage
- Preventing patient leakage across train/test
- Balancing model performance with clinical interpretability

---

## Expected Output

- Clean ML-ready datasets
- Documented preprocessing pipeline
- Experiment results comparing table groups
- Performance analysis
- Feature importance analysis
- Visualizations and interpretation

---

## Future Work

- Extend preprocessing beyond `CHARTEVENTS`
- Add table-specific feature builders
- Add model training and evaluation
- Add experiment tracking
- Add hyperparameter tuning
- Add feature importance analysis
- Add clinical interpretation of useful predictors
- Move heavy processing to BigQuery when needed

---

## Notes

- Start simple, then add table groups one by one
- Use all relevant tables, but prove their value experimentally
- Avoid data leakage even if it means discarding tempting variables
- Keep `SUBJECT_ID`, `HADM_ID`, and `ICUSTAY_ID` for traceability, but exclude them from model features

---

## Current Preprocessing Starter

This repository already includes a first local preprocessing pipeline in:

```text
src/preprocessing/preprocess.py
```

This first version currently uses:

```text
data/raw/CHARTEVENTS.csv.gz
data/raw/ICUSTAYS.csv.gz
data/raw/D_ITEMS.csv.gz
```

It is the starter pipeline, not the final full-table pipeline.

### Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
python -m src.preprocessing.preprocess --config configs/preprocessing.yaml
```

The script will:

- keep only valid numeric chart events
- restrict events to the first `window_hours` of each ICU stay
- select the top-N most frequent `ITEMID`s
- aggregate `mean`, `min`, `max`, `std`, `count`, and `last`
- create one row per `ICUSTAY_ID`
- split train/test by `SUBJECT_ID`
- impute missing values and scale features using training data only

Expected outputs:

```text
data/processed/features.csv
data/processed/train.csv
data/processed/test.csv
reports/selected_items.csv
```

### Next Implementation Checklist

- Confirm which raw MIMIC files are available locally
- Validate source columns for each table
- Run the current `CHARTEVENTS` starter pipeline
- Inspect `reports/selected_items.csv`
- Add demographic and admission features
- Add lab feature engineering
- Add input/output event feature engineering
- Add medication, diagnosis, and procedure features
- Create experiment configs for table-group comparisons
- Train baseline models and compare results
