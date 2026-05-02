# ICU Length of Stay Prediction – Data Preprocessing

## 📌 Project Overview

This project aims to analyse ICU patient data and build a machine learning pipeline to **predict Length of Stay (LOS)** based on early-stage clinical data.

The dataset is derived from the MIMIC database, with a primary focus on the `CHARTEVENTS` table, which contains time-series clinical measurements recorded during ICU stays.

This repository focuses specifically on the **data preprocessing and feature engineering pipeline**, which will later support model training and evaluation.

---

## 🎯 Objectives

- Process large-scale ICU data efficiently
- Extract meaningful features from raw event logs
- Prevent data leakage through proper temporal design
- Build a clean, structured dataset for ML models
- Ensure scalability using cloud-based tools (GCP)

---

## 📂 Data Sources

### Core Tables
- `CHARTEVENTS` → time-series clinical measurements
- `ICUSTAYS` → ICU admission information (used to define LOS)
- `D_ITEMS` → metadata for ITEMIDs (feature interpretation)

### Additional Tables (optional)
- `PATIENTS` → demographic data (age, gender)
- `ADMISSIONS` → hospital-level context

---

## 🧠 Problem Definition

- **Unit of analysis:** `ICUSTAY_ID`
- **Target variable:** Length of Stay (LOS)
- **Prediction setting:** Early prediction using only the first *X hours* (e.g. 24h)

---

## ⚙️ Tech Stack

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

## 🔄 Preprocessing Pipeline

### 1. Data Ingestion
- Upload `.csv.gz` files to Cloud Storage
- Load into BigQuery tables
- Validate schema and data types

---

### 2. Initial Filtering
- Select relevant columns:
  - `SUBJECT_ID`
  - `HADM_ID`
  - `ICUSTAY_ID`
  - `ITEMID`
  - `CHARTTIME`
  - `VALUENUM`
- Remove:
  - NULL values in `VALUENUM`
  - rows without `ICUSTAY_ID`

---

### 3. Feature Selection (ITEMIDs)
- Use `D_ITEMS` to interpret ITEMIDs
- Select:
  - vital signs
  - frequent measurements
- Optionally:
  - top-N most frequent ITEMIDs

---

### 4. Data Integration
- Join with:
  - `ICUSTAYS` → LOS + timestamps
  - optionally `PATIENTS`, `ADMISSIONS`

---

### 5. Temporal Windowing
- Restrict data to early ICU stay:
  - e.g. first 24 hours
- Prevents data leakage

---

### 6. Feature Engineering
Aggregate per `(ICUSTAY_ID, ITEMID)`:
- mean
- min
- max
- std
- count
- last value

Pivot into tabular format:

---

### 7. Missing Data Handling
- Impute values:
  - mean / median
- Add missing indicators (recommended)

---

### 8. Scaling
- Apply normalization:
  - StandardScaler or MinMaxScaler

---

### 9. Dataset Construction
Final dataset:
- 1 row per `ICUSTAY_ID`
- Features + target (LOS)

---

### 10. Train/Test Split
- Split by `SUBJECT_ID` (NOT random rows)
- Prevents leakage across admissions

---

### 11. Export
- Save processed dataset:
  - CSV or Parquet
- Share with ML pipeline

---

## ⏱️ Performance Considerations

- Avoid loading full dataset into memory
- Use BigQuery for heavy processing
- Track:
  - query execution time
  - preprocessing time
  - memory usage

---

## ⚠️ Key Challenges

- Large dataset size (~4.2GB compressed)
- High dimensionality (many ITEMIDs)
- Missing and irregular time-series data
- Preventing data leakage
- Efficient aggregation

---

## 📊 Expected Output

- Clean ML-ready dataset
- Documented preprocessing pipeline
- Performance analysis
- Visualizations (optional)

---

## 🚀 Future Work

- Model training and evaluation
- Hyperparameter tuning
- Feature importance analysis
- Clinical interpretation

---

## 📌 Notes

- Prioritize simplicity and scalability
- Avoid unnecessary tables and joins
- Focus on meaningful features

---

## 🧪 Preprocessing Starter

This repository includes a first local preprocessing pipeline in:

```text
src/preprocessing/preprocess.py
```

### Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place the raw MIMIC-like files in `data/raw/`:

```text
data/raw/CHARTEVENTS.csv.gz
data/raw/ICUSTAYS.csv.gz
data/raw/D_ITEMS.csv.gz
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

### Your Preprocessing Checklist

- Confirm the source files and column names match the expected MIMIC format.
- Decide the prediction window, for example 24h or 48h.
- Decide `top_n_items`, starting with 50 for a simple baseline.
- Inspect `reports/selected_items.csv` to verify the selected clinical measurements.
- Keep `SUBJECT_ID`, `HADM_ID`, and `ICUSTAY_ID` in the output for traceability, but do not use them as model features.
