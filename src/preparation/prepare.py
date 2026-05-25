import time
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import argparse
import joblib
import logging; logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from src.preprocessing.preprocess import read_config

@dataclass
class PrepareConfig:
    input_features: Path
    output_train: Path
    output_test: Path
    test_size: float = 0.2
    random_state: int = 42


def split_impute_scale(features: pd.DataFrame, config: PrepareConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the engineered ICU dataset into train/test sets grouped by SUBJECT_ID,
    then apply train-only preprocessing for machine learning preparation.

    The preparation pipeline:
    - separates identifiers and target variable
    - detects numeric and categorical features
    - imputes missing values
    - scales numeric variables
    - one-hot encodes categorical variables
    - prevents patient leakage through grouped splitting
    - saves the fitted preprocessing pipeline for reproducibility
    """

    # Traceability identifiers kept outside model features
    id_cols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]

    # Regression target: ICU Length of Stay
    target_col = "LOS"

    # Remove identifiers and target from feature matrix
    feature_cols = [col for col in features.columns if col not in id_cols + [target_col]]

    # Detect categorical columns for one-hot encoding
    categorical_cols = features[feature_cols].select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    # Remaining columns are treated as numeric
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    logging.info(f"Detected {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features")

    # Split by SUBJECT_ID to prevent the same patient
    # from appearing in both train and test sets
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    train_idx, test_idx = next(
        splitter.split(features, groups=features["SUBJECT_ID"])
    )

    train = features.iloc[train_idx].copy()
    test = features.iloc[test_idx].copy()

    logging.info(f"Split completed: train={len(train)} rows, test={len(test)} rows")

    transformers = []

    # Numeric preprocessing:
    # - median imputation
    # - missing indicators
    # - standard scaling
    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        )

        transformers.append(("num", numeric_pipeline, numeric_cols))

    # Categorical preprocessing:
    # - most frequent imputation
    # - one-hot encoding
    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        transformers.append(("cat", categorical_pipeline, categorical_cols))

    # Combine numeric and categorical pipelines
    preprocessor = ColumnTransformer(transformers=transformers)

    logging.info("Fitting preparation pipeline...")

    # Fit preprocessing ONLY on training data
    # to avoid data leakage
    train_values = preprocessor.fit_transform(train[feature_cols])

    # Apply learned transformations to test data
    test_values = preprocessor.transform(test[feature_cols])

    # Save fitted preprocessing pipeline for reproducibility
    Path("models").mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, "models/preprocessor.joblib")

    logging.info("Saved fitted preparation pipeline to models/preprocessor.joblib")

    # Recover transformed feature names after scaling/encoding
    transformed_cols = [
        col.replace("num__", "").replace("cat__", "")
        for col in preprocessor.get_feature_names_out()
    ]

    # Final ML-ready training dataset
    train_out = pd.concat(
        [
            train[id_cols + [target_col]].reset_index(drop=True),
            pd.DataFrame(train_values, columns=transformed_cols),
        ],
        axis=1,
    )

    # Final ML-ready testing dataset
    test_out = pd.concat(
        [
            test[id_cols + [target_col]].reset_index(drop=True),
            pd.DataFrame(test_values, columns=transformed_cols),
        ],
        axis=1,
    )

    logging.info("Transformation completed successfully")

    return train_out, test_out


def build_config(args: argparse.Namespace) -> PrepareConfig:
    """
    Build the preparation configuration using CLI arguments
    and fallback YAML configuration values.
    """
    
    raw = read_config(args.config) if args.config else {}
    output = raw.get("output", {})

    return PrepareConfig(
        input_features=Path(args.input_features or output.get("features", "data/processed/features_24h.csv")),

        output_train=Path(args.output_train or output.get("train", "data/processed/train.csv")),

        output_test=Path(args.output_test or output.get("test", "data/processed/test.csv")),

        test_size=args.test_size or raw.get("test_size", 0.2),

        random_state=args.random_state or raw.get("random_state", 42),
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ICU LOS preparation features from MIMIC-like CSV files.")

    """
    Parse command-line arguments for the preparation pipeline.
    """
    
    parser.add_argument("--config", type=Path, default=Path("configs/preprocessing.yaml"))
    parser.add_argument("--input-features", type=Path)
    parser.add_argument("--output-train", type=Path)
    parser.add_argument("--output-test", type=Path)
    parser.add_argument("--test-size", type=float)
    parser.add_argument("--random-state", type=int)

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the ICU LOS preparation pipeline.

    Loads engineered ICU features, performs grouped train/test splitting,
    applies preprocessing transformations, and exports ML-ready datasets.
    """
     
    logging.info("Starting preparation pipeline")
    start_time = time.time()

    config = build_config(parse_args())

    logging.info(f"Loading data from {config.input_features}")
    features = pd.read_csv(config.input_features)
    
    logging.info("Train-Test split, imputation and scaling in progress...")
    train, test = split_impute_scale(features, config)
    
    train.to_csv(config.output_train, index=False)
    test.to_csv(config.output_test, index=False)

    end_time = time.time()
    
    logging.info(f"Train saved: {train.shape[0]} rows -> {config.output_train}")
    logging.info(f"Test saved: {test.shape[0]} rows -> {config.output_test}")
    logging.info(f"Preparation pipeline finished successfully in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
