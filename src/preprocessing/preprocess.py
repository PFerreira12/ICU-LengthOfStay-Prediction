from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


CHART_COLS = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM"]
ICU_COLS = [
    "SUBJECT_ID",
    "HADM_ID",
    "ICUSTAY_ID",
    "DBSOURCE",
    "FIRST_CAREUNIT",
    "LAST_CAREUNIT",
    "INTIME",
    "OUTTIME",
    "LOS",
]
PATIENT_COLS = ["SUBJECT_ID", "GENDER", "DOB"]
ADMISSION_COLS = [
    "SUBJECT_ID",
    "HADM_ID",
    "ADMISSION_TYPE",
    "ADMISSION_LOCATION",
    "INSURANCE",
    "LANGUAGE",
    "RELIGION",
    "MARITAL_STATUS",
    "ETHNICITY",
    "EDREGTIME",
    "EDOUTTIME",
    "HAS_CHARTEVENTS_DATA",
]


@dataclass
class PreprocessConfig:
    chartevents: Path
    icustays: Path
    d_items: Path | None
    patients: Path | None
    admissions: Path | None
    output_features: Path
    output_train: Path
    output_test: Path
    output_item_summary: Path
    output_quality_report: Path
    window_hours: int = 24
    top_n_items: int = 50
    test_size: float = 0.2
    random_state: int = 42
    chunksize: int = 500_000


@dataclass
class CharteventsQuality:
    valid_numeric_events: int = 0
    linked_events: int = 0
    events_before_intime: int = 0
    events_in_window: int = 0
    events_after_window: int = 0


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [column.strip().upper() for column in df.columns]
    return df


def read_csv_columns(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    header = normalize_columns(pd.read_csv(path, nrows=0))
    available = [column for column in columns if column in header.columns]
    return normalize_columns(pd.read_csv(path, usecols=available))


def load_icustays(path: Path) -> pd.DataFrame:
    icu = read_csv_columns(path, ICU_COLS)
    required = {"SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME"}
    missing = required - set(icu.columns)
    if missing:
        raise ValueError(f"ICUSTAYS is missing required columns: {sorted(missing)}")

    icu["INTIME"] = pd.to_datetime(icu["INTIME"], errors="coerce")
    if "OUTTIME" in icu.columns:
        icu["OUTTIME"] = pd.to_datetime(icu["OUTTIME"], errors="coerce")

    if "LOS" not in icu.columns:
        if "OUTTIME" not in icu.columns:
            raise ValueError("ICUSTAYS needs either LOS or OUTTIME to build the target.")
        icu["LOS"] = (icu["OUTTIME"] - icu["INTIME"]).dt.total_seconds() / 86_400

    icu = icu.dropna(subset=["ICUSTAY_ID", "INTIME", "LOS"])
    icu["ICUSTAY_ID"] = icu["ICUSTAY_ID"].astype("int64")
    output_cols = [
        col
        for col in [
            "SUBJECT_ID",
            "HADM_ID",
            "ICUSTAY_ID",
            "DBSOURCE",
            "FIRST_CAREUNIT",
            "LAST_CAREUNIT",
            "INTIME",
            "LOS",
        ]
        if col in icu.columns
    ]
    return icu[output_cols]


def clean_categorical(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper().replace({"": pd.NA}).fillna("UNKNOWN")


def load_patients(path: Path) -> pd.DataFrame:
    patients = read_csv_columns(path, PATIENT_COLS)
    required = {"SUBJECT_ID", "GENDER", "DOB"}
    missing = required - set(patients.columns)
    if missing:
        raise ValueError(f"PATIENTS is missing required columns: {sorted(missing)}")

    patients["DOB"] = pd.to_datetime(patients["DOB"], errors="coerce")
    patients["GENDER"] = clean_categorical(patients["GENDER"])
    return patients[["SUBJECT_ID", "GENDER", "DOB"]]


def load_admissions(path: Path) -> pd.DataFrame:
    admissions = read_csv_columns(path, ADMISSION_COLS)
    required = {"SUBJECT_ID", "HADM_ID"}
    missing = required - set(admissions.columns)
    if missing:
        raise ValueError(f"ADMISSIONS is missing required columns: {sorted(missing)}")

    for col in ["EDREGTIME", "EDOUTTIME"]:
        if col in admissions.columns:
            admissions[col] = pd.to_datetime(admissions[col], errors="coerce")

    categorical_cols = [
        "ADMISSION_TYPE",
        "ADMISSION_LOCATION",
        "INSURANCE",
        "LANGUAGE",
        "RELIGION",
        "MARITAL_STATUS",
        "ETHNICITY",
    ]
    for col in categorical_cols:
        if col in admissions.columns:
            admissions[col] = clean_categorical(admissions[col])

    return admissions


def add_context_features(config: PreprocessConfig, icu: pd.DataFrame) -> pd.DataFrame:
    features = icu.copy()

    if config.patients and config.patients.exists():
        patients = load_patients(config.patients)
        features = features.merge(patients, on="SUBJECT_ID", how="left")
        age = features["INTIME"].dt.year - features["DOB"].dt.year
        birthday_not_reached = (
            (features["INTIME"].dt.month < features["DOB"].dt.month)
            | (
                (features["INTIME"].dt.month == features["DOB"].dt.month)
                & (features["INTIME"].dt.day < features["DOB"].dt.day)
            )
        )
        features["AGE"] = age - birthday_not_reached.astype("int64")
        features.loc[features["AGE"] > 120, "AGE"] = 90
        features.loc[features["AGE"] < 0, "AGE"] = np.nan
        features = features.drop(columns=["DOB"])

    if config.admissions and config.admissions.exists():
        admissions = load_admissions(config.admissions)
        features = features.merge(admissions, on=["SUBJECT_ID", "HADM_ID"], how="left")
        if {"EDREGTIME", "EDOUTTIME"}.issubset(features.columns):
            features["ED_LOS_HOURS"] = (
                features["EDOUTTIME"] - features["EDREGTIME"]
            ).dt.total_seconds() / 3_600
            features = features.drop(columns=["EDREGTIME", "EDOUTTIME"])

    for col in ["DBSOURCE", "FIRST_CAREUNIT", "LAST_CAREUNIT"]:
        if col in features.columns:
            features[col] = clean_categorical(features[col])

    return features.drop(columns=["INTIME"])


def iter_chartevents(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    header = normalize_columns(pd.read_csv(path, nrows=0))
    missing = set(CHART_COLS) - set(header.columns)
    if missing:
        raise ValueError(f"CHARTEVENTS is missing required columns: {sorted(missing)}")

    for chunk in pd.read_csv(path, usecols=CHART_COLS, chunksize=chunksize):
        chunk = normalize_columns(chunk)
        chunk = chunk.dropna(subset=["ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM"])
        chunk["ICUSTAY_ID"] = chunk["ICUSTAY_ID"].astype("int64")
        chunk["ITEMID"] = chunk["ITEMID"].astype("int64")
        chunk["CHARTTIME"] = pd.to_datetime(chunk["CHARTTIME"], errors="coerce")
        chunk["VALUENUM"] = pd.to_numeric(chunk["VALUENUM"], errors="coerce")
        yield chunk.dropna(subset=["CHARTTIME", "VALUENUM"])


def merge_with_icu_time(events: pd.DataFrame, icu: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    merged = events.merge(
        icu[["ICUSTAY_ID", "INTIME"]],
        on="ICUSTAY_ID",
        how="inner",
    )
    merged["WINDOW_END"] = merged["INTIME"] + pd.to_timedelta(window_hours, unit="h")
    return merged


def apply_temporal_window(events: pd.DataFrame, icu: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    merged = merge_with_icu_time(events, icu, window_hours)
    return merged[(merged["CHARTTIME"] >= merged["INTIME"]) & (merged["CHARTTIME"] <= merged["WINDOW_END"])]


def select_top_items(config: PreprocessConfig, icu: pd.DataFrame) -> tuple[list[int], CharteventsQuality]:
    counts: dict[int, int] = {}
    quality = CharteventsQuality()
    for chunk in iter_chartevents(config.chartevents, config.chunksize):
        quality.valid_numeric_events += len(chunk)
        merged = merge_with_icu_time(chunk, icu, config.window_hours)
        quality.linked_events += len(merged)
        before_mask = merged["CHARTTIME"] < merged["INTIME"]
        window_mask = (merged["CHARTTIME"] >= merged["INTIME"]) & (merged["CHARTTIME"] <= merged["WINDOW_END"])
        after_mask = merged["CHARTTIME"] > merged["WINDOW_END"]
        quality.events_before_intime += int(before_mask.sum())
        quality.events_in_window += int(window_mask.sum())
        quality.events_after_window += int(after_mask.sum())
        windowed = merged[window_mask]
        for item_id, count in windowed["ITEMID"].value_counts().items():
            counts[int(item_id)] = counts.get(int(item_id), 0) + int(count)

    ranked = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
    return [item_id for item_id, _ in ranked[: config.top_n_items]], quality


def aggregate_features(config: PreprocessConfig, icu: pd.DataFrame, selected_items: set[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in iter_chartevents(config.chartevents, config.chunksize):
        windowed = apply_temporal_window(chunk, icu, config.window_hours)
        windowed = windowed[windowed["ITEMID"].isin(selected_items)]
        if windowed.empty:
            continue

        windowed = windowed.sort_values("CHARTTIME")
        grouped = windowed.groupby(["ICUSTAY_ID", "ITEMID"])["VALUENUM"]
        agg = grouped.agg(["mean", "min", "max", "std", "count"]).reset_index()
        last = windowed.groupby(["ICUSTAY_ID", "ITEMID"]).tail(1)[["ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM"]]
        last = last.rename(columns={"CHARTTIME": "last_time", "VALUENUM": "last"})
        frames.append(agg.merge(last, on=["ICUSTAY_ID", "ITEMID"], how="left"))

    if not frames:
        raise ValueError("No CHARTEVENTS rows remained after filtering and temporal windowing.")

    partial = pd.concat(frames, ignore_index=True)
    partial["sum"] = partial["mean"] * partial["count"]
    partial["sum_sq"] = (partial["std"].fillna(0) ** 2) * (partial["count"] - 1) + partial["count"] * (partial["mean"] ** 2)

    stats = partial.groupby(["ICUSTAY_ID", "ITEMID"]).agg(
        count=("count", "sum"),
        sum=("sum", "sum"),
        sum_sq=("sum_sq", "sum"),
        min=("min", "min"),
        max=("max", "max"),
    ).reset_index()

    last_values = partial.sort_values("last_time").groupby(["ICUSTAY_ID", "ITEMID"]).tail(1)
    last_values = last_values[["ICUSTAY_ID", "ITEMID", "last"]]

    combined = stats.merge(last_values, on=["ICUSTAY_ID", "ITEMID"], how="left")
    combined["mean"] = combined["sum"] / combined["count"]
    variance = (combined["sum_sq"] - (combined["sum"] ** 2 / combined["count"])) / (combined["count"] - 1)
    combined["std"] = np.sqrt(np.maximum(variance, 0)).replace([np.inf, -np.inf], np.nan)
    combined.loc[combined["count"] <= 1, "std"] = np.nan

    long_features = combined[["ICUSTAY_ID", "ITEMID", "mean", "min", "max", "std", "count", "last"]]
    wide = long_features.pivot(index="ICUSTAY_ID", columns="ITEMID")
    wide.columns = [f"item_{item_id}_{stat}" for stat, item_id in wide.columns]
    wide = wide.reset_index()

    context = add_context_features(config, icu)
    return context.merge(wide, on="ICUSTAY_ID", how="inner")


def add_item_summary(config: PreprocessConfig, selected_items: list[int]) -> None:
    summary = pd.DataFrame({"ITEMID": selected_items})
    if config.d_items and config.d_items.exists():
        d_items = normalize_columns(pd.read_csv(config.d_items))
        cols = [col for col in ["ITEMID", "LABEL", "CATEGORY", "UNITNAME"] if col in d_items.columns]
        summary = summary.merge(d_items[cols], on="ITEMID", how="left")

    config.output_item_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.output_item_summary, index=False)


def write_quality_report(
    config: PreprocessConfig,
    icu: pd.DataFrame,
    features: pd.DataFrame,
    quality: CharteventsQuality,
    selected_items: list[int],
) -> None:
    id_cols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]
    target_col = "LOS"
    feature_cols = [col for col in features.columns if col not in id_cols + [target_col]]
    rows = [
        ("window_hours", config.window_hours),
        ("top_n_items", config.top_n_items),
        ("selected_items", len(selected_items)),
        ("icustays_after_target_filter", len(icu)),
        ("patients_after_target_filter", icu["SUBJECT_ID"].nunique()),
        ("preprocessed_rows", len(features)),
        ("preprocessed_columns", features.shape[1]),
        ("model_feature_columns", len(feature_cols)),
        ("valid_numeric_chartevents", quality.valid_numeric_events),
        ("chartevents_linked_to_icustay", quality.linked_events),
        ("chartevents_before_intime", quality.events_before_intime),
        ("chartevents_in_prediction_window", quality.events_in_window),
        ("chartevents_after_prediction_window", quality.events_after_window),
        ("icustays_with_chart_features", features["ICUSTAY_ID"].nunique()),
        ("icustays_without_chart_features", len(icu) - features["ICUSTAY_ID"].nunique()),
    ]

    missing_rates = features[feature_cols].isna().mean().sort_values(ascending=False)
    for col, rate in missing_rates.head(25).items():
        rows.append((f"missing_rate_top25.{col}", round(float(rate), 6)))

    report = pd.DataFrame(rows, columns=["metric", "value"])
    config.output_quality_report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.output_quality_report, index=False)


def split_impute_scale(features: pd.DataFrame, config: PreprocessConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_cols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]
    target_col = "LOS"
    feature_cols = [col for col in features.columns if col not in id_cols + [target_col]]
    categorical_cols = features[feature_cols].select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    train_idx, test_idx = next(splitter.split(features, groups=features["SUBJECT_ID"]))
    train = features.iloc[train_idx].copy()
    test = features.iloc[test_idx].copy()

    transformers = []
    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    train_values = preprocessor.fit_transform(train[feature_cols])
    test_values = preprocessor.transform(test[feature_cols])
    transformed_cols = [
        col.replace("num__", "").replace("cat__", "")
        for col in preprocessor.get_feature_names_out()
    ]

    train_out = pd.concat(
        [
            train[id_cols + [target_col]].reset_index(drop=True),
            pd.DataFrame(train_values, columns=transformed_cols),
        ],
        axis=1,
    )
    test_out = pd.concat(
        [
            test[id_cols + [target_col]].reset_index(drop=True),
            pd.DataFrame(test_values, columns=transformed_cols),
        ],
        axis=1,
    )
    return train_out, test_out


def build_config(args: argparse.Namespace) -> PreprocessConfig:
    raw = read_config(args.config)
    return PreprocessConfig(
        chartevents=Path(args.chartevents or raw["input"]["chartevents"]),
        icustays=Path(args.icustays or raw["input"]["icustays"]),
        d_items=Path(args.d_items or raw["input"].get("d_items")) if (args.d_items or raw["input"].get("d_items")) else None,
        patients=Path(args.patients or raw["input"].get("patients")) if (args.patients or raw["input"].get("patients")) else None,
        admissions=Path(args.admissions or raw["input"].get("admissions")) if (args.admissions or raw["input"].get("admissions")) else None,
        output_features=Path(args.output_features or raw["output"]["features"]),
        output_train=Path(args.output_train or raw["output"]["train"]),
        output_test=Path(args.output_test or raw["output"]["test"]),
        output_item_summary=Path(args.output_item_summary or raw["output"]["item_summary"]),
        output_quality_report=Path(args.output_quality_report or raw["output"]["quality_report"]),
        window_hours=args.window_hours or raw.get("window_hours", 24),
        top_n_items=args.top_n_items or raw.get("top_n_items", 50),
        test_size=args.test_size or raw.get("test_size", 0.2),
        random_state=args.random_state or raw.get("random_state", 42),
        chunksize=args.chunksize,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ICU LOS preprocessing features from MIMIC-like CSV files.")
    parser.add_argument("--config", type=Path, default=Path("configs/preprocessing.yaml"))
    parser.add_argument("--chartevents", type=Path)
    parser.add_argument("--icustays", type=Path)
    parser.add_argument("--d-items", type=Path)
    parser.add_argument("--patients", type=Path)
    parser.add_argument("--admissions", type=Path)
    parser.add_argument("--output-features", type=Path)
    parser.add_argument("--output-train", type=Path)
    parser.add_argument("--output-test", type=Path)
    parser.add_argument("--output-item-summary", type=Path)
    parser.add_argument("--output-quality-report", type=Path)
    parser.add_argument("--window-hours", type=int)
    parser.add_argument("--top-n-items", type=int)
    parser.add_argument("--test-size", type=float)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--chunksize", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    config = build_config(parse_args())
    for path in [config.output_features, config.output_train, config.output_test, config.output_quality_report]:
        path.parent.mkdir(parents=True, exist_ok=True)

    icu = load_icustays(config.icustays)
    selected_items, quality = select_top_items(config, icu)
    if not selected_items:
        raise ValueError("Could not select ITEMIDs. Check CHARTEVENTS, ICUSTAYS and the temporal window.")

    add_item_summary(config, selected_items)
    features = aggregate_features(config, icu, set(selected_items))
    features.to_csv(config.output_features, index=False)
    write_quality_report(config, icu, features, quality, selected_items)

    train, test = split_impute_scale(features, config)
    train.to_csv(config.output_train, index=False)
    test.to_csv(config.output_test, index=False)

    print(f"Selected {len(selected_items)} ITEMIDs.")
    print(f"Feature table: {features.shape[0]} rows x {features.shape[1]} columns")
    print(f"Train: {train.shape[0]} rows | Test: {test.shape[0]} rows")
    print(f"Quality report: {config.output_quality_report}")


if __name__ == "__main__":
    main()
