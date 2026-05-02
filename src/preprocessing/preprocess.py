from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


CHART_COLS = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM"]
ICU_COLS = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "LOS"]


@dataclass
class PreprocessConfig:
    chartevents: Path
    icustays: Path
    d_items: Path | None
    output_features: Path
    output_train: Path
    output_test: Path
    output_item_summary: Path
    window_hours: int = 24
    top_n_items: int = 50
    test_size: float = 0.2
    random_state: int = 42
    chunksize: int = 500_000


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
    return icu[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "LOS"]]


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


def apply_temporal_window(events: pd.DataFrame, icu: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    merged = events.merge(
        icu[["ICUSTAY_ID", "INTIME"]],
        on="ICUSTAY_ID",
        how="inner",
    )
    max_time = merged["INTIME"] + pd.to_timedelta(window_hours, unit="h")
    return merged[(merged["CHARTTIME"] >= merged["INTIME"]) & (merged["CHARTTIME"] <= max_time)]


def select_top_items(config: PreprocessConfig, icu: pd.DataFrame) -> list[int]:
    counts: dict[int, int] = {}
    for chunk in iter_chartevents(config.chartevents, config.chunksize):
        windowed = apply_temporal_window(chunk, icu, config.window_hours)
        for item_id, count in windowed["ITEMID"].value_counts().items():
            counts[int(item_id)] = counts.get(int(item_id), 0) + int(count)

    ranked = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
    return [item_id for item_id, _ in ranked[: config.top_n_items]]


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

    return icu[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "LOS"]].merge(wide, on="ICUSTAY_ID", how="inner")


def add_item_summary(config: PreprocessConfig, selected_items: list[int]) -> None:
    summary = pd.DataFrame({"ITEMID": selected_items})
    if config.d_items and config.d_items.exists():
        d_items = normalize_columns(pd.read_csv(config.d_items))
        cols = [col for col in ["ITEMID", "LABEL", "CATEGORY", "UNITNAME"] if col in d_items.columns]
        summary = summary.merge(d_items[cols], on="ITEMID", how="left")

    config.output_item_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.output_item_summary, index=False)


def split_impute_scale(features: pd.DataFrame, config: PreprocessConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_cols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]
    target_col = "LOS"
    feature_cols = [col for col in features.columns if col not in id_cols + [target_col]]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    train_idx, test_idx = next(splitter.split(features, groups=features["SUBJECT_ID"]))
    train = features.iloc[train_idx].copy()
    test = features.iloc[test_idx].copy()

    imputer = SimpleImputer(strategy="median", add_indicator=True)
    scaler = StandardScaler()

    train_values = imputer.fit_transform(train[feature_cols])
    test_values = imputer.transform(test[feature_cols])

    indicator_cols = [f"missing_{feature_cols[index]}" for index in imputer.indicator_.features_]
    transformed_cols = feature_cols + indicator_cols

    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.transform(test_values)

    train_out = pd.concat(
        [
            train[id_cols + [target_col]].reset_index(drop=True),
            pd.DataFrame(train_scaled, columns=transformed_cols),
        ],
        axis=1,
    )
    test_out = pd.concat(
        [
            test[id_cols + [target_col]].reset_index(drop=True),
            pd.DataFrame(test_scaled, columns=transformed_cols),
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
        output_features=Path(args.output_features or raw["output"]["features"]),
        output_train=Path(args.output_train or raw["output"]["train"]),
        output_test=Path(args.output_test or raw["output"]["test"]),
        output_item_summary=Path(args.output_item_summary or raw["output"]["item_summary"]),
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
    parser.add_argument("--output-features", type=Path)
    parser.add_argument("--output-train", type=Path)
    parser.add_argument("--output-test", type=Path)
    parser.add_argument("--output-item-summary", type=Path)
    parser.add_argument("--window-hours", type=int)
    parser.add_argument("--top-n-items", type=int)
    parser.add_argument("--test-size", type=float)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--chunksize", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    config = build_config(parse_args())
    for path in [config.output_features, config.output_train, config.output_test]:
        path.parent.mkdir(parents=True, exist_ok=True)

    icu = load_icustays(config.icustays)
    selected_items = select_top_items(config, icu)
    if not selected_items:
        raise ValueError("Could not select ITEMIDs. Check CHARTEVENTS, ICUSTAYS and the temporal window.")

    add_item_summary(config, selected_items)
    features = aggregate_features(config, icu, set(selected_items))
    features.to_csv(config.output_features, index=False)

    train, test = split_impute_scale(features, config)
    train.to_csv(config.output_train, index=False)
    test.to_csv(config.output_test, index=False)

    print(f"Selected {len(selected_items)} ITEMIDs.")
    print(f"Feature table: {features.shape[0]} rows x {features.shape[1]} columns")
    print(f"Train: {train.shape[0]} rows | Test: {test.shape[0]} rows")


if __name__ == "__main__":
    main()
