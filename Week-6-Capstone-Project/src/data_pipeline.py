"""Acquisition, inspection, conservative cleaning, and feature preparation."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
KAGGLE_SOURCE = "https://www.kaggle.com/competitions/titanic/data"
TARGET = "survived"
MODEL_FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "family_size", "is_alone"]
NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare", "family_size"]
CATEGORICAL_FEATURES = ["sex", "embarked", "is_alone"]


def acquire_data(raw_path: Path) -> tuple[pd.DataFrame, str]:
    """Read cached public CSV, downloading it once when absent."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "YuvaIntern-Week6-Capstone/1.0"})
        with urllib.request.urlopen(request, timeout=40) as response:
            payload = response.read()
        if not payload.startswith(b"survived,"):
            raise ValueError("Dataset download did not resemble the expected Titanic CSV.")
        raw_path.write_bytes(payload)
    return pd.read_csv(raw_path), SOURCE_URL + " (Seaborn example-data mirror; original source is Kaggle)"


def quality_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": {str(k): int(v) for k, v in df.isna().sum().items()},
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "target_counts": {str(k): int(v) for k, v in df[TARGET].value_counts(dropna=False).items()} if TARGET in df else {},
    }


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize text; mask impossible numeric/target values; drop exact duplicates only."""
    data = raw.copy()
    before = quality_summary(data)
    action_counts = {"whitespace_cells_normalized": 0, "invalid_age_values": 0, "invalid_fare_values": 0,
                     "invalid_target_values": 0, "exact_duplicate_rows_removed": 0,
                     "exact_duplicate_rows_retained_due_to_missing_passenger_ids": before["duplicate_rows"]}
    for column in data.select_dtypes(include="object").columns:
        non_null = data[column].notna()
        stripped = data.loc[non_null, column].astype(str).str.strip()
        action_counts["whitespace_cells_normalized"] += int((stripped.values != data.loc[non_null, column].astype(str).values).sum())
        data.loc[non_null, column] = stripped
    for col in ("age", "fare"):
        if col in data:
            numeric = pd.to_numeric(data[col], errors="coerce")
            invalid = numeric.lt(0) & numeric.notna()
            action_counts[f"invalid_{col}_values"] = int(invalid.sum())
            data.loc[invalid, col] = np.nan
            data[col] = numeric
    if TARGET in data:
        invalid = ~data[TARGET].isin([0, 1]) & data[TARGET].notna()
        action_counts["invalid_target_values"] = int(invalid.sum())
        data.loc[invalid, TARGET] = np.nan
    # The mirror omits passenger IDs/names, so exact row matches may be distinct
    # travelers. Preserve them and keep identical model records in one split later.
    # Rows lacking a known outcome cannot train/evaluate a supervised model.
    missing_target = int(data[TARGET].isna().sum())
    data = data.dropna(subset=[TARGET]).copy()
    data[TARGET] = data[TARGET].astype(int)
    after = quality_summary(data)
    return data, {"before": before, "after": after, "actions": action_counts,
                  "rows_without_target_removed": missing_target,
                  "outlier_policy": "Age and fare IQR flags are reported for review; valid extreme values are retained."}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create family size and alone indicator from family counts."""
    out = df.copy()
    out["family_size"] = out["sibsp"].fillna(0) + out["parch"].fillna(0) + 1
    out["is_alone"] = np.where((out["sibsp"].fillna(0) + out["parch"].fillna(0)) == 0, "yes", "no")
    return out


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
