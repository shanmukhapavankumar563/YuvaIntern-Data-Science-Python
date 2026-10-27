"""Descriptive summaries, data quality outputs, and report-ready visualizations."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")


def iqr_review(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.select_dtypes(include="number").columns:
        vals = df[col].dropna()
        if vals.empty:
            continue
        q1, q3 = vals.quantile([.25, .75]); spread = q3 - q1
        low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
        rows.append({"feature": col, "q1": q1, "q3": q3, "lower_bound": low, "upper_bound": high,
                     "flagged_count": int(((vals < low) | (vals > high)).sum())})
    return pd.DataFrame(rows)


def run_eda(df: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    df.head(12).to_csv(out / "sample_records.csv", index=False)
    df.describe(include="all").transpose().to_csv(out / "descriptive_statistics.csv")
    df.isna().sum().rename("missing_count").to_csv(out / "missing_values.csv")
    iqr = iqr_review(df); iqr.to_csv(out / "iqr_outlier_review.csv", index=False)
    numeric = df.select_dtypes(include="number").corr(numeric_only=True)
    numeric.to_csv(out / "numeric_correlations.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.countplot(data=df, x="survived", hue="survived", legend=False, ax=axes[0, 0], palette=["#c65b57", "#3e8e7e"])
    axes[0, 0].set(title="Outcome counts", xlabel="Survived (0 = no, 1 = yes)", ylabel="Passengers")
    sns.histplot(data=df, x="age", hue="survived", bins=24, kde=True, stat="density", common_norm=False, ax=axes[0, 1])
    axes[0, 1].set(title="Age distributions by observed outcome", xlabel="Age (years)", ylabel="Density")
    sns.barplot(data=df, x="pclass", y="survived", hue="pclass", legend=False, errorbar=("ci", 95), ax=axes[1, 0], palette="Blues")
    axes[1, 0].set(title="Observed survival proportion by class", xlabel="Passenger class", ylabel="Proportion", ylim=(0, 1))
    sns.barplot(data=df, x="sex", y="survived", hue="sex", legend=False, errorbar=("ci", 95), ax=axes[1, 1], palette="Set2")
    axes[1, 1].set(title="Observed survival proportion by recorded sex", xlabel="Recorded sex", ylabel="Proportion", ylim=(0, 1))
    fig.suptitle("Titanic passenger exploratory analysis", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "eda_overview.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.boxplot(data=df, x="pclass", y="fare", hue="pclass", legend=False, ax=axes[0], palette="Blues")
    axes[0].set(title="Fare variation across passenger class", xlabel="Passenger class", ylabel="Fare (source units)")
    sns.heatmap(numeric, cmap="vlag", center=0, annot=True, fmt=".2f", square=True, ax=axes[1], cbar_kws={"shrink": .75})
    axes[1].set(title="Numeric feature correlations")
    fig.tight_layout(); fig.savefig(out / "fare_and_correlation.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    by_sex = df.groupby("sex", dropna=False)["survived"].agg(["count", "mean"]).rename(columns={"mean": "survival_rate"})
    by_class = df.groupby("pclass")["survived"].agg(["count", "mean"]).rename(columns={"mean": "survival_rate"})
    by_age = pd.cut(df["age"], [0, 12, 18, 35, 60, float("inf")], right=False,
                    labels=["0-11", "12-17", "18-34", "35-59", "60+"]).astype("object").fillna("Missing")
    age_table = df.assign(age_group=by_age).groupby("age_group", observed=False)["survived"].agg(["count", "mean"]).rename(columns={"mean": "survival_rate"})
    for frame, name in [(by_sex,"survival_by_sex.csv"),(by_class,"survival_by_class.csv"),(age_table,"survival_by_age_group.csv")]:
        frame.to_csv(out / name)
    findings = {"overall_survival_rate": float(df.survived.mean()),
                "survival_rate_by_sex": {str(k): float(v) for k,v in by_sex.survival_rate.items()},
                "survival_rate_by_class": {str(k): float(v) for k,v in by_class.survival_rate.items()},
                "rows": int(len(df)), "columns": int(df.shape[1]),
                "fare_skew_note": "Fare has a long upper tail; high IQR flags are not automatically errors."}
    (out / "eda_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    return findings
