"""Exploratory analysis for the Titanic internship project.

Run from this directory: python titanic_eda.py
Week 1's cleaned file is preferred when available; otherwise the public source is used.
All statistics and report findings are computed from the dataset actually loaded.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from docx import Document
from docx.shared import Inches, Pt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT = HERE / "outputs"
WEEK1_CLEAN = PROJECT / "Week-1-Data-Cleaning" / "outputs" / "titanic_cleaned.csv"
LOCAL_SOURCE = HERE / "data" / "raw" / "titanic.csv"
SOURCE_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "axes.titleweight": "bold", "font.size": 10})
sns.set_theme(style="whitegrid", palette="deep")


def load_data() -> tuple[pd.DataFrame, str]:
    """Use the Week 1 product first, then a cached or freshly downloaded source CSV."""
    if WEEK1_CLEAN.exists():
        return pd.read_csv(WEEK1_CLEAN), str(WEEK1_CLEAN.relative_to(PROJECT))
    LOCAL_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if not LOCAL_SOURCE.exists():
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "YuvaIntern-Week2-EDA/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            LOCAL_SOURCE.write_bytes(response.read())
    return pd.read_csv(LOCAL_SOURCE), SOURCE_URL + " (cached at data/raw/titanic.csv)"


def require(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return requested columns that exist; reports gracefully if this is a variant."""
    return [column for column in columns if column in df.columns]


def safe_rate(group: pd.DataFrame, outcome: str = "survived") -> float:
    if outcome not in group or group[outcome].notna().sum() == 0:
        return float("nan")
    return float(group[outcome].mean())


def save_fig(fig: plt.Figure, filename: str) -> Path:
    path = OUT / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def create_charts(df: pd.DataFrame) -> list[Path]:
    charts: list[Path] = []
    numeric = df.select_dtypes(include="number").columns.tolist()
    if "age" in numeric:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df, x="age", kde=True, bins=25, ax=ax, color="#3979a8")
        ax.set(title="Passenger age distribution", xlabel="Age (years)", ylabel="Passenger count")
        charts.append(save_fig(fig, "age_distribution.png"))
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df, x="age", ax=ax, color="#82b6d9")
        ax.set(title="Passenger age: spread and potential extreme values", xlabel="Age (years)")
        charts.append(save_fig(fig, "age_boxplot.png"))
    if "fare" in numeric:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df, x="fare", kde=True, bins=30, ax=ax, color="#db9442")
        ax.set(title="Passenger fare distribution", xlabel="Fare (dataset units)", ylabel="Passenger count")
        charts.append(save_fig(fig, "fare_distribution.png"))
        if "pclass" in df:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df, x="pclass", y="fare", hue="pclass", legend=False, ax=ax, palette="Blues")
            ax.set(title="Fare distributions by passenger class", xlabel="Passenger class (1 = first)", ylabel="Fare (dataset units)")
            charts.append(save_fig(fig, "fare_by_class.png"))
    if "survived" in df:
        if "pclass" in df:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=df, x="pclass", y="survived", hue="pclass", legend=False, errorbar=("ci", 95), ax=ax, palette="Blues")
            ax.set(title="Observed survival proportion by passenger class", xlabel="Passenger class (1 = first)", ylabel="Survival proportion", ylim=(0, 1))
            charts.append(save_fig(fig, "survival_by_class.png"))
        if "sex" in df:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.barplot(data=df, x="sex", y="survived", hue="sex", legend=False, errorbar=("ci", 95), ax=ax, palette="Set2")
            ax.set(title="Observed survival proportion by sex", xlabel="Recorded sex", ylabel="Survival proportion", ylim=(0, 1))
            charts.append(save_fig(fig, "survival_by_sex.png"))
        if "age" in df:
            # Fixed, readable age bands; preserve missing ages as a visible category in summaries.
            bins = [0, 12, 18, 35, 60, float("inf")]
            labels = ["0–11", "12–17", "18–34", "35–59", "60+"]
            plot_df = df[["age", "survived"]].copy()
            plot_df["age_group"] = pd.cut(plot_df["age"], bins=bins, labels=labels, right=False)
            plot_df["age_group"] = plot_df["age_group"].cat.add_categories("Missing age").fillna("Missing age")
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.barplot(data=plot_df, x="age_group", y="survived", hue="age_group", legend=False, errorbar=("ci", 95), ax=ax, palette="crest")
            ax.set(title="Observed survival proportion across age groups", xlabel="Age group (years)", ylabel="Survival proportion", ylim=(0, 1))
            charts.append(save_fig(fig, "survival_by_age_group.png"))
    if "fare" in numeric and "age" in numeric and "survived" in df:
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(data=df, x="age", y="fare", hue="survived", style="survived", alpha=.65, ax=ax, palette={0: "#c55a5a", 1: "#3a9873"})
        ax.set(title="Age and fare by recorded survival outcome", xlabel="Age (years)", ylabel="Fare (dataset units)")
        ax.legend(title="Survived (0 = no, 1 = yes)")
        charts.append(save_fig(fig, "age_fare_survival_scatter.png"))
    corr_cols = [c for c in numeric if df[c].nunique(dropna=True) > 1]
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(max(7, len(corr_cols) * .65), max(5, len(corr_cols) * .55)))
        sns.heatmap(corr, cmap="vlag", center=0, annot=len(corr_cols) <= 10, fmt=".2f", square=True, ax=ax)
        ax.set_title("Correlation among numeric columns")
        charts.append(save_fig(fig, "numeric_correlation_heatmap.png"))
    for col in require(df, ["pclass", "sex", "embarked"]):
        fig, ax = plt.subplots(figsize=(7, 4))
        order = df[col].value_counts(dropna=False).index
        sns.countplot(data=df, x=col, order=order, ax=ax, color="#5687a8")
        ax.set(title=f"Passenger counts by {col}", xlabel=col.replace("_", " ").title(), ylabel="Passenger count")
        charts.append(save_fig(fig, f"count_{col}.png"))
    return charts


def add_df_table(doc: Document, frame: pd.DataFrame, max_rows: int = 20) -> None:
    frame = frame.head(max_rows)
    table = doc.add_table(rows=1, cols=len(frame.columns)); table.style = "Light Shading Accent 1"
    for cell, val in zip(table.rows[0].cells, frame.columns): cell.text = str(val)
    for _, row in frame.iterrows():
        cells = table.add_row().cells
        for cell, val in zip(cells, row):
            cell.text = "—" if pd.isna(val) else (f"{val:.3f}" if isinstance(val, float) else str(val))


def create_report(df: pd.DataFrame, source: str, stats: pd.DataFrame, missing: pd.Series,
                  survival_tables: dict[str, pd.DataFrame], charts: list[Path],
                  age_group_table: pd.DataFrame, corr: pd.DataFrame) -> None:
    doc = Document()
    doc.add_heading("Week 2: Exploratory Data Analysis and Visualization", 0)
    doc.add_paragraph("YuvaIntern Virtual Data Science with Python Trainee Internship")
    doc.add_heading("Objective", 1)
    doc.add_paragraph("Explore Titanic passenger characteristics and observed survival patterns with descriptive statistics and visualizations. This is descriptive analysis; associations do not establish causation.")
    doc.add_heading("Dataset source and description", 1)
    doc.add_paragraph(f"Dataset loaded from: {source}. The public fallback is the Titanic teaching dataset in the seaborn-data GitHub repository ({SOURCE_URL}). The script prefers Week 1's cleaned CSV when present; otherwise it downloads and caches the source CSV in this Week 2 folder.")
    doc.add_paragraph(f"Loaded data contains {df.shape[0]} records and {df.shape[1]} fields. Fields include passenger outcome, class, sex, age, fare, family counts and embarkation details when present. Source columns are not removed by this EDA workflow.")
    doc.add_heading("Workflow and initial inspection", 1)
    doc.add_paragraph("The workflow inspects dimensions, names, pandas data types, sample records, descriptive statistics, and per-column missing counts. Numeric and categorical fields are identified from their pandas dtypes. Figures and group summaries are created only for columns that exist, so compatible Titanic variants can still be explored.")
    doc.add_paragraph("Missing values are not globally imputed or dropped for EDA. Grouped analyses exclude missing outcome values from the rate denominator; missing age is represented as its own group in the age summary. This avoids silently changing the data and makes available denominators explicit.")
    doc.add_paragraph("Shape and field names are available in outputs/inspection.json; data types, missing counts, sample rows and descriptive statistics are saved as CSV files in outputs/.")
    doc.add_heading("Descriptive statistics", 1)
    add_df_table(doc, stats.reset_index().rename(columns={"index": "variable"}).head(18))
    doc.add_paragraph("Missing cell counts by field:")
    add_df_table(doc, missing.rename_axis("field").reset_index(name="missing_count"))
    doc.add_heading("Exploratory comparisons", 1)
    for label, table in survival_tables.items():
        doc.add_heading(label, 2)
        add_df_table(doc, table.reset_index())
    if not age_group_table.empty:
        doc.add_heading("Age groups and survival", 2); add_df_table(doc, age_group_table.reset_index())
    if not corr.empty:
        doc.add_heading("Numeric correlations", 2)
        doc.add_paragraph("Pearson correlations summarize linear association among numeric columns. Binary and encoded category columns require careful interpretation; a correlation does not imply a causal effect.")
        add_df_table(doc, corr.reset_index().rename(columns={"index": "variable"}).head(12))
    doc.add_heading("Visualizations and interpretations", 1)
    descriptions = {
        "age_distribution.png": "Age distribution and KDE show the shape of recorded passenger ages; KDE smooths the histogram and can be unreliable at boundaries.",
        "age_boxplot.png": "The box plot summarizes age spread and marks possible extreme values; flagged values may still be valid passengers.",
        "fare_distribution.png": "Fare distribution shows concentration and skew; high fares can reflect real ticket or party differences.",
        "fare_by_class.png": "Class-wise fare distributions compare spread and central tendency across passenger classes.",
        "survival_by_class.png": "Bars show observed survival proportions by class with uncertainty intervals; they are descriptive group comparisons.",
        "survival_by_sex.png": "Bars compare observed survival proportions for the dataset's recorded sex categories.",
        "survival_by_age_group.png": "Age bands aid comparison; the selected cut points are analytical choices, not natural boundaries.",
        "age_fare_survival_scatter.png": "Each point is a passenger; overplotting and skew can hide patterns, and survival color is not evidence of cause.",
        "numeric_correlation_heatmap.png": "Cell colors summarize pairwise linear correlations among numeric columns; encoded categories can make values misleading.",
    }
    for chart in charts:
        doc.add_picture(str(chart), width=Inches(6.1))
        doc.add_paragraph(descriptions.get(chart.name, f"Passenger counts by {chart.stem.removeprefix('count_')} category."))
    doc.add_heading("Transformations, limitations, and improvements", 1)
    doc.add_paragraph("Survival rates are group means of the binary survived field, interpretable as observed proportions. Passenger age is grouped into 0–11, 12–17, 18–34, 35–59, and 60+ bands to support comparison; missing ages are kept as a separate group. No additional transformations are applied to the source data.")
    doc.add_paragraph("Potential anomalies include strongly skewed fare values, sparse categories, overlapping scatter points, and missing records. Titanic data describe a historical passenger sample and are not representative of all people. The public teaching CSV may contain derived or simplified fields. Results are sensitive to missing-data handling, age-band boundaries, and unequal group sizes. Future work could show group denominators, use robust summaries, inspect individual flagged records, and validate patterns with appropriate inferential methods.")
    doc.add_heading("Conclusion", 1)
    doc.add_paragraph("This analysis summarizes the available Titanic dataset and visualizes key distributions and observed survival associations. The generated tables and plots support further questions, but should be treated as exploratory evidence rather than causal conclusions or predictions.")
    doc.save(OUT / "report.docx")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, source = load_data()
    # Store core inspection artifacts before analysis, preserving nulls in the loaded data.
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    stats = df.describe(include="all").transpose()
    missing = df.isna().sum()
    stats.to_csv(OUT / "descriptive_statistics.csv")
    missing.rename("missing_count").to_csv(OUT / "missing_values.csv")
    df.head(10).to_csv(OUT / "sample_records.csv", index=False)
    inspection = {"source": source, "shape": list(df.shape), "columns": list(df.columns),
                  "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
                  "numeric_columns": numeric_cols, "categorical_columns": categorical_cols,
                  "missing_values": {str(k): int(v) for k, v in missing.items()},
                  "rows_with_any_missing": int(df.isna().any(axis=1).sum())}
    (OUT / "inspection.json").write_text(json.dumps(inspection, indent=2), encoding="utf-8")

    survival_tables: dict[str, pd.DataFrame] = {}
    if "survived" in df:
        for col, title in [("pclass", "Survival by passenger class"), ("sex", "Survival by sex"), ("embarked", "Survival by embarkation")]:
            if col in df:
                grouped = df.groupby(col, dropna=False)["survived"].agg(passengers="count", survivors="sum", survival_rate="mean")
                survival_tables[title] = grouped
                grouped.to_csv(OUT / f"survival_by_{col}.csv")
    age_group_table = pd.DataFrame()
    if "age" in df and "survived" in df:
        bins = [0, 12, 18, 35, 60, float("inf")]; labels = ["0–11", "12–17", "18–34", "35–59", "60+"]
        temp = df[["age", "survived"]].copy()
        temp["age_group"] = pd.cut(temp["age"], bins=bins, labels=labels, right=False)
        temp["age_group"] = temp["age_group"].cat.add_categories("Missing age").fillna("Missing age")
        age_group_table = temp.groupby("age_group", observed=False)["survived"].agg(passengers="count", survivors="sum", survival_rate="mean")
        age_group_table.to_csv(OUT / "survival_by_age_group.csv")
    corr = df[numeric_cols].corr(numeric_only=True) if len(numeric_cols) >= 2 else pd.DataFrame()
    if not corr.empty: corr.to_csv(OUT / "numeric_correlations.csv")
    charts = create_charts(df)
    create_report(df, source, stats, missing, survival_tables, charts, age_group_table, corr)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from {source}")
    print(f"Created {len(charts)} charts and the report in {OUT}")


if __name__ == "__main__":
    main()
