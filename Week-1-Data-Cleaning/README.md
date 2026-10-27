# Week 1: Data Acquisition, Cleaning, and Preprocessing

## Objective

Acquire a public dataset, inspect its structure and quality, clean missing values and invalid entries using documented choices, review potential outliers, and produce an auditable cleaned dataset and report.

## Dataset

This project uses the Titanic passenger teaching dataset from [mwaskom/seaborn-data](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv). The script downloads the CSV from the repository's raw file on the first run and saves it as `data/raw/titanic.csv`. Keep that source file unchanged; generated artifacts are in `outputs/`.

## Requirements

- Python 3.10 or later
- pandas
- matplotlib
- python-docx
- Internet access for the first download (or place the source CSV at `data/raw/titanic.csv`)

## Setup and execution

From this directory, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python titanic_data_cleaning.py
```

On macOS/Linux, activate with `source .venv/bin/activate` instead. Run the script again whenever you want to regenerate the cleaned data, summaries, figures, and Word report. The downloaded source CSV is reused on later runs.

## Cleaning decisions

- Inspect schema, data types, descriptive statistics, missing cells, duplicate rows and numeric IQR flags before cleaning.
- Remove exact full-row duplicates only; retain records that are merely similar.
- Normalize surrounding whitespace in text values while preserving category spelling.
- Mark negative age/fare and non-binary survival values invalid, then handle them as missing.
- Fill age and other numeric missing values with their column median; fill `embarked` with its mode.
- Keep `deck` and other categorical fields, using `Unknown` when a value remains unavailable. Derive `embark_town` from `embarked` where possible.
- Report IQR-flagged observations for review; do not automatically drop or cap valid passengers.

The report and JSON logs contain data-dependent counts from the actual local run. No results are pre-filled or assumed.

## Outputs

| Path | Contents |
|---|---|
| `data/raw/titanic.csv` | Downloaded source data, kept unchanged |
| `outputs/titanic_cleaned.csv` | Cleaned passenger data |
| `outputs/report.docx` | Detailed report with findings, code examples and charts |
| `outputs/quality_summary_before_after.json` | Dataset dimensions, duplicate counts, missing cells, dtypes and missing counts |
| `outputs/cleaning_log.json` | Source and recorded cleaning actions |
| `outputs/iqr_outlier_review.json` | IQR bounds and flagged counts for numeric fields |
| `outputs/data_types.csv`, `descriptive_statistics.csv`, `missing_values.csv` | Initial exploration tables |
| `outputs/*.png` | Missingness, numeric outlier review and survival charts |

## Reproducibility notes

The program is deterministic for a fixed source CSV and dependency versions. For predictive modeling, imputation values should be learned from training data only to avoid leakage. The script does not install packages automatically.
