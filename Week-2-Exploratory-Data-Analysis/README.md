# Week 2: Exploratory Data Analysis and Visualization

## Objective

Explore the Titanic dataset using Python, pandas, seaborn, and matplotlib. Summarize its structure and quality, investigate useful distributions and relationships, and explain the results in a generated Word report. The analysis is descriptive and does not treat association as causation.

## Dataset selection

The script first looks for `Week-1-Data-Cleaning/outputs/titanic_cleaned.csv`. That file was not present when this Week 2 project was prepared. If it remains unavailable, the script downloads the public [Titanic teaching dataset from mwaskom/seaborn-data](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv), caches it as `data/raw/titanic.csv`, and records the selected source in `outputs/inspection.json` and the report. If Week 1's cleaned CSV is later generated, it will automatically take precedence on the next run.

## Requirements

- Python 3.10 or later
- pandas
- matplotlib
- seaborn
- python-docx
- Internet access on first run when no local Week 1 or cached source CSV exists

Install from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, use `source .venv/bin/activate` to activate the environment.

## Run

```powershell
python eda_analysis.py
```

Run again to regenerate the tables, charts and report. The expected `eda_analysis.py` entry point delegates to the existing `titanic_eda.py` implementation. The script gracefully skips analyses that need columns absent from a compatible dataset variant. It does not impute or remove missing values for this exploratory analysis. In group survival rates, rows with missing survival outcomes do not contribute to the denominator; missing age has a separate age-group label.

## Outputs

All generated artifacts are in `outputs/`:

- `report.docx` — detailed report with findings, limitations and visualizations.
- `inspection.json` — source, dimensions, fields, data types, field roles, and missingness.
- `sample_records.csv`, `descriptive_statistics.csv`, `missing_values.csv` — initial inspection.
- `survival_by_*.csv`, `survival_by_age_group.csv` — grouped counts and observed survival proportions.
- `numeric_correlations.csv` — pairwise Pearson correlations for numeric fields.
- `*_distribution.png`, `*_boxplot.png`, `count_*.png`, `survival_*.png`, `fare_by_class.png`, `age_fare_survival_scatter.png`, `numeric_correlation_heatmap.png` — charts generated when required columns are available.

The age groups are fixed for readable comparisons: 0–11, 12–17, 18–34, 35–59, and 60+. The grouping is an analysis choice; the original age column remains unchanged. Rates are means of the binary `survived` field, so they describe proportions in this dataset, not causal effects.

## Internship instruction coverage

| Screenshot requirement | Where it is addressed |
|---|---|
| Public dataset, Python libraries, and EDA objective | Dataset section, dependencies, and report objective |
| Document the analysis approach and key findings | Generated Word report |
| DOCX deliverable | `outputs/report.docx` |
| Annotated visualizations | Chart functions set titles, labels, legends when applicable, and figure sizes |
| Basic statistics and interesting fields | Inspection artifacts and report descriptive-statistics section |
| Multiple charts for trends, relationships, and anomalies | Histograms, KDE, box plots, count plots, group survival charts, scatter plot, and numeric heatmap where columns permit |
| Interpret results and discuss why patterns may appear | Report visualization captions and exploratory comparisons |
| Explain transformations and aggregations | Age-band definition, survival-rate calculation and grouped tables documented in code and report |
| Depth, clarity, critical interpretation | Missing-data and dataset limitations plus non-causal interpretation included in report |
| Clear, organized DOCX | Sectioned report with tables and embedded charts |
| Quality of code and narrative | Commented, reusable script generates report figures and findings from the selected input |
| Approximate 30–35 hour task scope | This repository provides the deliverable workflow; time spent is not inferred or claimed |

## Limitations

The Titanic file is a historical teaching dataset. Missingness, unequal group sizes, skewed fares, overlapping points, encoded categorical variables, and chosen age boundaries affect interpretation. Correlation and grouped differences cannot establish causation. Further work may include denominators on every plot, robust statistical comparisons, and validation on independently sourced data.
