# Week 6 Capstone: Titanic Survival Analysis

## Project overview

This final capstone joins the most relevant earlier internship work into one reproducible pipeline: Titanic acquisition/cleaning (Weeks 1–2) and leakage-aware supervised classification (Week 4). It asks whether passenger attributes in this historical teaching dataset can predict the recorded binary survival outcome. It uses one coherent dataset and supervised models; no unsupervised stage is added because clusters would not directly answer the prediction question.

This is an educational analysis, not a causal study, safety policy, or tool for decisions about people.

## Dataset attribution and use

- Underlying source: [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data). Its data page documents survival, passenger class, age, family counts, fare, cabin, and embarkation fields.
- Download mirror used by the script: [Seaborn example-data Titanic CSV](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv), which identifies Kaggle as the source. The mirror README cautions that it exists for examples and may change; it is not an archival source.
- The CSV is downloaded on first run and cached at `data/raw/titanic.csv`. That data folder is Git-ignored. Kaggle competition rules and access conditions apply; review them before redistributing the dataset. The project does not claim a separate open-data license for the mirror.

The `survived` field is the target (0 = did not survive, 1 = survived). The chosen inputs are passenger class, recorded sex, age, sibling/spouse count, parent/child count, fare, and embarkation port. `family_size` and `is_alone` are engineered from the two family counts. The exact target duplicate `alive` and derived/redundant fields (`class`, `adult_male`, `who`, `alone`, `embark_town`) are excluded, along with identifiers and cabin/name/ticket fields.

## Pipeline and methodology

1. Acquire and cache the source CSV; validate that required fields exist.
2. Record dimensions, types, missingness, target balance and exact duplicates.
3. Trim text whitespace, mark negative age/fare and invalid binary outcomes missing, and exclude records with unknown target. Exact row matches are retained because this mirror lacks passenger identifiers; matching observed attributes do not prove the passengers are duplicate records. IQR flags are reported for review; valid extremes are retained.
4. Explore outcome, age, sex, passenger class, fares, correlations and missing values. Group survival rates describe sample associations and do not establish causality.
5. Make a stratified group holdout split before fitting imputation or encoding. Identical model feature/label rows stay together in train/test and five-fold CV because the dataset lacks passenger IDs. Compare Logistic Regression and Random Forest using CV on the training partition; choose by mean CV F1. Final metrics use the held-out test partition once.
6. Use scikit-learn pipelines: median numeric imputation, standardization for Logistic Regression, most-frequent categorical imputation and one-hot encoding. Transformations are fitted within each fold to prevent leakage.

## Requirements and installation

- Python 3.10+
- Internet access for first dataset download (or manually place the source CSV at `data/raw/titanic.csv`)
- Packages listed in `requirements.txt`

From this folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_capstone.py
```

On macOS/Linux, activate with `source .venv/bin/activate`. Rerun `python run_capstone.py` to reproduce the tables, plots, model, and Word report. The workflow uses a fixed random seed (42); exact results can still vary with software versions and runtime environment.

## Outputs

All generated results are stored in `outputs/`:

- `report.docx` — report with actual findings, model results, charts, limitations and recommendations.
- `titanic_cleaned.csv`, `modeling_dataset.csv` — analyzed records and engineered fields.
- `data_quality_before_after.json`, `source_attribution.json`, `run_summary.json` — audit trail, source and run metadata.
- `sample_records.csv`, `descriptive_statistics.csv`, `missing_values.csv`, `iqr_outlier_review.csv`, `numeric_correlations.csv` — inspection and quality outputs.
- `survival_by_*.csv`, `eda_findings.json` — grouped descriptive rates and findings.
- `eda_overview.png`, `fare_and_correlation.png` — exploratory visualizations.
- `model_comparison.csv`, `test_metrics.json`, `classification_report.txt`, `confusion_matrix.csv`, `confusion_matrix.png`, `roc_curve.png` — training CV comparison and held-out evaluation.
- `test_predictions.csv`, `selected_model.joblib` — test predictions and fitted preprocessing/model pipeline.
- `model_coefficients.csv` and `model_coefficients.png` — produced when Logistic Regression wins training CV; coefficients describe associations conditional on this model and are not causal effects.

## Limitations and ethical considerations

This is a small historical dataset with non-random sampling, missing fields, correlated family records, and a target shaped by a specific disaster. The result may reflect structural conditions and recorded sex/class attributes. Group rates and coefficients do not establish causes. A single split has sampling uncertainty; cross-validation is used for model selection but does not replace independent external validation. Do not use the model to make decisions about people. Useful extensions include family-group-aware splits, repeated/nested validation, uncertainty intervals, calibration, subgroup error analysis, threshold review, and independently sourced validation data.

## Week 6 instruction checklist

| Task requirement | Where addressed |
|---|---|
| Meaningful problem and public dataset | Survival prediction and source section above; source is recorded in outputs and report |
| Reproducible acquisition, inspection, cleaning, preprocessing, EDA | `src/data_pipeline.py`, `src/exploration.py`, `run_capstone.py`; ambiguous exact matches retained |
| Charts and explained patterns/limitations | EDA charts, grouped summaries, report interpretation and caveats |
| Target, baseline/improved model, validation and leakage controls | `src/modeling.py`; Logistic Regression and Random Forest, group-aware train-only CV and isolated test partition |
| Actual model results and practical interpretation | Generated `outputs/test_metrics.json`, comparison, confusion matrix, classification report, and report |
| Recommendations, bias and future improvements | README limitations and report ethics/recommendations sections |
| Modular readable code and run instructions | `src/` modules, documented commands and pinned minimum dependencies |
| Organized outputs and professional DOCX | `outputs/`; generated `outputs/report.docx` |
| Preserve Weeks 1–5 | Capstone adds only files under this folder; previous project folders were inspected and not modified during this work |
| Compare against original Week 6 task | Table above maps all requested project phases and deliverables |

The project does not claim a specified number of hours spent. Model scores and charts are generated by the run and must not be treated as guaranteed results on other samples.
