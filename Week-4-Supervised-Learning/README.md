# Week 4: Supervised Learning Model Implementation

## Objective and prediction problem

This project compares binary classifiers for predicting the diagnosis label in the Breast Cancer Wisconsin (Diagnostic) teaching dataset. The target is `diagnosis` (`benign` or `malignant`); the inputs are 30 continuous cell-nucleus measurements. The positive class is `malignant`, so malignant recall is reported alongside precision, F1, accuracy, and ROC-AUC. This is an educational model-evaluation exercise only, not a diagnostic or treatment tool.

## Dataset source

The data are loaded from scikit-learn's bundled copy of the [UCI Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) using `sklearn.datasets.load_breast_cancer(as_frame=True)`. The package copy contains 569 observations, 30 numeric measurements, and the diagnosis target, and requires no network access when scikit-learn is installed. The source describes features derived from digitized fine needle aspirate images of breast masses; see the UCI page for attribution and CC BY 4.0 license information.

## Method

- Inspect shape, data types, class counts, missing values, and exact duplicate rows.
- Use all numeric feature measurements; the bundled feature matrix omits the UCI ID. Keep diagnosis strictly as the target.
- Split into stratified 80% training and 20% test sets using `random_state=42` before fitting any preprocessing.
- Use median imputation in a `ColumnTransformer` and a scikit-learn `Pipeline`. Logistic Regression also standardizes features. Random Forest uses imputation without scaling.
- Compare Logistic Regression (baseline) and Random Forest with five-fold stratified cross-validation on training data only. Select by mean cross-validation malignant-class F1, then report both models on the held-out test split.
- Save confusion matrix, ROC curve, feature interpretation plot, model, predictions, metric tables, and a data-dependent Word report.

Every preprocessing operation is inside each model pipeline, so cross-validation fits it only on each fold's training portion and final fitting uses only the training partition. The test set is reserved for final evaluation. No test labels are used for feature selection, hyperparameter tuning, or model selection.

## Setup and execution

Python 3.10 or later is recommended. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python supervised_learning.py
```

On macOS/Linux, activate with `source .venv/bin/activate`. The dataset is bundled in scikit-learn, so no separate data download is needed. The report and all metrics are generated from the run; they are not prefilled examples. `python-docx` is used when installed; otherwise, the script creates a Word-compatible DOCX with Python's standard library.

## Outputs

Generated files are saved under `outputs/`:

- `report.docx` — structured methodology, actual results, interpretation, ethics, and limitations.
- `model_comparison.csv` — training cross-validation means/standard deviations and held-out test metrics for each model.
- `classification_report.txt` — class-specific precision, recall, F1, and support for the selected model on the test split.
- `confusion_matrix.png`, `roc_curve.png` — held-out evaluation figures.
- `feature_importance.png`, `feature_importance.csv` — top model-specific coefficients/importances (not causal explanations).
- `best_model.joblib` — fitted selected pipeline.
- `test_predictions.csv` — held-out labels and malignant probability.
- `data_quality.json`, `feature_summary.csv`, `missing_values.csv`, `sample_records.csv`, `dataset_snapshot.csv` — inspection artifacts.
- `run_summary.json` — selected model, split details, versions, and selected model's test metrics.

## Internship instruction checklist

| Screenshot requirement | Project implementation |
|---|---|
| Design and implement supervised model in Python | `supervised_learning.py` implements a complete binary-classification workflow |
| Choose regression or classification using public data | UCI diagnostic classification dataset bundled by scikit-learn |
| Document preparation, feature engineering, training, validation, evaluation | Script, generated DOCX report, and README methodology |
| Complete DOCX modeling report | `outputs/report.docx` contains source, workflow, actual metrics, plots, limitations, and conclusion |
| Rationale, code, metrics, strengths and limitations | Report has model rationale, code excerpts, CV/test tables, and limitations |
| Define a prediction problem | Predict benign/malignant diagnosis; target and inputs are documented |
| Preprocess and engineer features | Train-only imputation/scaling pipelines; documented choice to keep original features and avoid unvalidated engineered features |
| Implement chosen model with scikit-learn | Logistic Regression baseline plus Random Forest comparison |
| Validate with cross-validation and relevant metrics | Five-fold stratified CV on training data; accuracy, malignant precision/recall/F1, ROC-AUC, and test confusion matrix |
| Discuss improvements and implications | Report discusses external validation, calibration, threshold choice, fairness, and clinical limits |
| Coherent report and analysis | Sectioned report with actual generated results and visualizations |
| Detailed DOC demonstrating practical skills | Data-dependent Word report and reproducible script |
| Estimated 30–35 focused hours | Deliverables are provided; time spent is not inferred or claimed |

## Limitations and ethical use

This is a modest, historical dataset with one train/test split and no external validation. Correlated measurements, sampling differences, and threshold choice affect results. Medical false negatives and false positives have serious consequences. Do not use this model for diagnosis, treatment, triage, or other real-world medical decisions. Real deployment would require representative validation, calibration, clinically chosen thresholds, privacy protections, fairness analysis, regulatory review, and clinician oversight.
