"""Run the complete Week 6 Titanic survival capstone from acquisition to report."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data_pipeline import SOURCE_URL, acquire_data, clean_data, quality_summary, save_json, add_features
from src.exploration import run_eda
from src.modeling import train_evaluate
from src.reporting import Report


def create_report(raw: pd.DataFrame, clean: pd.DataFrame, audit: dict, findings: dict, model: dict, out: Path):
    report = Report()
    report.heading("Week 6 Capstone: Titanic Survival Analysis", 1)
    report.paragraph("YuvaIntern Virtual Data Science with Python Trainee Internship | Final capstone")
    report.heading("Executive summary")
    m = model["metrics"]
    report.paragraph(f"This reproducible project combines public-data acquisition, careful data inspection, exploratory analysis and supervised classification. From {len(clean)} passenger records with known outcomes, the selected {model['selected_model']} achieved held-out accuracy {m['accuracy']:.3f}, survival-class F1 {m['f1_survival']:.3f}, and ROC-AUC {m['roc_auc']:.3f}. These are results from a single stratified split of a small historical teaching dataset, not evidence of causal effects or real-world predictive performance.")
    report.heading("Problem statement and objectives")
    report.paragraph("The objective is to examine which recorded passenger attributes were associated with Titanic survival and to build a reproducible educational model predicting the binary outcome survived (0 = did not survive, 1 = survived). The project demonstrates an end-to-end workflow; it is not intended to inform safety policy or decisions about people.")
    report.heading("Dataset source, acquisition and description")
    report.paragraph(f"The source CSV is downloaded from {SOURCE_URL}. It is a Seaborn example-data mirror; the repository identifies Kaggle's Titanic competition as the underlying source. Kaggle's data page describes the fields and competition train/test setup. Competition access and use are subject to Kaggle's competition rules; check those terms before redistributing the source data. This project caches the file locally and excludes it from Git.")
    report.paragraph(f"Raw source dimensions: {raw.shape[0]} rows × {raw.shape[1]} columns. The file includes survival, passenger class, sex, age, family counts, fare, embarkation, and redundant/derived columns. The modeling target is survived.")
    report.heading("Data quality, cleaning and preprocessing")
    report.table(["Quality check","Before","After"], [["Rows", audit["before"]["rows"], audit["after"]["rows"]], ["Columns", audit["before"]["columns"], audit["after"]["columns"]], ["Exact duplicate rows", audit["before"]["duplicate_rows"], audit["after"]["duplicate_rows"]], ["Missing cells", audit["before"]["missing_cells"], audit["after"]["missing_cells"]]])
    report.paragraph(f"Whitespace around text categories is trimmed. Negative ages/fares and non-binary outcomes, if present, are marked missing and counted. {audit['before']['duplicate_rows']} exact row matches were found but retained: this mirror has no passenger IDs, so identical observed attributes do not prove that two rows represent the same traveler. Identical model feature/label rows are grouped and kept on one side of every train/test and cross-validation split. Rows without an outcome cannot train this supervised model and are excluded. Missing predictor values are imputed within each training fold through the model pipeline. IQR flags are saved for review; high fares can be legitimate, so outliers are not automatically deleted.")
    report.paragraph("The EDA describes distributions, missingness, group survival proportions, fare differences and numeric correlations. Age groups are used only for readable descriptive summaries. Associations do not demonstrate causation.")
    report.heading("Feature engineering and leakage prevention")
    report.paragraph("Inputs include passenger class, recorded sex, age, sibling/spouse count, parent/child count, fare and embarkation port. Family size is derived as sibsp + parch + 1, and is_alone from family counts. `alive` is an exact text duplicate of the target; `class` repeats pclass; `adult_male`, `who`, `alone`, `embark_town`, names, ticket and cabin are excluded because they duplicate, derive from, identify, or have extensive missingness beyond this capstone's chosen inputs. The target and test labels are never predictors.")
    report.heading("Modeling, validation and metrics")
    report.paragraph(f"A stratified group holdout split (random_state=42) is created before fitting transformations. Identical model feature/label rows are kept together because the source has no passenger identifiers; the held-out fraction is {m['test_size']:.1%}. Logistic Regression is the interpretable baseline; Random Forest tests a nonlinear alternative. Five-fold stratified group cross-validation on training data selects the model with highest mean F1. Median numeric imputation, standardization for Logistic Regression, most-frequent categorical imputation and one-hot encoding are contained within sklearn pipelines, so each fold learns preprocessing only from its training partition. The held-out test set is used once for final reporting.")
    report.table(["Candidate","CV F1 mean","CV F1 std"], [[r["model"], f"{r['cv_f1_mean']:.3f}", f"{r['cv_f1_std']:.3f}"] for _,r in model["model_comparison"].iterrows()])
    report.table(["Final held-out metric","Value"], [["Selected model", model["selected_model"]], ["Accuracy",f"{m['accuracy']:.3f}"], ["Survival precision",f"{m['precision_survival']:.3f}"], ["Survival recall",f"{m['recall_survival']:.3f}"], ["Survival F1",f"{m['f1_survival']:.3f}"], ["ROC-AUC",f"{m['roc_auc']:.3f}"]])
    report.paragraph("A confusion matrix and class-specific report are included in outputs. Accuracy summarizes all decisions; survival precision describes the correctness of predicted survivors, while recall describes the share of actual survivors identified at the default 0.5 threshold. The ROC-AUC summarizes ranking across thresholds. None alone establishes operational usefulness.")
    report.heading("Exploratory findings and interpretation")
    sex_rates = findings["survival_rate_by_sex"]
    class_rates = findings["survival_rate_by_class"]
    report.paragraph(f"Observed overall survival proportion was {findings['overall_survival_rate']:.1%}. By recorded sex, proportions were {', '.join(f'{k}: {v:.1%}' for k,v in sex_rates.items())}. By passenger class, observed rates were {', '.join(f'class {k}: {v:.1%}' for k,v in class_rates.items())}. These differences reflect this historical sample and its context; they do not establish that any one attribute caused survival.")
    for filename, caption in [("eda_overview.png","Outcome, age, class, and sex exploratory charts"),("fare_and_correlation.png","Fare distribution by class and numeric correlations"),("confusion_matrix.png","Selected model confusion matrix on untouched test partition"),("roc_curve.png","Selected model ROC curve on untouched test partition")]:
        path=out/filename
        if path.exists(): report.image(path,caption)
    if (out/"model_coefficients.png").exists(): report.image(out/"model_coefficients.png","Largest standardized logistic coefficients; associations, not causal effects")
    report.heading("Recommendations, limitations and ethics")
    report.paragraph("Treat the findings as descriptive and educational. The sample is small, historical, non-random for modern populations, and records survival under an exceptional disaster; its class, sex and age patterns can encode social and operational inequities. Predictions must not be used to rank people or justify differential treatment. A single split has uncertainty, and records may not be independent where family groups traveled together. Future work could evaluate family-group-aware validation, repeated nested validation, calibration and threshold tradeoffs, confidence intervals, subgroup error rates, and external data if a defensible source is available. No clustering is included because unsupervised passenger segments would not directly improve the defined survival prediction question.")
    report.heading("Reproducibility and conclusion")
    report.paragraph("Run `python run_capstone.py` after installing requirements. The script downloads the source on the first run, saves the raw data locally (Git-ignored), regenerates tables, figures, predictions, a fitted pipeline and this report. Random seeds and split rules are fixed. Model performance is a measured result of this run, not a guarantee of future performance.")
    report.save(out/"report.docx")


def main():
    data_dir=ROOT/"data"/"raw"; out=ROOT/"outputs"; out.mkdir(parents=True,exist_ok=True)
    raw,source=acquire_data(data_dir/"titanic.csv")
    required={"survived","pclass","sex","age","sibsp","parch","fare","embarked"}
    missing=required-set(raw.columns)
    if missing: raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    clean,audit=clean_data(raw)
    clean.to_csv(out/"titanic_cleaned.csv",index=False)
    add_features(clean).to_csv(out/"modeling_dataset.csv",index=False)
    save_json(out/"data_quality_before_after.json",audit)
    (out/"source_attribution.json").write_text(json.dumps({"mirror_url":SOURCE_URL,"underlying_source":"Kaggle Titanic: Machine Learning from Disaster","kaggle_data_page":"https://www.kaggle.com/competitions/titanic/data","use_conditions":"Kaggle competition rules apply; review before redistribution.","local_cache":str((data_dir/"titanic.csv").relative_to(ROOT))},indent=2),encoding="utf-8")
    findings=run_eda(clean,out)
    model=train_evaluate(clean,out)
    summary={"source":source,"python":platform.python_version(),"pandas":pd.__version__,"scikit_learn":sklearn.__version__,"raw_shape":list(raw.shape),"clean_shape":list(clean.shape),"model":model["metrics"]}
    (out/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    create_report(raw,clean,audit,findings,model,out)
    print(json.dumps({"report":str(out/"report.docx"),"selected_model":model["selected_model"],"metrics":model["metrics"],"clean_rows":len(clean)},indent=2))


if __name__ == "__main__": main()
