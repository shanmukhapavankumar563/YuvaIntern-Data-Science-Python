"""Leakage-aware baseline/model comparison and final held-out evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score,
                             recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_pipeline import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, TARGET, add_features


def make_pipeline(model, scale_numeric: bool) -> Pipeline:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(num_steps)
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES),
                              ("categorical", categorical, CATEGORICAL_FEATURES)])
    return Pipeline([("preprocess", prep), ("model", model)])


def train_evaluate(df: pd.DataFrame, out: Path, seed: int = 42) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    featured = add_features(df)
    X, y = featured[MODEL_FEATURES], featured[TARGET]
    # Keep identical feature/label records together across evaluation splits.
    groups = pd.util.hash_pandas_object(featured[MODEL_FEATURES + [TARGET]], index=False).to_numpy()
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(outer_cv.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups[train_idx]
    candidates = {
        "Logistic Regression": make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed), True),
        "Random Forest": make_pipeline(RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1), False),
    }
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    records = []; fitted = {}
    for name, pipeline in candidates.items():
        result = cross_validate(pipeline, X_train, y_train, groups=groups_train, cv=cv, scoring=scoring, n_jobs=1, return_train_score=False)
        record = {"model": name}
        for metric in scoring:
            record[f"cv_{metric}_mean"] = float(np.mean(result[f"test_{metric}"]))
            record[f"cv_{metric}_std"] = float(np.std(result[f"test_{metric}"], ddof=1))
        records.append(record)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
    comparison = pd.DataFrame(records).sort_values("cv_f1_mean", ascending=False)
    comparison.to_csv(out / "model_comparison.csv", index=False)
    selected_name = str(comparison.iloc[0]["model"])
    selected = fitted[selected_name]
    pred = selected.predict(X_test)
    probability = selected.predict_proba(X_test)[:, 1]
    metrics = {"accuracy": float(accuracy_score(y_test, pred)),
               "precision_survival": float(precision_score(y_test, pred, zero_division=0)),
               "recall_survival": float(recall_score(y_test, pred, zero_division=0)),
               "f1_survival": float(f1_score(y_test, pred, zero_division=0)),
               "roc_auc": float(roc_auc_score(y_test, probability)),
               "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
               "test_size": float(len(y_test) / len(y)), "random_state": seed,
               "split_strategy": "StratifiedGroupKFold: identical model feature/label rows stay in one split.",
               "selection_rule": "Highest mean five-fold stratified group CV F1 on training partition; test set not used for selection.",
               "selected_model": selected_name}
    report = classification_report(y_test, pred, labels=[0,1], target_names=["Did not survive", "Survived"], digits=4, zero_division=0)
    (out / "classification_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame({"actual_survived": y_test.to_numpy(), "predicted_survived": pred,
                  "predicted_survival_probability": probability}).to_csv(out / "test_predictions.csv", index=False)
    cm = confusion_matrix(y_test, pred, labels=[0,1]); pd.DataFrame(cm, index=["actual_0","actual_1"], columns=["predicted_0","predicted_1"]).to_csv(out / "confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay(cm, display_labels=["Did not survive","Survived"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Held-out test confusion matrix\n{selected_name}"); fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,5)); RocCurveDisplay.from_predictions(y_test, probability, ax=ax, name=selected_name)
    ax.plot([0,1],[0,1], linestyle="--", color="gray", label="Chance"); ax.set_title("Held-out test ROC curve"); ax.legend(); fig.tight_layout(); fig.savefig(out / "roc_curve.png", dpi=170); plt.close(fig)
    # Permutation-free model explanation: coefficients for the selected logistic model, if selected.
    if selected_name == "Logistic Regression":
        pre = selected.named_steps["preprocess"]
        names = pre.get_feature_names_out()
        values = selected.named_steps["model"].coef_[0]
        importance = pd.DataFrame({"feature": names, "coefficient": values}).assign(abs_coefficient=lambda x: x.coefficient.abs()).sort_values("abs_coefficient", ascending=False)
        importance.to_csv(out / "model_coefficients.csv", index=False)
        top = importance.head(12).sort_values("coefficient")
        fig, ax = plt.subplots(figsize=(8,6)); ax.barh(top.feature, top.coefficient, color=np.where(top.coefficient > 0, "#3e8e7e", "#c65b57"))
        ax.axvline(0, color="black", linewidth=.8); ax.set(title="Largest standardized Logistic Regression coefficients", xlabel="Coefficient (direction of association)")
        fig.tight_layout(); fig.savefig(out / "model_coefficients.png", dpi=170); plt.close(fig)
    joblib.dump(selected, out / "selected_model.joblib")
    metrics["positive_class"] = "survived=1"
    metrics["versions"] = {"pandas": pd.__version__}
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"model_comparison": comparison, "selected_model": selected_name, "metrics": metrics,
            "classification_report": report, "confusion_matrix": cm.tolist(),
            "feature_names": MODEL_FEATURES, "train_rows": len(X_train), "test_rows": len(X_test)}
