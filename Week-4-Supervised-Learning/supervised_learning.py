"""Supervised classification project using scikit-learn's bundled public dataset.

Run from this directory with: python supervised_learning.py
The data are loaded from scikit-learn; evaluated artifacts are written to outputs/.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, make_scorer, precision_score, recall_score,
                             roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"
POSITIVE_LABEL = "malignant"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def write_docx_fallback(path: Path, data: pd.DataFrame, source_url: str, quality: dict,
                        cv_table: pd.DataFrame, comparison: pd.DataFrame, selected_name: str,
                        selected_metrics: dict, report_text: str, confusion_path: Path,
                        roc_path: Path, feature_path: Path, feature_names: list[str]) -> None:
    """Build a simple, valid Word document with stdlib OOXML if python-docx is unavailable."""
    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
    A = "http://schemas.openxmlformats.org/drawingml/2006/main"
    PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
    REL = "http://schemas.openxmlformats.org/package/2006/relationships"
    CT = "http://schemas.openxmlformats.org/package/2006/content-types"
    for prefix, uri in [("w", W), ("r", R), ("wp", WP), ("a", A), ("pic", PIC), ("rel", REL), ("ct", CT)]:
        ET.register_namespace(prefix, uri)
    q = lambda ns, name: f"{{{ns}}}{name}"
    root = ET.Element(q(W, "document"))
    body = ET.SubElement(root, q(W, "body"))
    image_rels = []
    image_counter = 0

    def paragraph(text: str, style: str | None = None):
        p = ET.SubElement(body, q(W, "p"))
        if style:
            ppr = ET.SubElement(p, q(W, "pPr")); ps = ET.SubElement(ppr, q(W, "pStyle")); ps.set(q(W, "val"), style)
        run = ET.SubElement(p, q(W, "r")); t = ET.SubElement(run, q(W, "t")); t.text = str(text)
        if text.startswith(" ") or text.endswith(" "):
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")

    def heading(text: str, level: int = 1):
        paragraph(text, f"Heading{level}")

    def table(frame: pd.DataFrame, limit: int = 30):
        frame = frame.head(limit)
        if frame.empty:
            paragraph("No rows available."); return
        tbl = ET.SubElement(body, q(W, "tbl"))
        tblpr = ET.SubElement(tbl, q(W, "tblPr")); borders = ET.SubElement(tblpr, q(W, "tblBorders"))
        for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
            el = ET.SubElement(borders, q(W, edge)); el.set(q(W, "val"), "single"); el.set(q(W, "sz"), "4"); el.set(q(W, "color"), "B7C9D6")
        for row_index, values in enumerate([list(frame.columns)] + frame.astype(object).values.tolist()):
            tr = ET.SubElement(tbl, q(W, "tr"))
            for value in values:
                tc = ET.SubElement(tr, q(W, "tc")); p = ET.SubElement(tc, q(W, "p")); r = ET.SubElement(p, q(W, "r")); t = ET.SubElement(r, q(W, "t"))
                t.text = str(value) if not pd.isna(value) else "—"
                if row_index == 0:
                    rpr = ET.SubElement(r, q(W, "rPr")); ET.SubElement(rpr, q(W, "b"))

    def image(img_path: Path, caption: str):
        nonlocal image_counter
        image_counter += 1
        rel_id = f"rIdImage{image_counter}"
        filename = f"image{image_counter}.png"
        image_rels.append((rel_id, filename, img_path.read_bytes()))
        png = img_path.read_bytes()
        width_px = int.from_bytes(png[16:20], "big"); height_px = int.from_bytes(png[20:24], "big")
        cx = min(6_100_000, 5_900_000)
        cy = int(cx * height_px / max(width_px, 1))
        if cy > 5_200_000:
            cy = 5_200_000; cx = int(cy * width_px / max(height_px, 1))
        p = ET.SubElement(body, q(W, "p")); r = ET.SubElement(p, q(W, "r")); drawing = ET.SubElement(r, q(W, "drawing"))
        inline = ET.SubElement(drawing, q(WP, "inline")); inline.set("distT", "0"); inline.set("distB", "0"); inline.set("distL", "0"); inline.set("distR", "0")
        ET.SubElement(inline, q(WP, "extent"), {"cx": str(cx), "cy": str(cy)})
        ET.SubElement(inline, q(WP, "docPr"), {"id": str(image_counter), "name": filename, "descr": caption})
        frame = ET.SubElement(inline, q(WP, "cNvGraphicFramePr")); ET.SubElement(frame, q(A, "graphicFrameLocks"), {"noChangeAspect": "1"})
        graphic = ET.SubElement(inline, q(A, "graphic")); gd = ET.SubElement(graphic, q(A, "graphicData")); gd.set("uri", PIC)
        pic = ET.SubElement(gd, q(PIC, "pic")); nv = ET.SubElement(pic, q(PIC, "nvPicPr"))
        ET.SubElement(nv, q(PIC, "cNvPr"), {"id": str(image_counter), "name": filename}); ET.SubElement(nv, q(PIC, "cNvPicPr"))
        fill = ET.SubElement(pic, q(PIC, "blipFill")); ET.SubElement(fill, q(A, "blip"), {q(R, "embed"): rel_id}); stretch = ET.SubElement(fill, q(A, "stretch")); ET.SubElement(stretch, q(A, "fillRect"))
        sp = ET.SubElement(pic, q(PIC, "spPr")); xfrm = ET.SubElement(sp, q(A, "xfrm")); ET.SubElement(xfrm, q(A, "off"), {"x": "0", "y": "0"}); ET.SubElement(xfrm, q(A, "ext"), {"cx": str(cx), "cy": str(cy)})
        geom = ET.SubElement(sp, q(A, "prstGeom")); geom.set("prst", "rect"); ET.SubElement(geom, q(A, "avLst"))
        paragraph(caption)

    heading("Week 4: Supervised Learning Model Implementation", 1)
    paragraph("YuvaIntern Virtual Data Science with Python Trainee Internship")
    heading("Introduction and problem definition")
    paragraph("This project compares binary classifiers for predicting the diagnosis label in a public teaching dataset from numeric measurements. The positive class is malignant, making malignant-class recall important alongside precision, F1, accuracy, and ROC-AUC. This is an educational exercise only, not a clinical decision system and must not be used to diagnose or guide patient care.")
    heading("Dataset source and description")
    paragraph(f"Source: UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic): {source_url}. Loaded from scikit-learn {sklearn.__version__}'s bundled copy; no network download was needed. This run used {len(data)} observations and {len(feature_names)} real-valued features. Features describe cell-nucleus measurements derived from digitized fine needle aspirate images. The target is diagnosis: benign or malignant.")
    heading("Exploration and data quality")
    paragraph(f"Loaded shape: {quality['rows']} rows by {quality['columns']} columns including target. Missing feature cells: {quality['missing_feature_cells']}; exact duplicate rows: {quality['duplicate_rows']}. The target remains separate from predictors, and the bundled matrix contains no identifier feature. Class counts:")
    table(pd.DataFrame({"class": list(quality["class_counts"].keys()), "count": list(quality["class_counts"].values())}))
    paragraph("All 30 available numeric measurements were retained. Missingness, dtypes, example rows, and descriptive statistics are saved as CSV/JSON artifacts.")
    heading("Preprocessing and feature engineering")
    paragraph("The data were split into stratified 80% training and 20% test sets before fitting transformations. A ColumnTransformer and Pipeline put median imputation inside cross-validation folds, preventing preprocessing leakage. Logistic Regression also uses StandardScaler because the feature units differ. Random Forest uses median imputation without scaling because decision trees do not require standardization. No additional feature engineering was applied; retaining documented measures avoids unvalidated derived variables. The diagnosis is used only as the target.")
    heading("Models and validation strategy")
    paragraph("Logistic Regression is the regularized linear baseline (C=1, lbfgs solver, max_iter=3000). Random Forest is a nonlinear comparison (300 trees, unlimited depth, min_samples_leaf=1). Five-fold StratifiedKFold cross-validation compares candidates using only training data. The selection rule is highest mean training-fold malignant-class F1; the held-out test scores are not used to select the model. The test partition is evaluated once after training.")
    cv_report_cols = ["model", "cv_accuracy_mean", "cv_precision_malignant_mean", "cv_recall_malignant_mean", "cv_f1_malignant_mean", "cv_roc_auc_malignant_mean"]
    test_report_cols = ["model", "test_accuracy", "test_precision_malignant", "test_recall_malignant", "test_f1_malignant", "test_roc_auc_malignant"]
    heading("Cross-validation results (mean of five training folds)"); table(cv_table[cv_report_cols])
    heading("Held-out test model comparison"); table(comparison[test_report_cols])
    paragraph(f"Selected by training cross-validation malignant F1: {selected_name}. Its held-out test metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in selected_metrics.items()) + ". These results describe one split and do not establish perfect generalization.")
    best_test = comparison.loc[comparison["test_f1_malignant"].idxmax()]
    paragraph(f"The highest malignant F1 on this test split belongs to {best_test['model']} ({best_test['test_f1_malignant']:.4f}); this test result does not override the training-only cross-validation selection, because using it to select a model would leak test information into model choice.")
    heading("Classification report")
    for line in report_text.strip().splitlines(): paragraph(line)
    heading("Confusion matrix and ROC curve")
    image(confusion_path, "Held-out test confusion matrix: actual rows, predicted columns, benign and malignant classes.")
    image(roc_path, "Held-out ROC curve for the selected model; threshold performance and clinical costs require separate analysis.")
    heading("Feature interpretation")
    image(feature_path, "Top model-specific feature coefficients or importances; these are not causal explanations.")
    paragraph("Coefficients or impurity-based importances can be unstable when measurements are correlated. They describe the fitted model, not medical facts or causes.")
    heading("Strengths, limitations, ethics, and future improvements")
    paragraph("Strengths include public data, train-only preprocessing pipelines, stratified validation, cross-validation, an independent test partition, and transparent model comparison. Limitations include a modest historical sample, one split, no external validation, correlated features, and no decision-cost analysis. Medical errors have serious consequences. The model is not for real-world diagnosis or treatment. Further work would require representative external validation, calibration, threshold selection with clinical experts, privacy protections, fairness assessment, regulatory review, and human oversight.")
    heading("Conclusion")
    paragraph("This project demonstrates a reproducible supervised classification workflow with a linear baseline, a nonlinear comparison, cross-validation, and held-out metrics. Results are for learning model evaluation only.")
    heading("Reproducibility code excerpts")
    paragraph("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)")
    paragraph("pipeline = Pipeline([('preprocess', preprocessor), ('model', classifier)])")
    paragraph("cross_validate(pipeline, X_train, y_train, cv=StratifiedKFold(5), scoring=scoring)")

    sect = ET.SubElement(body, q(W, "sectPr")); ET.SubElement(sect, q(W, "pgSz"), {q(W, "w"): "12240", q(W, "h"): "15840"})
    ET.SubElement(sect, q(W, "pgMar"), {q(W, "top"): "900", q(W, "right"): "900", q(W, "bottom"): "900", q(W, "left"): "900"})
    styles = ET.Element(q(W, "styles"))
    defaults = ET.SubElement(styles, q(W, "docDefaults")); rpr_default = ET.SubElement(defaults, q(W, "rPrDefault")); rpr = ET.SubElement(rpr_default, q(W, "rPr"))
    ET.SubElement(rpr, q(W, "rFonts"), {q(W, "ascii"): "Calibri", q(W, "hAnsi"): "Calibri"}); ET.SubElement(rpr, q(W, "sz"), {q(W, "val"): "21"})
    for sid, size, color in [("Normal", "21", "202124"), ("Heading1", "32", "174A6E"), ("Heading2", "26", "245B7A")]:
        style = ET.SubElement(styles, q(W, "style"), {q(W, "type"): "paragraph", q(W, "styleId"): sid})
        ET.SubElement(style, q(W, "name"), {q(W, "val"): sid}); ppr = ET.SubElement(style, q(W, "pPr")); ET.SubElement(ppr, q(W, "keepNext"))
        srpr = ET.SubElement(style, q(W, "rPr")); ET.SubElement(srpr, q(W, "b")); ET.SubElement(srpr, q(W, "sz"), {q(W, "val"): size}); ET.SubElement(srpr, q(W, "color"), {q(W, "val"): color})
    relationships = ET.Element(q(REL, "Relationships"))
    ET.SubElement(relationships, q(REL, "Relationship"), {"Id": "rIdStyles", "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles", "Target": "styles.xml"})
    for rel_id, filename, _ in image_rels:
        ET.SubElement(relationships, q(REL, "Relationship"), {"Id": rel_id, "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", "Target": f"media/{filename}"})
    pkg_rels = ET.Element(q(REL, "Relationships")); ET.SubElement(pkg_rels, q(REL, "Relationship"), {"Id": "rId1", "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "Target": "word/document.xml"})
    types = ET.Element(q(CT, "Types")); ET.SubElement(types, q(CT, "Default"), {"Extension": "rels", "ContentType": "application/vnd.openxmlformats-package.relationships+xml"}); ET.SubElement(types, q(CT, "Default"), {"Extension": "xml", "ContentType": "application/xml"}); ET.SubElement(types, q(CT, "Default"), {"Extension": "png", "ContentType": "image/png"}); ET.SubElement(types, q(CT, "Override"), {"PartName": "/word/document.xml", "ContentType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"}); ET.SubElement(types, q(CT, "Override"), {"PartName": "/word/styles.xml", "ContentType": "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"})
    with ZipFile(path, "w", ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", ET.tostring(types, encoding="utf-8", xml_declaration=True))
        package.writestr("_rels/.rels", ET.tostring(pkg_rels, encoding="utf-8", xml_declaration=True))
        package.writestr("word/document.xml", ET.tostring(root, encoding="utf-8", xml_declaration=True))
        package.writestr("word/styles.xml", ET.tostring(styles, encoding="utf-8", xml_declaration=True))
        package.writestr("word/_rels/document.xml.rels", ET.tostring(relationships, encoding="utf-8", xml_declaration=True))
        for _, filename, contents in image_rels:
            package.writestr(f"word/media/{filename}", contents)


def load_dataset() -> tuple[pd.DataFrame, str, str]:
    """Load the packaged copy of the public UCI Wisconsin Diagnostic dataset."""
    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.data.copy()
    # Decode sklearn's documented target codes into readable labels; sklearn encodes 0=M, 1=B.
    target_map = {int(code): str(name) for code, name in enumerate(bunch.target_names)}
    frame["diagnosis"] = bunch.target.map(target_map)
    return frame, str(bunch.DESCR), sklearn.__version__


def build_preprocessor(features: list[str], scale: bool) -> ColumnTransformer:
    """Impute within each fitted pipeline; scale only for the linear model."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(steps=numeric_steps)
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipe, features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def metric_values(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_malignant": float(precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "recall_malignant": float(recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "f1_malignant": float(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "roc_auc_malignant": float(roc_auc_score((y_true == POSITIVE_LABEL).astype(int), y_score)),
    }


def save_confusion_plot(y_true: pd.Series, y_pred: np.ndarray, labels: list[str]) -> Path:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(title="Held-out test confusion matrix", xlabel="Predicted diagnosis", ylabel="Actual diagnosis",
           xticks=range(len(labels)), yticks=range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black", fontsize=12)
    fig.colorbar(image, ax=ax, label="Sample count")
    fig.tight_layout()
    path = OUT / "confusion_matrix.png"; fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    return path


def write_report(data: pd.DataFrame, source_description: str, source_url: str,
                 quality: dict, cv_table: pd.DataFrame, comparison: pd.DataFrame,
                 selected_name: str, selected_metrics: dict, report_text: str,
                 confusion_path: Path, roc_path: Path, feature_path: Path,
                 feature_names: list[str], fitted_model: Pipeline) -> None:
    """Create a report from observed run outputs; fail clearly if python-docx is absent."""
    try:
        from docx import Document
        from docx.shared import Inches, Pt
    except ImportError as exc:
        write_docx_fallback(OUT / "report.docx", data, source_url, quality, cv_table,
                            comparison, selected_name, selected_metrics, report_text,
                            confusion_path, roc_path, feature_path, feature_names)
        return

    def add_table(doc, frame: pd.DataFrame, limit: int = 35):
        frame = frame.head(limit)
        table = doc.add_table(rows=1, cols=len(frame.columns)); table.style = "Light Shading Accent 1"
        for cell, column in zip(table.rows[0].cells, frame.columns): cell.text = str(column)
        for _, row in frame.iterrows():
            cells = table.add_row().cells
            for cell, value in zip(cells, row):
                cell.text = "—" if pd.isna(value) else (f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))

    doc = Document()
    doc.add_heading("Week 4: Supervised Learning Model Implementation", 0)
    doc.add_paragraph("YuvaIntern Virtual Data Science with Python Trainee Internship")
    doc.add_heading("Introduction and problem definition", 1)
    doc.add_paragraph("This project trains supervised binary classifiers to predict the diagnosis label in a public teaching dataset from numeric measurements. The positive class is malignant, making malignant-class recall a key measure alongside precision, F1, accuracy, and ROC-AUC. This is a reproducible learning exercise only; it is not a clinical decision system and must not be used to diagnose or guide patient care.")
    doc.add_heading("Dataset source and description", 1)
    doc.add_paragraph(f"Source: UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic), at {source_url}. The dataset is loaded through scikit-learn's bundled copy (scikit-learn version {sklearn.__version__}), avoiding a runtime download. It contains {len(data)} observations and {len(feature_names)} real-valued features computed from digitized fine needle aspirate images. The target is diagnosis: benign or malignant. UCI describes 569 instances and 30 features and reports no missing values in the source.")
    doc.add_paragraph("The features summarize cell-nucleus characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. The intended use here is to demonstrate a supervised classification workflow, not clinical diagnosis.")
    doc.add_heading("Exploration and data quality", 1)
    doc.add_paragraph(f"Loaded shape: {quality['rows']} rows × {quality['columns']} columns including the target. The analysis inspected data types, class counts, missing values, and exact duplicate rows. Missing feature cells: {quality['missing_feature_cells']}; exact duplicate full rows: {quality['duplicate_rows']}. No ID field is supplied by the bundled feature matrix, so there is no identifier to use as a predictor. Potential leakage was addressed by keeping diagnosis exclusively as the target and fitting preprocessing only inside each training fold.")
    doc.add_paragraph("Class distribution:")
    add_table(doc, pd.DataFrame({"class": list(quality["class_counts"].keys()), "count": list(quality["class_counts"].values())}))
    doc.add_paragraph("Feature descriptions and descriptive statistics are saved in outputs/feature_summary.csv; schema and quality details are in outputs/data_quality.json.")
    doc.add_heading("Preprocessing and feature engineering", 1)
    doc.add_paragraph("All 30 numeric measurements were retained; no feature was selected using the test labels. Because the bundled dataset has numeric features only, categorical encoding was not needed. Median imputation is included in each model pipeline as a robust missing-value fallback even though the inspected data may have no missing cells. Logistic Regression also uses StandardScaler because feature magnitudes differ. Random Forest uses median imputation without scaling because tree split rules do not require standardized units. No additional derived features were added: preserving documented measurements keeps the baseline interpretable and avoids unvalidated transformations.")
    doc.add_heading("Models and validation strategy", 1)
    doc.add_paragraph("The data was split once into stratified 80% training and 20% held-out test partitions using random_state=42. Stratification preserves class proportions. All preprocessing is part of a scikit-learn Pipeline and is therefore fitted separately inside each training fold or on the training partition only. Five-fold StratifiedKFold cross-validation compares candidates on training data. The test set is not used to choose features, fit preprocessing, or select the model.")
    doc.add_paragraph("Logistic Regression is the baseline: it is a regularized linear classifier and provides a useful comparison for continuous measurements. It uses C=1.0, solver='lbfgs', max_iter=3000, and random_state=42. Random Forest adds a nonlinear tree-ensemble comparison with 300 trees, no depth limit, min_samples_leaf=1, and random_state=42. The model is selected by highest mean cross-validation malignant-class F1; final test scores are reported only after this training-only selection.")
    cv_report_cols = ["model", "cv_accuracy_mean", "cv_precision_malignant_mean", "cv_recall_malignant_mean", "cv_f1_malignant_mean", "cv_roc_auc_malignant_mean"]
    test_report_cols = ["model", "test_accuracy", "test_precision_malignant", "test_recall_malignant", "test_f1_malignant", "test_roc_auc_malignant"]
    doc.add_heading("Cross-validation results (mean of five training folds)", 1); add_table(doc, cv_table[cv_report_cols])
    doc.add_heading("Held-out test comparison", 1); add_table(doc, comparison[test_report_cols])
    doc.add_paragraph(f"Selected model by mean training-fold malignant F1: {selected_name}. Its held-out test metrics were: " + ", ".join(f"{key}={value:.4f}" for key, value in selected_metrics.items()) + ". These results describe one fixed split and do not establish perfect generalization.")
    best_test = comparison.loc[comparison["test_f1_malignant"].idxmax()]
    doc.add_paragraph(f"The highest malignant F1 on this test split belongs to {best_test['model']} ({best_test['test_f1_malignant']:.4f}); this test result does not override the training-only cross-validation selection, because using it to select a model would leak test information into model choice.")
    doc.add_heading("Confusion matrix and ROC curve", 1)
    doc.add_picture(str(confusion_path), width=Inches(5.6)); doc.add_paragraph("The confusion matrix shows held-out counts for each actual and predicted class. False negatives for the malignant class are particularly important to inspect in this educational setting.")
    doc.add_picture(str(roc_path), width=Inches(5.8)); doc.add_paragraph("The ROC curve summarizes ranking performance across thresholds; it does not choose an operating threshold or account for clinical harms and costs.")
    doc.add_heading("Feature interpretation", 1)
    doc.add_picture(str(feature_path), width=Inches(6.0))
    doc.add_paragraph("The plotted coefficients or impurity-based importances are model-specific associations. They are not causal explanations and can be unstable when features are correlated. Interpret them as a diagnostic of the fitted model, not a medical finding.")
    doc.add_heading("Strengths, limitations, ethics, and improvements", 1)
    doc.add_paragraph("Strengths include a public documented dataset, stratified splitting, preprocessing pipelines, cross-validation on training data, and a held-out comparison. Limitations include a modest historical sample, no external validation, correlated measurements, one random split, potential differences from current clinical populations and imaging workflows, and metrics that do not incorporate real decision costs. Medical labels and false negatives have serious consequences. Any real medical tool would require clinical validation, representative data, calibration, fairness assessment, privacy protections, regulatory review, and human oversight. Future work could use nested cross-validation, repeated external datasets, calibrated probabilities, threshold analysis with domain experts, and more robust feature interpretation.")
    doc.add_heading("Conclusion", 1)
    doc.add_paragraph("This project demonstrates a leakage-aware supervised classification workflow with a linear baseline, a nonlinear comparison, cross-validation, and held-out metrics. The computed results support learning about model evaluation, not clinical use.")
    doc.add_heading("Reproducibility code excerpts", 1)
    for text in [
        "X_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.20, stratify=y, random_state=42\n)",
        "pipeline = Pipeline([('preprocess', preprocessor), ('model', classifier)])\ncv_results = cross_validate(pipeline, X_train, y_train, cv=StratifiedKFold(5), scoring=scoring)",
    ]:
        p = doc.add_paragraph(); run = p.add_run(text); run.font.name = "Consolas"; run.font.size = Pt(9)
    doc.save(OUT / "report.docx")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data, source_description, sklearn_version = load_dataset()
    target = data["diagnosis"].copy()
    X = data.drop(columns="diagnosis")
    feature_names = list(X.columns)
    rows, columns = data.shape
    class_counts = target.value_counts().sort_index().to_dict()
    quality = {
        "rows": int(rows), "columns": int(columns),
        "feature_count": len(feature_names),
        "dtypes": {str(k): str(v) for k, v in data.dtypes.items()},
        "missing_feature_cells": int(X.isna().sum().sum()),
        "missing_by_column": {str(k): int(v) for k, v in data.isna().sum().items()},
        "duplicate_rows": int(data.duplicated().sum()),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "target": "diagnosis", "positive_class": POSITIVE_LABEL,
        "identifier_features_excluded": True,
        "dataset_description": source_description,
        "source_url": SOURCE_URL, "scikit_learn_version": sklearn_version,
    }
    (OUT / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    data.describe(include="all").transpose().to_csv(OUT / "feature_summary.csv")
    data.isna().sum().rename("missing_count").to_csv(OUT / "missing_values.csv")
    data.head(10).to_csv(OUT / "sample_records.csv", index=False)
    data.to_csv(OUT / "dataset_snapshot.csv", index=False)

    # Hold out the test data before any learned preprocessing is fitted.
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target
    )
    models = {
        "Logistic Regression": Pipeline([
            ("preprocess", build_preprocessor(feature_names, scale=True)),
            ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", build_preprocessor(feature_names, scale=False)),
            ("model", RandomForestClassifier(n_estimators=300, max_depth=None,
                min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=1)),
        ]),
    }
    scorer = {
        "accuracy": "accuracy",
        "precision_malignant": make_scorer(precision_score, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall_malignant": make_scorer(recall_score, pos_label=POSITIVE_LABEL, zero_division=0),
        "f1_malignant": make_scorer(f1_score, pos_label=POSITIVE_LABEL, zero_division=0),
        "roc_auc_malignant": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    fitted = {}
    test_rows = []
    prediction_store = {}
    for name, pipeline in models.items():
        result = cross_validate(pipeline, X_train, y_train, scoring=scorer, cv=cv,
                                return_train_score=False, n_jobs=1, error_score="raise")
        cv_rows.append({"model": name, **{f"cv_{metric}_mean": float(np.mean(result[f"test_{metric}"]))
                        for metric in scorer}, **{f"cv_{metric}_std": float(np.std(result[f"test_{metric}"]))
                        for metric in scorer}})
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        pred = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(POSITIVE_LABEL)]
        values = metric_values(y_test, pred, probabilities)
        test_rows.append({"model": name, **{f"test_{key}": value for key, value in values.items()}})
        prediction_store[name] = (pred, probabilities, values)

    cv_table = pd.DataFrame(cv_rows)
    test_table = pd.DataFrame(test_rows)
    comparison = cv_table.merge(test_table, on="model")
    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    # Select exclusively with training cross-validation, not the held-out test scores.
    selected_name = str(cv_table.loc[cv_table["cv_f1_malignant_mean"].idxmax(), "model"])
    selected_model = fitted[selected_name]
    predictions, probabilities, selected_metrics = prediction_store[selected_name]
    report_text = classification_report(y_test, predictions, labels=["benign", "malignant"],
                                        target_names=["benign", "malignant"], zero_division=0)
    (OUT / "classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame({"actual": y_test.to_numpy(), "predicted": predictions,
                  "malignant_probability": probabilities}, index=y_test.index).sort_index().to_csv(OUT / "test_predictions.csv", index_label="row_id")
    confusion_path = save_confusion_plot(y_test, predictions, ["benign", "malignant"])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    RocCurveDisplay.from_estimator(selected_model, X_test, y_test, pos_label=POSITIVE_LABEL, ax=ax, name=selected_name)
    ax.set_title(f"Held-out ROC curve: {selected_name}"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    roc_path = OUT / "roc_curve.png"; fig.tight_layout(); fig.savefig(roc_path, dpi=170, bbox_inches="tight"); plt.close(fig)
    final_estimator = selected_model.named_steps["model"]
    if hasattr(final_estimator, "feature_importances_"):
        weights = np.asarray(final_estimator.feature_importances_)
        title = "Random Forest impurity-based feature importance"
    elif hasattr(final_estimator, "coef_"):
        weights = np.abs(np.asarray(final_estimator.coef_)[0])
        title = "Absolute standardized Logistic Regression coefficients"
    else:
        weights = np.zeros(len(feature_names)); title = "Feature importance unavailable"
    top_idx = np.argsort(weights)[-15:]
    feature_importance = pd.DataFrame({"feature": np.array(feature_names)[top_idx], "importance": weights[top_idx]}).sort_values("importance")
    feature_importance.to_csv(OUT / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 7)); ax.barh(feature_importance.feature, feature_importance.importance, color="#3e7da8")
    ax.set(title=title, xlabel="Model-specific importance", ylabel="Feature")
    feature_path = OUT / "feature_importance.png"; fig.tight_layout(); fig.savefig(feature_path, dpi=170, bbox_inches="tight"); plt.close(fig)
    joblib.dump(selected_model, OUT / "best_model.joblib")
    run = {"selected_model": selected_name, "selection_metric": "mean 5-fold training CV malignant F1",
           "test_size": TEST_SIZE, "random_state": RANDOM_STATE, "positive_class": POSITIVE_LABEL,
           "test_metrics": selected_metrics, "scikit_learn_version": sklearn_version,
           "train_rows": len(X_train), "test_rows": len(X_test)}
    (OUT / "run_summary.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    write_report(data, source_description, SOURCE_URL, quality, cv_table, comparison,
                 selected_name, selected_metrics, report_text, confusion_path, roc_path,
                 feature_path, feature_names, selected_model)
    print(f"Dataset: {rows} rows, {len(feature_names)} features; train={len(X_train)}, test={len(X_test)}")
    print(f"Selected by CV malignant-class F1: {selected_name}")
    print(comparison.to_string(index=False))
    print(f"Artifacts written to {OUT}")


if __name__ == "__main__":
    main()
