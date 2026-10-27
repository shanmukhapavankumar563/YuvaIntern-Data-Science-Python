# Week 3: Unsupervised Learning and Clustering Analysis

## Objective

Apply K-Means to a public customer dataset, compare plausible cluster counts, visualize and profile the resulting groups, and document the workflow and limitations.

## Dataset and attribution

This project uses the Mall Customers teaching dataset: [MahimaSriK/Mall-Customer-Segmentation-Dataset](https://github.com/MahimaSriK/Mall-Customer-Segmentation-Dataset/blob/main/Mall_Customers.csv). The script downloads the CSV on first run and caches it at `data/raw/Mall_Customers.csv`. The file contains customer ID, gender, age, annual income (k$), and spending score (1–100). It is a small instructional dataset; the repository does not establish a representative sampling method.

## Feature and preprocessing choices

- K-Means uses age, annual income, and spending score. These numeric fields describe distinct customer characteristics.
- CustomerID is excluded because its numeric values identify rows and do not represent customer similarity.
- Gender is retained for descriptive cluster profiling only. It is not encoded into arbitrary numeric distances.
- Exact duplicate rows are counted and removed. Missing numeric feature values are median-imputed, and nonnumeric feature values are converted to missing and included in that imputation count. Other customer fields are retained for traceability.
- Negative age/income and spending scores outside 1–100 are treated as invalid values and median-imputed; IQR outliers are counted for review, not deleted.
- StandardScaler puts the three model features on a comparable scale before Euclidean-distance clustering.
- Potential extreme observations are not automatically removed; K-Means is sensitive to outliers, so inspect distributions and data-quality outputs.

## Model and evaluation

The workflow evaluates k values from 2 through 10 (bounded by the available row count), with k-means++ initialization, `n_init=20`, `max_iter=300`, and `random_state=42`. It records inertia and silhouette score for each candidate. The saved model uses the candidate with the highest silhouette score; the elbow plot is a secondary visual check, since its bend is subjective. Silhouette favors compact, well-separated groups and can prefer simple partitions; neither metric validates business usefulness. The selection rule is transparent and can be reconsidered using domain knowledge.

PCA maps the standardized three-feature data to two dimensions for an additional visualization. This projection may hide separations present in the full feature space. The income-versus-spending plot shows two original features, while age still contributes to the clustering.

## Setup and execution

Python 3.10 or later is recommended. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python clustering_analysis.py
```

On macOS/Linux, activate using `source .venv/bin/activate`. Internet access is needed for the initial dataset download unless the CSV has been placed in the cache path. The script produces data-dependent results and a Word report when run; no example metric is presented as an executed result.

## Generated outputs

The script writes outputs under `outputs/`:

- `report.docx` — methodology, evaluation, data-dependent cluster profiles, figures, limitations, and cautious application discussion.
- `elbow_method.png`, `silhouette_scores.png` — cluster-count diagnostics.
- `cluster_visualization.png`, `cluster_pca_visualization.png` — original feature and PCA views of assigned labels.
- `feature_distributions.png` — feature distribution review.
- `cluster_summary.csv`, `cluster_interpretations.csv` — original-unit cluster statistics and sample-relative summaries.
- `cluster_assignments.csv`, `cluster_centers_original_units.csv` — customer labels and centroids.
- `cluster_evaluation.csv`, `run_summary.json` — candidate metrics and selected configuration.
- `outlier_review.csv` — IQR thresholds and flagged feature counts.
- `data_quality.json`, `descriptive_statistics.csv`, `missing_values.csv`, `sample_records.csv` — source inspection and preprocessing details.
- `prepared_customers.csv`, `scaled_features.csv`, `pca_coordinates.csv` — analysis inputs and projection output.

## Internship instruction checklist

| Screenshot item | Project coverage |
|---|---|
| Apply unsupervised learning to public data | Public Mall Customers CSV and K-Means workflow |
| Segment and analyze group characteristics | Cluster assignment, original-unit summary table, and per-cluster relative descriptions |
| Use Python libraries such as scikit-learn | pandas, NumPy, matplotlib, seaborn, scikit-learn, python-docx |
| Detailed DOCX methodology, results, and interpretations | Generated `outputs/report.docx` |
| Code snippets, figures, and rationale | Script comments and report's feature, metric, and visualization explanations |
| Select a suitable public dataset | Dataset selected for its numeric age, income, and spending attributes; source linked above |
| Preprocess appropriately | Duplicate/missing checks, median imputation, ID exclusion, feature scaling |
| At least one algorithm and parameter choices | K-Means parameters documented in code, README, and report |
| Visualize clusters | Income/spending scatter and PCA projection, with labeled axes and cluster legend |
| Discuss cluster characteristics and possible implications | Profile tables plus sample-relative prose; applications framed as hypotheses |
| Evaluation clarity and accuracy | Inertia and silhouette candidate table/plots; limitations explained |
| Organized report and conceptual clarity | Sectioned generated DOCX and this README |
| Approximately 30–35 hours expected | Deliverables are provided; time spent is not inferred or claimed |

## Limitations

The data is small, the sampling process and score construction are not established in this source, and K-Means favors compact roughly spherical clusters. Results depend on features, scaling, outliers, and k. Silhouette and inertia are internal diagnostics rather than external validation. Clusters should support exploratory questions, not be treated as definitive customer identities or proof that a marketing action will work.
