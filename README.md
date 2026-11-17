# Wholesale Customer Structure Discovery

Discover customer segments from annual wholesale purchasing data using unsupervised learning. The workflow prepares spending features, reduces their dimensions with PCA, compares K-Means with Ward hierarchical clustering, identifies unusual customers with DBSCAN, and produces spending profiles and behavioral descriptions.

There is no target label. Model selection uses internal clustering metrics, not Channel, Region, or known customer classes.

## Project files

| File | Purpose |
| --- | --- |
| [wholesale_customer_clustering.py](./wholesale_customer_clustering.py) | Class implementing the complete workflow |
| [test_wholesale_customer_clustering.py](./test_wholesale_customer_clustering.py) | Synthetic tests and controlled edge-case checks |
| [requirements.txt](./requirements.txt) | NumPy, pandas, and scikit-learn dependencies |

The real `Wholesale_Customers.csv` is not included. Supply your own copy before running the real-data workflow. Tests generate their inputs in memory and do not require this file.

## View this documentation

Open this folder in VS Code, open `README.md`, and choose **Open Preview** from the editor context menu. You can also read the Markdown source directly or view it rendered on a repository hosting service.

The project is a Python library: it has no web interface or dashboard. Results appear when you print them from a Python script or notebook, or open exported CSV files in a spreadsheet application. Running the class module directly does not execute an analysis because it only defines the class.

## Install and run

Open a terminal in the project folder. Use a Python version supported by your installed dependency releases; the project has been tested locally using Python 3.14. No minimum Python version is formally declared.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` directly in place of `python` in subsequent commands. In VS Code, select the environment's interpreter through **Python: Select Interpreter**.

### Create a runner

Place `Wholesale_Customers.csv` in the project folder. Create a file called `run_clustering.py` alongside the class module and paste this code:

```python
from wholesale_customer_clustering import WholesaleCustomerClustering

model = WholesaleCustomerClustering(k_min=2, k_max=10, random_state=42)
results = model.run("Wholesale_Customers.csv")

print("Customers:", results["number_of_customers"])
print("PCA components:", results["pca_components"])
print("Retained variance:", results["cumulative_explained_variance"][-1])
print("Best method:", results["best_method"])
print("Best K:", results["best_k"])
print("Algorithm agreement (ARI):", results["cluster_agreement"])

print("\nClustering comparison:")
print(results["clustering_comparison"].to_string(index=False))
print("\nDBSCAN experiments:")
print(results["dbscan_results"].to_string(index=False))
print("Selected DBSCAN parameters:", results["best_dbscan_params"])
print("\nCluster profiles:")
print(results["cluster_profiles"].to_string())
print("\nCluster descriptions:")
for label, description in results["cluster_descriptions"].items():
    print(f"Cluster {label}: {description}")
```

Run it:

```bash
python run_clustering.py
```

`model.run()` without a path also looks for `Wholesale_Customers.csv` in the terminal's current working directory. Supply an absolute path when running from another folder. The workflow returns results in memory; it does not automatically print or save them.

## Use your own dataset

Use a comma-separated CSV with one customer per row and these exact, case-sensitive headers:

```csv
Channel,Region,Fresh,Milk,Grocery,Frozen,Detergents_Paper,Delicassen
1,3,12669,9656,7561,214,2674,1338
2,1,7057,9810,9568,1762,3293,1776
1,2,6353,8808,7684,2405,3516,7844
```

These three rows illustrate the format only; they are not an example analysis or a sufficient dataset for the runner's `k_max=10`.

| Columns | Role and requirements |
| --- | --- |
| Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen | The only clustering features; nonnegative spending measured over a consistent period and in consistent units |
| Channel, Region | Required headers; used only for the most common value within each final segment |
| Additional columns, such as CustomerID | Preserved in the full dataset, excluded from all clustering features and model selection |

Data rules:

- At least three customers and some variation in spending are required.
- K bounds must be integers satisfying `2 <= k_min <= k_max < number_of_customers`. For 8 customers, for example, `k_max` cannot exceed 7.
- Spending strings convertible to numbers are accepted. Missing or unparseable entries are converted to missing values and filled with their column median for clustering.
- Every spending column must contain at least one valid numerical observation. Entirely missing columns are rejected.
- Negative spending and positive or negative infinity are rejected. Zero spending is allowed.
- Original numerical spending is stored separately before imputation for profiles; missing observations are excluded from profile medians.
- Channel and Region can contain missing values. A cluster with no observed value for a metadata column receives `None` as its mode.
- Row order is preserved. Labels correspond to CSV rows in that order; an identifier column makes exported assignments easier to trace.

For a custom path:

```python
model = WholesaleCustomerClustering(2, 6, 42)
results = model.run("data/my_customers.csv")
```

If your columns have other names, rename them to the required schema before running. If Channel or Region is genuinely unavailable, add that column with empty values rather than inventing metadata. A dataset with unrelated feature meanings requires adapting the feature schema and descriptions in the implementation.

## What happens during the workflow

| Stage / method | Behavior |
| --- | --- |
| `__init__(k_min, k_max, random_state)` | Preserves configuration and initializes storage |
| `load_data(file_path)` | Reads the full CSV, validates required columns and nonempty data, and resets results from any previous dataset |
| `prepare_features()` | Selects the six spending columns, converts to numeric, median-imputes, applies `log1p`, and standardizes |
| `apply_pca()` | Finds the smallest component count whose cumulative explained variance reaches at least 90%, then stores the PCA representation |
| `run_kmeans_experiments()` | Fits K-Means for every K in the inclusive range, using `random_state` and `n_init=10` |
| `run_hierarchical_experiments()` | Fits agglomerative clustering with Ward linkage for every K in the same range |
| `select_best_clustering()` | Selects method and K using the metric ranking below; does not fit the final model |
| `fit_final_clustering()` | Fits the selected configuration and stores one final label per customer |
| `measure_cluster_agreement()` | Independently fits both algorithms at the selected K and returns their ARI |
| `detect_outliers_with_dbscan()` | Evaluates eight density configurations and stores the selected DBSCAN labels |
| `create_cluster_profiles()` | Summarizes final segments using observed, untransformed spending and metadata |
| `describe_clusters()` | Produces descriptions from measured spending comparisons |
| `compare_methods()` | Combines both experiment tables and sorts them using the selection ranking |
| `run(file_path)` | Executes all stages in order and returns the result dictionary |

Log transformation reduces the influence of strong positive skew. Standardization prevents a large numerical spending scale from dominating distances. PCA retains most variance while removing redundant dimensions. All clustering fits and internal clustering metrics use the same PCA representation.

You can inspect intermediate outputs by calling the methods individually in the order above. Loading another dataset clears the previous analysis. For a fresh analysis after editing input data or configuration, call `run()` again rather than mixing results from different stages.

## How the winner is selected

Configurations are ranked lexicographically:

1. Highest Silhouette Score.
2. Lowest Davies–Bouldin Index when silhouettes are exactly equal.
3. Highest Calinski–Harabasz Score when both previous metrics are exactly equal.

SSE, ARI, DBSCAN results, Channel, and Region do not select the final clustering. If every ranking metric ties, the implementation preserves experiment order. Configurations without a valid silhouette are excluded; if none remain, selection raises an error.

| Metric | Interpretation |
| --- | --- |
| `silhouette` | Higher is better; usually interpreted on a −1 to 1 scale. Measures separation and cohesion |
| `davies_bouldin` | Lower is better. Measures similarity between clusters relative to their spread |
| `calinski_harabasz` | Higher is better. Compares between-cluster and within-cluster dispersion |
| `sse` | K-Means sum of squared distances to centers; lower means tighter fit, but it typically decreases as K increases. Not used for final selection |
| `cluster_agreement` | Adjusted Rand Index between K-Means and Ward partitions at the selected K. 1 means identical partitions up to label renaming, near 0 indicates chance-level agreement, and negative values are possible |

ARI measures agreement, not accuracy against ground truth. Strong internal scores do not by themselves prove a segment is commercially useful.

## DBSCAN and unusual customers

The default search evaluates the Cartesian product:

```text
eps:         0.5, 0.8, 1.0, 1.2
min_samples: 5, 10
```

`eps` is a neighborhood radius in PCA space; `min_samples` controls neighborhood density. Each experiment records `eps`, `min_samples`, `clusters` (excluding noise), `noise_count`, `noise_percentage`, and `silhouette`.

Silhouette is calculated only on non-noise customers and only when their cluster count is at least two and smaller than their customer count. Otherwise it is `NaN`.

Selection chooses the highest valid non-noise silhouette, breaking silhouette ties by lower noise percentage. If all silhouettes are unavailable, it chooses the lowest noise percentage. Remaining ties preserve search order.

DBSCAN label `-1` means noise. Other labels identify DBSCAN groups. Noise customers still retain their final K-Means or Ward segment label; DBSCAN does not replace final segmentation. Noise indicates an unusual local density pattern, not automatically an error or a bad customer. A high silhouette with high noise can describe only a small retained subset, so inspect both measures.

The API optionally accepts `dbscan_parameter_grid` as exactly eight `(eps, min_samples)` pairs. Leave this argument unset to comply with the assignment's specified grid.

## Results you will receive

Exact component counts, winning algorithm, K, metrics, assignments, and descriptions depend on the supplied dataset. No numerical real-data results are claimed here because the real CSV has not been supplied.

### Main result keys

| Key | Contents |
| --- | --- |
| `number_of_customers` | Number of input rows |
| `pca_components` | Selected PCA component count |
| `clustering_comparison` | One row per method and K with columns `method`, `k`, `sse`, `silhouette`, `davies_bouldin`, `calinski_harabasz` |
| `best_method` | `"kmeans"` or `"hierarchical"` |
| `best_k` | Selected cluster count |
| `cluster_agreement` | ARI between independently fitted partitions |
| `dbscan_result` | Dictionary with `parameters`, `labels`, and `experiments` |
| `cluster_profiles` | DataFrame indexed by final cluster label |
| `cluster_descriptions` | Dictionary mapping final cluster labels to behavioral text |

Hierarchical `sse` is unavailable (`NaN`). With K from 2 through 10, the combined comparison contains 18 experiments: nine for each algorithm. The default DBSCAN table contains eight experiments.

### Additional outputs for inspection

| Key | Contents |
| --- | --- |
| `original_data` | Complete input DataFrame, including extra columns |
| `original_spending_features` | Six numeric spending columns before imputation, log transformation, or scaling |
| `spending_features` | Median-imputed spending values in original units |
| `transformed_features` | Log-transformed, standardized array of shape `(customers, 6)` |
| `pca_data` | PCA array of shape `(customers, pca_components)` |
| `n_components` | Same count as `pca_components` |
| `explained_variance_ratios` | Variance fraction for each retained component |
| `cumulative_explained_variance` | Running variance fractions; final value is at least 0.90 |
| `kmeans_results`, `hierarchical_results` | Individual experiment DataFrames |
| `experiments` | Combined, ranked comparison table |
| `final_labels` | Final segment assignment for every input row |
| `dbscan_results` | All DBSCAN experiments |
| `best_dbscan_params` | Selected `eps` and `min_samples` |
| `dbscan_labels` | Selected DBSCAN assignment for every input row |

The fitted scaler, PCA object, and final clustering model remain available as `model.scaler`, `model.pca`, and `model.final_model`.

### Reading profiles and descriptions

Each profile contains `customer_count`, the six spending medians, `dbscan_noise_percentage`, and the most common `Channel` and `Region`. Spending medians use observed values in their original units, excluding missing entries. If a cluster has no observed value for a spending category, that median remains `NaN`. Metadata mode ties use the first value returned by pandas' mode calculation.

Descriptions compare cluster medians with overall observed customer medians:

- **High:** greater than 120% of the overall median.
- **Low:** less than 80% of the overall median.
- **Typical:** within those bounds, including the boundaries.
- **High/low overall spending:** at least four of six categories are high/low.
- **Category focus:** the category is high and its cluster-to-overall median ratio exceeds the median of available category ratios by more than 20%.
- **Grocery and detergent focused:** both categories satisfy that focus rule.
- **Zero overall median:** a positive cluster median is high and qualifies as focused; zero baselines are excluded from the relative-ratio median.
- **No observed spending:** the cluster has no observed value for that category; it is not described as typical.

Descriptions can include Fresh, Frozen, or Grocery and detergent focus, along with category-level spending assessments. These are explicit descriptive heuristics, not trained labels. Cluster numbers are arbitrary identifiers: cluster 0 is not inherently the lowest-spending group, and numbering may change between analyses.

## Export and view the results

Add this after the runner's analysis code to create files you can open in VS Code or a spreadsheet application:

```python
from pathlib import Path
import json

output = Path("results")
output.mkdir(exist_ok=True)

results["clustering_comparison"].to_csv(output / "clustering_comparison.csv", index=False)
results["dbscan_results"].to_csv(output / "dbscan_experiments.csv", index=False)
results["cluster_profiles"].to_csv(output / "cluster_profiles.csv")

customers = results["original_data"].copy()
customers["Cluster"] = results["final_labels"]
customers["DBSCAN_Label"] = results["dbscan_labels"]
customers["Is_DBSCAN_Noise"] = results["dbscan_labels"] == -1
customers.to_csv(output / "customer_assignments.csv", index=False)

descriptions = {str(label): text
                for label, text in results["cluster_descriptions"].items()}
(output / "cluster_descriptions.json").write_text(
    json.dumps(descriptions, indent=2), encoding="utf-8"
)
```

The entire result dictionary is not directly JSON serializable because it includes pandas DataFrames and NumPy arrays. Export tables as CSV, or explicitly convert arrays and numerical scalars if building a JSON report. Re-running this export overwrites files with the same names.

## Run the checks

From the project folder with dependencies installed:

```bash
python -m unittest -v
```

The current suite contains 18 tests covering CSV validation, numeric conversion, imputation, preservation of original values, standardization, minimal PCA component selection, independently recalculated metrics, selection tie-breaks, separate final fitting, ARI, the exact DBSCAN grid, valid selection and noise fallback, profiles, descriptions, metadata independence, prerequisite errors, minimum dataset size, and state reset on reload.

Passing synthetic tests verifies implementation behavior; running your actual CSV is still necessary to assess its customer structure.

## Troubleshooting and practical limits

| Symptom | Action |
| --- | --- |
| `FileNotFoundError` | Check the CSV path and terminal working directory; use an absolute path if needed |
| `ModuleNotFoundError` | Activate the intended environment and install dependencies using that interpreter's `python -m pip` |
| Missing required columns | Match all eight header names exactly, including capitalization and underscores |
| Dataset must contain customers | Supply at least one data row; the full workflow requires at least three |
| Every spending column needs a numeric value | Repair or populate the entirely missing/unparseable category |
| Spending must be finite and nonnegative | Correct negative or infinite entries before running |
| PCA requires varying spending | Identical customer spending profiles cannot support this workflow |
| K bounds error | Use integer bounds with `2 <= k_min <= k_max < customer count` |
| K-Means reports fewer distinct clusters than requested | Check duplicated spending rows and whether the K range is too large for the distinct data points |
| `NaN` DBSCAN silhouettes | The configurations did not produce enough valid non-noise groups to calculate silhouette; inspect noise percentages and the fallback selection |
| A method asks for an earlier stage | Use `run()` or call the public methods in workflow order |

The analysis describes the supplied batch. It does not implement automatic assignment of future customers: Ward clustering has no native `predict()` method, and a deployment workflow would need a separate assignment strategy. Changing the dataset refits medians, scaling, PCA, and the clusters. A fixed `random_state` makes K-Means repeatable within a consistent software environment; unpinned dependencies can change numerical results across versions.

Large datasets may require substantial memory and time, particularly for hierarchical clustering and silhouette evaluation. The current project targets batch exploration rather than streaming or large-scale production serving. No plots or Matplotlib installation are required.
