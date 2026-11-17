"""Discover wholesale customer segments from spending data."""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler


class WholesaleCustomerClustering:
    """Store configuration and results for the clustering workflow."""

    SPENDING_COLUMNS = ("Fresh", "Milk", "Grocery", "Frozen",
                        "Detergents_Paper", "Delicassen")
    REQUIRED_COLUMNS = (*SPENDING_COLUMNS, "Channel", "Region")

    def __init__(self, k_min, k_max, random_state):
        # Preserve caller-supplied configuration without changing its values.
        self.k_min = k_min
        self.k_max = k_max
        self.random_state = random_state

        self.original_data = None
        self.spending_features = None
        self.transformed_features = None
        self.original_spending_features = None
        self.feature_medians = None
        self.scaler = None
        self.experiments = None
        self.final_model = None

        self.pca = None
        self.pca_data = None
        self.n_components = None
        self.explained_variance_ratios = None
        self.cumulative_explained_variance = None

        self.kmeans_results = []
        self.hierarchical_results = []
        self.best_method = None
        self.best_k = None
        self.final_labels = None
        self.cluster_agreement = None

        self.dbscan_results = []
        self.best_dbscan_params = None
        self.dbscan_labels = None

        self.cluster_profiles = None
        self.cluster_descriptions = {}

    def load_data(self, file_path="Wholesale_Customers.csv"):
        """Read a CSV, validating its schema before replacing existing state."""
        data = pd.read_csv(file_path)
        missing = sorted(set(self.REQUIRED_COLUMNS) - set(data.columns))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        if data.empty:
            raise ValueError("The dataset must contain customers.")
        self.__init__(self.k_min, self.k_max, self.random_state)
        self.original_data = data.copy()
        return self.original_data

    def prepare_features(self):
        """Median-impute spending, log-transform, and standardize it."""
        if self.original_data is None:
            raise ValueError("Load data first.")
        spending = self.original_data.loc[:, self.SPENDING_COLUMNS].apply(
            pd.to_numeric, errors="coerce")
        if np.isinf(spending.to_numpy()).any() or (spending < 0).any().any():
            raise ValueError("Spending must be finite and nonnegative.")
        self.feature_medians = spending.median()
        if self.feature_medians.isna().any():
            raise ValueError("Every spending column needs at least one numeric value.")
        self.original_spending_features = spending.copy()
        self.spending_features = spending.fillna(self.feature_medians)
        self.scaler = StandardScaler()
        self.transformed_features = self.scaler.fit_transform(
            np.log1p(self.spending_features))
        return self.transformed_features

    def apply_pca(self):
        """Choose the smallest component count reaching 90% variance."""
        if self.transformed_features is None:
            raise ValueError("Prepare features first.")
        if len(self.transformed_features) < 3 or not np.any(
                np.var(self.transformed_features, axis=0) > 0):
            raise ValueError("PCA requires at least three customers and varying spending.")
        full = PCA(svd_solver="full").fit(self.transformed_features)
        self.n_components = int(np.searchsorted(
            np.cumsum(full.explained_variance_ratio_), 0.9) + 1)
        self.pca = PCA(n_components=self.n_components, svd_solver="full")
        self.pca_data = self.pca.fit_transform(self.transformed_features)
        self.explained_variance_ratios = self.pca.explained_variance_ratio_.copy()
        self.cumulative_explained_variance = np.cumsum(self.explained_variance_ratios)
        return self.pca_data

    def _k_values(self):
        if self.pca_data is None:
            raise ValueError("Apply PCA first.")
        if (isinstance(self.k_min, bool) or isinstance(self.k_max, bool)
                or not isinstance(self.k_min, (int, np.integer))
                or not isinstance(self.k_max, (int, np.integer))
                or not 2 <= self.k_min <= self.k_max < len(self.pca_data)):
            raise ValueError("K must satisfy 2 <= k_min <= k_max < customer count.")
        return range(self.k_min, self.k_max + 1)

    def _metrics(self, labels, data=None):
        data = self.pca_data if data is None else data
        if not 2 <= len(np.unique(labels)) < len(labels):
            return dict(silhouette=np.nan, davies_bouldin=np.nan,
                        calinski_harabasz=np.nan)
        return dict(silhouette=silhouette_score(data, labels),
                    davies_bouldin=davies_bouldin_score(data, labels),
                    calinski_harabasz=calinski_harabasz_score(data, labels))

    def run_kmeans_experiments(self):
        rows = []
        for k in self._k_values():
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(self.pca_data)
            rows.append(dict(method="kmeans", k=k, sse=model.inertia_,
                             **self._metrics(labels)))
        self.kmeans_results = pd.DataFrame(rows)
        return self.kmeans_results

    def run_hierarchical_experiments(self):
        rows = []
        for k in self._k_values():
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(
                self.pca_data)
            rows.append(dict(method="hierarchical", k=k, sse=np.nan,
                             **self._metrics(labels)))
        self.hierarchical_results = pd.DataFrame(rows)
        return self.hierarchical_results

    def compare_methods(self):
        """Combine all experiments using one consistent metric schema."""
        if len(self.kmeans_results) == 0 or len(self.hierarchical_results) == 0:
            raise ValueError("Run both clustering experiment methods first.")
        self.experiments = pd.concat(
            [pd.DataFrame(self.kmeans_results), pd.DataFrame(self.hierarchical_results)],
            ignore_index=True).sort_values(
                ["silhouette", "davies_bouldin", "calinski_harabasz"],
                ascending=[False, True, False], kind="stable")
        return self.experiments

    def select_best_clustering(self):
        """Rank by silhouette, Davies–Bouldin, then Calinski–Harabasz."""
        self.compare_methods()
        valid = self.experiments.dropna(subset=["silhouette"])
        if valid.empty:
            raise ValueError("No configuration produced valid internal metrics.")
        winner = valid.iloc[0]
        self.best_method, self.best_k = winner["method"], int(winner["k"])
        return self.best_method, self.best_k

    def fit_final_clustering(self):
        """Fit the selected configuration on the same PCA representation."""
        if self.best_method not in ("kmeans", "hierarchical") or self.best_k is None:
            raise ValueError("Select a configuration first.")
        self.final_model = (KMeans(n_clusters=self.best_k, n_init=10,
                                  random_state=self.random_state)
                            if self.best_method == "kmeans" else
                            AgglomerativeClustering(n_clusters=self.best_k, linkage="ward"))
        self.final_labels = self.final_model.fit_predict(self.pca_data)
        return self.final_labels

    def measure_cluster_agreement(self):
        if self.best_k is None:
            raise ValueError("Select a configuration first.")
        km = KMeans(n_clusters=self.best_k, n_init=10,
                    random_state=self.random_state).fit_predict(self.pca_data)
        ward = AgglomerativeClustering(n_clusters=self.best_k, linkage="ward").fit_predict(
            self.pca_data)
        self.cluster_agreement = adjusted_rand_score(km, ward)
        return self.cluster_agreement

    def detect_outliers_with_dbscan(self, parameter_grid=None):
        """Search the required eight trials; maximize non-noise silhouette.

        If no trial has a valid silhouette, minimize noise, preserving grid order
        for remaining ties. Pass eight (eps, min_samples) pairs to override.
        """
        if self.pca_data is None:
            raise ValueError("Apply PCA first.")
        grid = list(parameter_grid) if parameter_grid is not None else [
            (eps, minimum) for eps in (0.5, 0.8, 1.0, 1.2) for minimum in (5, 10)]
        if len(grid) != 8 or any(eps <= 0 or not isinstance(m, int) or m < 1
                                 for eps, m in grid):
            raise ValueError("Provide eight positive (eps, integer min_samples) pairs.")
        rows, candidates = [], []
        for eps, minimum in grid:
            labels = DBSCAN(eps=eps, min_samples=minimum).fit_predict(self.pca_data)
            mask = labels != -1
            count = len(np.unique(labels[mask]))
            score = (silhouette_score(self.pca_data[mask], labels[mask])
                     if 2 <= count < mask.sum() else np.nan)
            rows.append(dict(eps=eps, min_samples=minimum, clusters=count,
                             noise_count=int((~mask).sum()),
                             noise_percentage=float((~mask).mean() * 100),
                             silhouette=score))
            candidates.append(labels)
        self.dbscan_results = pd.DataFrame(rows)
        index = self.dbscan_results.sort_values(
            ["silhouette", "noise_percentage"], ascending=[False, True],
            na_position="last", kind="stable").index[0]
        row = self.dbscan_results.loc[index]
        self.best_dbscan_params = dict(eps=float(row["eps"]),
                                       min_samples=int(row["min_samples"]))
        self.dbscan_labels = candidates[index]
        return self.dbscan_results

    def create_cluster_profiles(self):
        if self.final_labels is None or self.dbscan_labels is None:
            raise ValueError("Fit final clustering and DBSCAN first.")
        rows = []
        for label in np.unique(self.final_labels):
            mask = self.final_labels == label
            row = dict(cluster=int(label), customer_count=int(mask.sum()),
                       dbscan_noise_percentage=float(
                           (self.dbscan_labels[mask] == -1).mean() * 100))
            row.update(self.original_spending_features.loc[mask].median().to_dict())
            for column in ("Channel", "Region"):
                modes = self.original_data.loc[mask, column].mode()
                row[column] = modes.iloc[0] if not modes.empty else None
            rows.append(row)
        self.cluster_profiles = pd.DataFrame(rows).set_index("cluster")
        return self.cluster_profiles

    def describe_clusters(self):
        """Describe medians relative to overall medians using ±20% bands."""
        if self.cluster_profiles is None:
            raise ValueError("Build profiles first.")
        overall = self.original_spending_features.median()
        self.cluster_descriptions = {}
        for label, row in self.cluster_profiles.iterrows():
            levels = {"high": [], "low": [], "typical": []}
            unavailable = []
            for column in self.SPENDING_COLUMNS:
                value, baseline = row[column], overall[column]
                if pd.isna(value):
                    unavailable.append(column)
                    continue
                level = ("high" if value > baseline * 1.2 else
                         "low" if value < baseline * 0.8 else "typical")
                levels[level].append(column)
            patterns = []
            if len(levels["high"]) >= 4:
                patterns.append("High overall spending")
            elif len(levels["low"]) >= 4:
                patterns.append("Low overall spending")
            # A focus must exceed the cluster's general relative spending level;
            # uniformly high spenders should not be called focused on everything.
            ratios = row.loc[list(self.SPENDING_COLUMNS)].astype(float).div(
                overall.replace(0, np.nan))
            relative_level = ratios.median()
            focused = {column for column in levels["high"]
                       if (overall[column] == 0 or
                           ratios[column] > relative_level * 1.2)}
            if "Fresh" in focused:
                patterns.append("Fresh-product focused")
            if {"Grocery", "Detergents_Paper"}.issubset(focused):
                patterns.append("Grocery and detergent focused")
            if "Frozen" in focused:
                patterns.append("Frozen-product focused")
            if unavailable:
                patterns.append("No observed spending: " + ", ".join(unavailable))
            self.cluster_descriptions[label] = "; ".join(patterns + [
                f"{level.capitalize()} spending: {', '.join(columns)}"
                for level, columns in levels.items() if columns]) + "."
        return self.cluster_descriptions

    def run(self, file_path="Wholesale_Customers.csv", dbscan_parameter_grid=None):
        """Run the full pipeline; Channel and Region are profiling metadata only."""
        self.load_data(file_path)
        self.prepare_features()
        self.apply_pca()
        self.run_kmeans_experiments()
        self.run_hierarchical_experiments()
        self.select_best_clustering()
        self.fit_final_clustering()
        self.measure_cluster_agreement()
        self.detect_outliers_with_dbscan(dbscan_parameter_grid)
        self.create_cluster_profiles()
        self.describe_clusters()
        if self.cumulative_explained_variance[-1] < 0.9:
            raise RuntimeError("PCA retained less than 90% variance.")
        if len(self.final_labels) != len(self.original_data):
            raise RuntimeError("Every customer must have a final label.")
        results = {name: getattr(self, name) for name in (
            "original_data", "original_spending_features", "spending_features",
            "transformed_features", "pca_data",
            "n_components", "explained_variance_ratios", "cumulative_explained_variance",
            "kmeans_results", "hierarchical_results", "experiments", "best_method",
            "best_k", "final_labels", "cluster_agreement", "dbscan_results",
            "best_dbscan_params", "dbscan_labels", "cluster_profiles",
            "cluster_descriptions")}
        results.update(number_of_customers=len(self.original_data),
                       pca_components=self.n_components,
                       clustering_comparison=self.compare_methods(),
                       dbscan_result={"parameters": self.best_dbscan_params,
                                      "labels": self.dbscan_labels,
                                      "experiments": self.dbscan_results})
        return results

    # Backward-compatible names from the initial implementation.
    test_kmeans = run_kmeans_experiments
    test_hierarchical = run_hierarchical_experiments
    measure_algorithm_agreement = measure_cluster_agreement
    search_dbscan = detect_outliers_with_dbscan
    build_customer_profiles = create_cluster_profiles
    generate_descriptions = describe_clusters

    def select_best_configuration(self):
        """Legacy helper selecting and fitting the final configuration."""
        self.select_best_clustering()
        self.fit_final_clustering()
        return self.experiments
