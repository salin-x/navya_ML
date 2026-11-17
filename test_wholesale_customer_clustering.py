"""Synthetic checks; no external customer dataset is required."""
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

from wholesale_customer_clustering import WholesaleCustomerClustering as Workflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        spending = np.expm1(rng.normal(
            np.repeat([3., 6., 9.], 20)[:, None], 0.2, (60, 6)))
        self.data = pd.DataFrame(spending, columns=Workflow.SPENDING_COLUMNS)
        self.data["Channel"] = rng.integers(1, 3, 60)
        self.data["Region"] = rng.integers(1, 4, 60)

    def csv(self, data=None):
        return StringIO((self.data if data is None else data).to_csv(index=False))

    def test_loading(self):
        workflow = Workflow(2, 4, 42)
        self.assertEqual((workflow.k_min, workflow.k_max, workflow.random_state), (2, 4, 42))
        pd.testing.assert_frame_equal(workflow.load_data(self.csv()), self.data)
        with self.assertRaisesRegex(ValueError, "Region"):
            workflow.load_data(self.csv(self.data.drop(columns="Region")))
        with self.assertRaisesRegex(ValueError, "contain customers"):
            workflow.load_data(self.csv(self.data.iloc[:0]))

    def test_preprocessing(self):
        self.data.loc[0, "Milk"] = np.nan
        workflow = Workflow(2, 4, 42)
        workflow.load_data(self.csv())
        transformed = workflow.prepare_features()
        self.assertEqual(workflow.spending_features.loc[0, "Milk"], self.data.Milk.median())
        np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(transformed.std(axis=0), 1)
        workflow.apply_pca()
        full = PCA().fit(transformed)
        self.assertGreaterEqual(workflow.cumulative_explained_variance[-1], .9)
        if workflow.n_components > 1:
            self.assertLess(full.explained_variance_ratio_[:workflow.n_components-1].sum(), .9)

    def test_full_pipeline_and_metadata_independence(self):
        first = Workflow(2, 4, 42).run(self.csv())
        self.data["Channel"] = 999
        self.data["Region"] = "changed"
        second = Workflow(2, 4, 42).run(self.csv())
        for key in ("pca_data", "final_labels", "dbscan_labels"):
            np.testing.assert_array_equal(first[key], second[key])
        for key in ("kmeans_results", "hierarchical_results", "experiments", "dbscan_results"):
            pd.testing.assert_frame_equal(first[key], second[key])
        self.assertEqual(len(first["experiments"]), 6)
        self.assertEqual(len(first["dbscan_results"]), 8)
        self.assertTrue(first["hierarchical_results"].sse.isna().all())
        self.assertEqual(len(first["final_labels"]), 60)
        self.assertEqual(first["cluster_profiles"].customer_count.sum(), 60)
        self.assertGreaterEqual(first["cluster_agreement"], -1)
        self.assertLessEqual(first["cluster_agreement"], 1)
        winner = first["experiments"].iloc[0]
        self.assertEqual((first["best_method"], first["best_k"]), (winner.method, winner.k))

    def test_invalid_spending_and_k(self):
        workflow = Workflow(2, 60, 42)
        workflow.load_data(self.csv())
        workflow.prepare_features()
        workflow.apply_pca()
        with self.assertRaisesRegex(ValueError, "K must"):
            workflow.test_kmeans()
        self.data["Milk"] = np.nan
        workflow.load_data(self.csv())
        with self.assertRaisesRegex(ValueError, "at least one numeric"):
            workflow.prepare_features()

    def prepared(self):
        workflow = Workflow(2, 4, 42)
        workflow.load_data(self.csv())
        workflow.prepare_features()
        workflow.apply_pca()
        return workflow

    def test_experiments_against_independent_calculations(self):
        workflow = self.prepared()
        for method, model_class in ((workflow.run_kmeans_experiments, KMeans),
                                    (workflow.run_hierarchical_experiments,
                                     AgglomerativeClustering)):
            table = method()
            self.assertEqual(list(table.k), [2, 3, 4])
            for row in table.itertuples():
                options = (dict(random_state=42, n_init=10) if model_class is KMeans
                           else dict(linkage="ward"))
                model = model_class(n_clusters=row.k, **options)
                labels = model.fit_predict(workflow.pca_data)
                self.assertAlmostEqual(row.silhouette, silhouette_score(workflow.pca_data, labels))
                self.assertAlmostEqual(row.davies_bouldin,
                                       davies_bouldin_score(workflow.pca_data, labels))
                self.assertAlmostEqual(row.calinski_harabasz,
                                       calinski_harabasz_score(workflow.pca_data, labels))
                if model_class is KMeans:
                    self.assertAlmostEqual(row.sse, model.inertia_)

    def test_selection_tiebreaks_and_separate_fit(self):
        for scores, expected in [
            ((.8, 2., 10., .7, 1., 20.), "kmeans"),
            ((.8, 2., 10., .8, 1., 20.), "hierarchical"),
            ((.8, 1., 10., .8, 1., 20.), "hierarchical"),
        ]:
            workflow = self.prepared()
            workflow.kmeans_results = pd.DataFrame([dict(
                method="kmeans", k=2, silhouette=scores[0], davies_bouldin=scores[1],
                calinski_harabasz=scores[2], sse=1.)])
            workflow.hierarchical_results = pd.DataFrame([dict(
                method="hierarchical", k=3, silhouette=scores[3],
                davies_bouldin=scores[4], calinski_harabasz=scores[5], sse=np.nan)])
            workflow.select_best_clustering()
            self.assertEqual(workflow.best_method, expected)
            self.assertIsNone(workflow.final_labels)
            self.assertEqual(len(workflow.fit_final_clustering()), 60)
            km = KMeans(n_clusters=workflow.best_k, random_state=42, n_init=10).fit_predict(
                workflow.pca_data)
            ward = AgglomerativeClustering(n_clusters=workflow.best_k).fit_predict(
                workflow.pca_data)
            self.assertAlmostEqual(workflow.measure_cluster_agreement(),
                                   adjusted_rand_score(km, ward))

    def test_dbscan_grid_scores_and_selection(self):
        workflow = self.prepared()
        table = workflow.detect_outliers_with_dbscan()
        self.assertEqual(list(zip(table.eps, table.min_samples)),
                         [(e, m) for e in (.5, .8, 1., 1.2) for m in (5, 10)])
        mask = workflow.dbscan_labels != -1
        valid = table.dropna(subset=["silhouette"])
        if not valid.empty:
            self.assertAlmostEqual(valid.silhouette.max(), silhouette_score(
                workflow.pca_data[mask], workflow.dbscan_labels[mask]))
        self.assertAlmostEqual((~mask).mean()*100, table.loc[
            (table.eps == workflow.best_dbscan_params["eps"]) &
            (table.min_samples == workflow.best_dbscan_params["min_samples"]),
            "noise_percentage"].iloc[0])

    def test_dbscan_all_noise_and_one_cluster_fallback(self):
        workflow = self.prepared()
        labels = [np.full(60, -1) for _ in range(8)]
        labels[3] = np.r_[np.zeros(50, dtype=int), np.full(10, -1)]
        with patch("wholesale_customer_clustering.DBSCAN") as mock:
            mock.return_value.fit_predict.side_effect = labels
            table = workflow.detect_outliers_with_dbscan()
        self.assertTrue(table.silhouette.isna().all())
        self.assertEqual(workflow.best_dbscan_params, dict(eps=.8, min_samples=10))
        np.testing.assert_array_equal(workflow.dbscan_labels, labels[3])

    def test_profiles_use_observed_spending_and_descriptions_ignore_ids(self):
        workflow = self.prepared()
        workflow.final_labels = np.repeat([17, 29, 83], 20)
        workflow.dbscan_labels = np.r_[np.full(10, -1), np.zeros(50, dtype=int)]
        workflow.original_spending_features.loc[:4, "Milk"] = np.nan
        profiles = workflow.create_cluster_profiles()
        for label in (17, 29, 83):
            mask = workflow.final_labels == label
            for column in Workflow.SPENDING_COLUMNS:
                self.assertAlmostEqual(profiles.loc[label, column],
                    workflow.original_spending_features.loc[mask, column].median())
            for column in ("Channel", "Region"):
                self.assertEqual(profiles.loc[label, column],
                                 self.data.loc[mask, column].mode().iloc[0])
        self.assertEqual(profiles.loc[17, "dbscan_noise_percentage"], 50)
        descriptions = workflow.describe_clusters().copy()
        self.assertIn("Low overall spending", descriptions[17])
        self.assertIn("High overall spending", descriptions[83])
        workflow.cluster_profiles = profiles.rename(index={17: 1, 29: 2, 83: 3})
        renamed = workflow.describe_clusters()
        self.assertEqual(descriptions[17], renamed[1])
        self.assertEqual(descriptions[83], renamed[3])

    def test_reloading_clears_results_and_run_returns_required_results(self):
        workflow = Workflow(2, 4, 42)
        result = workflow.run(self.csv())
        self.assertTrue({"number_of_customers", "pca_components", "clustering_comparison",
                         "best_method", "best_k", "cluster_agreement", "dbscan_result",
                         "cluster_profiles", "cluster_descriptions"} <= result.keys())
        self.assertEqual(result["number_of_customers"], 60)
        workflow.load_data(self.csv())
        self.assertIsNone(workflow.final_labels)
        self.assertIsNone(workflow.scaler)
        self.assertIsNone(workflow.experiments)

    def test_bad_spending_and_degenerate_data(self):
        for value in (-1, np.inf):
            data = self.data.copy()
            data.loc[0, "Fresh"] = value
            workflow = Workflow(2, 4, 42)
            workflow.load_data(self.csv(data))
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                workflow.prepare_features()
        data = self.data.copy()
        data.loc[:, list(Workflow.SPENDING_COLUMNS)] = 0
        workflow.load_data(self.csv(data))
        workflow.prepare_features()
        with self.assertRaisesRegex(ValueError, "varying spending"):
            workflow.apply_pca()

    def test_numeric_coercion_and_original_values_preserved(self):
        data = self.data.astype({"Fresh": object})
        data.loc[0, "Fresh"] = "bad value"
        workflow = Workflow(2, 4, 42)
        workflow.load_data(self.csv(data))
        workflow.prepare_features()
        self.assertEqual(workflow.original_data.loc[0, "Fresh"], "bad value")
        self.assertTrue(pd.isna(workflow.original_spending_features.loc[0, "Fresh"]))
        self.assertAlmostEqual(workflow.spending_features.loc[0, "Fresh"],
                               self.data.loc[1:, "Fresh"].median())
        self.assertEqual(list(workflow.spending_features.columns),
                         list(Workflow.SPENDING_COLUMNS))
        self.assertTrue(np.isfinite(workflow.transformed_features).all())

    def test_pca_multiple_components(self):
        workflow = Workflow(2, 4, 42)
        rng = np.random.default_rng(1)
        data = self.data.copy()
        data.loc[:, list(Workflow.SPENDING_COLUMNS)] = np.expm1(
            rng.uniform(0, 10, (60, 6)))
        workflow.load_data(self.csv(data))
        workflow.prepare_features()
        workflow.apply_pca()
        full = PCA(svd_solver="full").fit(workflow.transformed_features)
        self.assertGreater(workflow.n_components, 1)
        self.assertGreaterEqual(workflow.cumulative_explained_variance[-1], .9)
        self.assertLess(full.explained_variance_ratio_[:workflow.n_components-1].sum(), .9)

    def test_dbscan_valid_partition_beats_lower_noise_invalid_partition(self):
        workflow = self.prepared()
        valid = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int), np.full(20, -1)]
        candidates = [np.zeros(60, dtype=int), valid] + [np.full(60, -1)] * 6
        with patch("wholesale_customer_clustering.DBSCAN") as mock:
            mock.return_value.fit_predict.side_effect = candidates
            table = workflow.detect_outliers_with_dbscan()
        self.assertTrue(np.isnan(table.loc[0, "silhouette"]))
        self.assertAlmostEqual(table.loc[1, "silhouette"], silhouette_score(
            workflow.pca_data[:40], valid[:40]))
        self.assertEqual(workflow.best_dbscan_params, dict(eps=.5, min_samples=10))
        np.testing.assert_array_equal(workflow.dbscan_labels, valid)

    def test_description_distinguishes_overall_spending_from_category_focus(self):
        workflow = self.prepared()
        baseline = workflow.original_spending_features.median()
        workflow.cluster_profiles = pd.DataFrame(
            [baseline * 2, baseline.copy(), baseline.copy()], index=[9, 18, 27])
        workflow.cluster_profiles.loc[18, "Fresh"] = baseline["Fresh"] * 3
        workflow.cluster_profiles.loc[27, ["Grocery", "Detergents_Paper"]] = (
            baseline[["Grocery", "Detergents_Paper"]] * 3)
        descriptions = workflow.describe_clusters()
        self.assertIn("High overall spending", descriptions[9])
        self.assertNotIn("focused", descriptions[9])
        self.assertIn("Fresh-product focused", descriptions[18])
        self.assertIn("Grocery and detergent focused", descriptions[27])

    def test_missing_profile_values_are_not_described_as_typical(self):
        workflow = self.prepared()
        workflow.final_labels = np.repeat([0, 1, 2], 20)
        workflow.dbscan_labels = np.full(60, -1)
        workflow.original_spending_features.loc[:19, "Frozen"] = np.nan
        workflow.original_data["Channel"] = np.nan
        profiles = workflow.create_cluster_profiles()
        self.assertTrue(pd.isna(profiles.loc[0, "Frozen"]))
        self.assertIsNone(profiles.loc[0, "Channel"])
        self.assertEqual(profiles.loc[0, "dbscan_noise_percentage"], 100)
        self.assertIn("No observed spending: Frozen", workflow.describe_clusters()[0])

    def test_minimum_dataset_and_real_all_noise_dbscan(self):
        workflow = Workflow(2, 2, 42)
        result = workflow.run(self.csv(self.data.iloc[[0, 20, 40]]))
        self.assertEqual(result["number_of_customers"], 3)
        self.assertEqual(len(result["final_labels"]), 3)
        self.assertTrue((result["dbscan_labels"] == -1).all())
        self.assertTrue((result["dbscan_results"].noise_percentage == 100).all())

    def test_methods_require_their_prerequisites(self):
        workflow = Workflow(2, 4, 42)
        for method in (workflow.prepare_features, workflow.apply_pca,
                       workflow.run_kmeans_experiments, workflow.run_hierarchical_experiments,
                       workflow.compare_methods, workflow.select_best_clustering,
                       workflow.fit_final_clustering, workflow.measure_cluster_agreement,
                       workflow.detect_outliers_with_dbscan, workflow.create_cluster_profiles,
                       workflow.describe_clusters):
            with self.subTest(method=method.__name__), self.assertRaises(ValueError):
                method()


if __name__ == "__main__":
    unittest.main()
