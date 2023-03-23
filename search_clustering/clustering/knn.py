from abc import abstractmethod
from typing import Callable, Optional, Tuple

import hdbscan
import numpy as np
from sklearn import cluster
from sklearn.base import ClusterMixin

from search_clustering.clustering._base import Clustering


class KNNClustering(Clustering):
    """Perform clustering on vector embeddings."""

    @abstractmethod
    def fit_predict(self, vecs: np.ndarray) -> Tuple[np.ndarray, float]:
        raise NotImplementedError


class DummyClustering(KNNClustering):
    """Perform no clustering."""

    def fit_predict(self, vecs: np.ndarray) -> Tuple[np.ndarray, float]:
        return np.zeros(vecs.shape[0], dtype=np.int8), 0


class NClustersOptimization(KNNClustering):
    """Perform clustering with n_clusters parameter optimization for algorithms
    such as KMeans."""

    @abstractmethod
    def init_model(self, n_clusters: int) -> ClusterMixin:
        raise NotImplementedError

    def fit_predict(self, vecs: np.ndarray) -> Tuple[np.ndarray, float]:
        best_labels = np.zeros(vecs.shape[0])
        best_score = -self.best(-np.inf, np.inf)

        for k in range(max(2, vecs.shape[0] // 100), vecs.shape[0] // 2):
            model = self.init_model(n_clusters=k)
            labels = model.fit_predict(vecs)
            score = self.score(vecs, labels)
            if score != self.best(score, best_score):
                break
            best_score = score
            best_labels = labels

        return best_labels, best_score


class BisectingOptimization(KNNClustering):
    """Perform clustering with eps or min_samples parameter optimization for
    DBSCAN and OPTICS."""

    @abstractmethod
    def init_model(self, param: float) -> ClusterMixin:
        raise NotImplementedError

    def fit_predict(self, vecs: np.ndarray) -> Tuple[np.ndarray, float]:
        best_labels = np.zeros(vecs.shape[0])
        best_score = -self.best(-np.inf, np.inf)

        boundaries = [0.01, 0.5, 0.99]

        for _ in range(5):
            params = [0.5 * sum(boundaries[:2]), 0.5 * sum(boundaries[1:])]
            models = [self.init_model(param) for param in params]
            labels = [model.fit_predict(vecs) for model in models]
            scores = [self.score(vecs, label) for label in labels]

            better_score = self.best(scores[0], scores[1])

            if abs(best_score - better_score) < 10e-5:
                break

            better_idx = scores.index(better_score)
            best_score = better_score
            best_labels = labels[better_idx]

            boundaries[2 - 2 * better_idx] = params[better_idx]
            boundaries[1] = 0.5 * (boundaries[0] + boundaries[2])

        return best_labels, best_score


class KMeans(NClustersOptimization):
    """Perform K-Means clustering on vector embeddings."""

    def init_model(self, n_clusters: int) -> ClusterMixin:
        return cluster.KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)


class BisectingKMeans(NClustersOptimization):
    """Perform bisecting K-Means clustering on vector embeddings."""

    def init_model(self, n_clusters: int) -> ClusterMixin:
        return cluster.BisectingKMeans(n_clusters=n_clusters, random_state=42)


class SpectralClustering(NClustersOptimization):
    """Perform spectral clustering on vector embeddings."""

    def init_model(self, n_clusters: int) -> ClusterMixin:
        return cluster.SpectralClustering(
            n_clusters=n_clusters, assign_labels="cluster_qr", random_state=42
        )


class HierarchicalClustering(NClustersOptimization):
    """Perform hierarchical clustering on vector embeddings."""

    def __init__(
        self,
        linkage: str = "ward",
        affinity: str = "euclidean",
        connectivity: Optional[Callable] = None,
        metric: str = "silhouette",
    ) -> None:
        super().__init__(metric)
        self.linkage = linkage
        self.affinity = affinity
        self.connectivity = connectivity

    def init_model(self, n_clusters: int) -> ClusterMixin:
        return cluster.AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=self.linkage,
            metric=self.affinity,
            connectivity=self.connectivity,
        )


class MeanShift(KNNClustering):
    """Perform mean shift clustering on vector embeddings, automatically
    creates a label for outliers."""

    def fit_predict(self, vecs: np.ndarray) -> Tuple[np.ndarray, float]:
        model = cluster.MeanShift(cluster_all=False)
        labels = model.fit_predict(vecs)
        return labels, self.score(vecs, labels)


class DBSCAN(BisectingOptimization):
    """Perform density-based spatial clustering with DBSCAN on vector
    embeddings."""

    def init_model(self, eps: float) -> ClusterMixin:
        return cluster.DBSCAN(eps)


class OPTICS(BisectingOptimization):
    """Perform density-based spatial clustering with OPTICS on vector
    embeddings."""

    def init_model(self, min_samples: float) -> ClusterMixin:
        return cluster.OPTICS(min_samples=min_samples)


class HDBSCAN(KNNClustering):
    """Perform hierarchical density-based spatial clustering on vector
    embeddings."""

    def __init__(self, metric: str = "silhouette", min_samples=None) -> None:
        super().__init__(metric)
        self.min_samples = min_samples

    def fit_predict(self, vecs: np.ndarray) -> Tuple[np.ndarray, float]:
        model = hdbscan.HDBSCAN(
            min_cluster_size=max(2, vecs.shape[0] // 100), min_samples=self.min_samples
        )
        labels = model.fit_predict(vecs)
        return labels, self.score(vecs, labels)
