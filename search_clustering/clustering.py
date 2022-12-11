from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
from sklearn import cluster
from sklearn.base import ClusterMixin
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


class Clustering(ABC):
    """Perform clustering on vector embeddings."""

    def __init__(self, metric: str = "silhouette") -> None:
        if metric == "calinski-harabasz":
            self.metric = calinski_harabasz_score
            self.best = max
        elif metric == "davies-bouldin":
            self.metric = davies_bouldin_score
            self.best = min
        elif metric == "silhouette":
            self.metric = silhouette_score
            self.best = max
        else:
            raise ValueError

    def score(self, vecs: np.ndarray, labels: np.ndarray) -> float:
        if np.unique(labels).shape[0] == 1:
            return -self.best(-np.inf, np.inf)
        return self.metric(vecs, labels)

    @abstractmethod
    def fit_predict(self, vecs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NClustersOptimization(Clustering, ABC):
    """Perform clustering with n_clusters parameter optimization for algorithms
    such as KMeans."""

    @abstractmethod
    def init_model(self, n_clusters: int) -> ClusterMixin:
        raise NotImplementedError

    def fit_predict(self, vecs: np.ndarray) -> np.ndarray:
        best_score = -self.best(-np.inf, np.inf)
        best_labels = np.zeros(vecs.shape[0])

        for k in range(2, vecs.shape[0] // 2):
            model = self.init_model(n_clusters=k)
            labels = model.fit_predict(vecs)
            score = self.score(vecs, labels)
            if score != self.best(score, best_score):
                break
            best_score = score
            best_labels = labels

        return best_labels


class BisectingOptimization(Clustering, ABC):
    """Perform clustering with eps or min_samples parameter optimization for
    DBSCAN and OPTICS."""

    @abstractmethod
    def init_model(self, param: float) -> ClusterMixin:
        raise NotImplementedError

    def fit_predict(self, vecs: np.ndarray) -> np.ndarray:
        best_score = -self.best(-np.inf, np.inf)
        best_labels = np.zeros(vecs.shape[0])

        boundaries = [0.01, 0.99]

        for _ in range(5):
            models = [self.init_model(boundary) for boundary in boundaries]
            labels = [model.fit_predict(vecs) for model in models]
            scores = [self.score(vecs, label) for label in labels]
            print(scores)

            better_score = self.best(scores[0], scores[1])

            if better_score != self.best(better_score, best_score):
                break

            better_idx = scores.index(better_score)
            best_score = better_score
            best_labels = labels[better_idx]

            boundaries[1 - better_idx] = 0.5 * sum(boundaries)

        return best_labels


class KMeans(NClustersOptimization):
    """Perform K-Means clustering on vector embeddings."""

    def init_model(self, n_clusters: int) -> ClusterMixin:
        return cluster.KMeans(n_clusters=n_clusters)


class BisectingKMeans(NClustersOptimization):
    """Perform bisecting K-Means clustering on vector embeddings."""

    def init_model(self, n_clusters: int) -> ClusterMixin:
        return cluster.BisectingKMeans(n_clusters=n_clusters)


class SpectralClustering(NClustersOptimization):
    """Perform spectral clustering on vector embeddings."""

    def init_model(self, n_clusters: int) -> ClusterMixin:
        return cluster.SpectralClustering(
            n_clusters=n_clusters, assign_labels="cluster_qr"
        )


class HierarchicalClustering(NClustersOptimization):
    """Perform hierarchical clustering on vector embeddings."""

    def __init__(
        self,
        linkage: str = "ward",
        affinity: str = "l2",
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


class MeanShift(Clustering):
    """Perform mean shift clustering on vector embeddings, automatically
    creates a label for outliers."""

    def fit_predict(self, vecs: np.ndarray) -> np.ndarray:
        model = cluster.MeanShift(cluster_all=False)
        return model.fit_predict(vecs)


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
