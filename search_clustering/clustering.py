from abc import ABC, abstractmethod

import numpy as np
from sklearn.cluster import KMeans
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

    @abstractmethod
    def fit_predict(self, vecs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class KMeansClustering(Clustering):
    """Perform clustering on vector embeddings."""

    def fit_predict(self, vecs: np.ndarray) -> np.ndarray:
        best_score = -self.best(-np.inf, np.inf)
        best_labels = np.zeros(vecs.shape[0])

        for k in range(2, vecs.shape[0]):
            labels = KMeans(n_clusters=k).fit_predict(vecs)
            score = self.metric(vecs, labels)
            if score == self.best(score, best_score):
                best_score = score
                best_labels = labels
                continue

        return best_labels
