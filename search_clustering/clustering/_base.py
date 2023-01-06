from abc import ABC

import numpy as np
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
        n_labels = np.unique(labels).shape[0]
        if n_labels == 1:
            return -self.best(-np.inf, np.inf)
        if n_labels > 2:
            vecs = vecs[labels >= 0]
            labels = labels[labels >= 0]
        return self.metric(vecs, labels)
