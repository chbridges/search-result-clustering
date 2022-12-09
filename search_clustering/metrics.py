from abc import ABC, abstractmethod
from typing import List

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


class Metric(ABC):
    """Compute metric for pairs of embeddings and clusters."""

    def __init__(self):
        self.best = np.argmax

    @abstractmethod
    def compute(self, X: List[np.ndarray], labels: np.ndarray) -> str:
        raise NotImplementedError


class CalinskiHarabasz(Metric):
    """Computte Calinski-Harabasz score."""

    def compute(self, X: List[np.ndarray], labels: np.ndarray) -> str:
        return calinski_harabasz_score(X, labels)


class DaviesBouldin(Metric):
    """Compute Davies-Bouldin score."""

    def __init__(self):
        self.best = np.argmin

    def compute(self, X: List[np.ndarray], labels: np.ndarray) -> str:
        return davies_bouldin_score(X, labels)


class Silhouette(Metric):
    """Compute Silhouette score."""

    def compute(self, X: List[np.ndarray], labels: np.ndarray) -> str:
        return silhouette_score(X, labels)
