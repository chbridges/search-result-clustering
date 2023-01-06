from abc import ABC, abstractmethod
from math import log

import numpy as np
import sklearn.decomposition
import umap


class Reduction(ABC):
    """Reduce dimensionality of word embeddings."""

    @abstractmethod
    def transform(self, vecs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PCA(Reduction):
    """Perform Principal Component Analysis on word embeddings."""

    def __init__(self, n_components: int = 64) -> None:
        self.n_components = n_components

    def transform(self, vecs: np.ndarray) -> np.ndarray:
        pca = sklearn.decomposition.PCA(n_components=min(len(vecs), self.n_components))
        return pca.fit_transform(vecs)


class UMAP(Reduction):
    """Perform Principal Component Analysis on word embeddings."""

    def __init__(self, n_components: int = 8) -> None:
        self.n_components = n_components
        self.densmap = False

    def transform(self, vecs: np.ndarray) -> np.ndarray:
        return umap.UMAP(
            densmap=self.densmap,
            n_neighbors=round(2 * log(len(vecs), 2)),
            min_dist=0.0,
            n_components=max(self.n_components, 2),
            random_state=42,
        ).fit_transform(vecs)


class DensMAP(UMAP):
    """Perform Principal Component Analysis on word embeddings."""

    def __init__(self, n_components: int = 8) -> None:
        super().__init__(n_components)
        self.densmap = True
