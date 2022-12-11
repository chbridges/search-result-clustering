from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from search_clustering.clustering import Clustering
from search_clustering.embedding import Embedding
from search_clustering.labeling import Labeling
from search_clustering.preprocessing import Preprocessing


class Pipeline:
    """Pipeline Preprocessing, Embedding, Clustering, Labeling, and
    Visualization."""

    def __init__(
        self,
        preprocessing: Preprocessing,
        embedding: Embedding,
        clustering: Clustering,
        labeling: Labeling,
    ):
        self.preprocessing = preprocessing
        self.embedding = embedding
        self.clustering = clustering
        self.labeling = labeling

    def run(self, docs: List[dict], visualize=True) -> Tuple[np.ndarray, np.ndarray]:
        docs = self.preprocessing.transform(docs)
        vecs = self.embedding.transform(docs)
        clusters = self.clustering.fit_predict(vecs)
        labels = self.labeling.fit_predict(docs, clusters)

        for label in labels:
            print(label)
        if visualize:
            self.visualize(vecs, clusters, "")

        return vecs, clusters

    def visualize(self, vecs, clusters, labels):
        pca = PCA(n_components=2).fit_transform(vecs)
        colors = [f"C{c}" if c >= 0 else "black" for c in clusters]

        plt.scatter(pca[:, 0], pca[:, 1], color=colors)
        plt.show()
