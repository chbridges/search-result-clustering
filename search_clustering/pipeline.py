from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
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

    def run(
        self, docs: List[dict], visualize=True
    ) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
        docs = self.preprocessing.transform(docs)
        vecs = self.embedding.transform(docs)
        clusters, score = self.clustering.fit_predict(vecs)
        labels = self.labeling.fit_predict(docs, clusters)

        if visualize:
            self.visualize(vecs, clusters, labels)
            for label in labels:
                print(label)

        return vecs, clusters, labels, score

    def visualize(self, vecs, clusters, labels):
        pca = PCA(n_components=2).fit_transform(vecs)
        colors = [f"C{i}" for i in range(len(labels))]
        counts = [len(clusters[clusters == c]) for c in sorted(set(clusters))]
        if -1 in clusters:
            colors[-1] = "black"
            counts.append(counts.pop(0))

        plt.scatter(pca[:, 0], pca[:, 1], color=[colors[c] for c in clusters])

        handles = handles = [
            plt.Line2D(
                [],
                [],
                linestyle="None",
                color=colors[i],
                marker="o",
                label=f"{counts[i]}",
            )
            for i in range(len(labels))
        ]
        plt.legend(handles=handles)
        plt.show()
