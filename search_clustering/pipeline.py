from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP

from search_clustering.labeling import Labeling
from search_clustering.preprocessing import Preprocessing
from search_clustering.spatial.clustering import SpatialClustering
from search_clustering.spatial.embedding import Embedding
from search_clustering.spatial.reduction import Reduction


class Pipeline:
    """Pipeline Preprocessing, Embedding, Clustering, Labeling, and
    Visualization."""

    def __init__(
        self,
        preprocessing: Preprocessing,
        embedding: Embedding,
        reductions: Union[Reduction, List[Reduction]],
        clustering: SpatialClustering,
        labeling: Labeling,
    ):
        self.preprocessing = preprocessing
        self.embedding = embedding
        self.reductions = reductions if isinstance(reductions, list) else [reductions]
        self.clustering = clustering
        self.labeling = labeling

    def run(
        self, docs: List[dict], visualize=True
    ) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
        steps = 5 + visualize

        print(f"[1/{steps}] Preprocessing")
        docs = self.preprocessing.transform(docs)

        print(f"[2/{steps}] Embedding")
        vecs = self.embedding.transform(docs)

        print(f"[3/{steps}] Reducing Dimensionality")
        for reduction in self.reductions:
            vecs = reduction.transform(vecs)

        print(f"[4/{steps}] Clustering")
        clusters, score = self.clustering.fit_predict(vecs)

        print(f"[5/{steps}] Labeling")
        labels = self.labeling.fit_predict(docs, clusters)

        if visualize:
            print(f"[6/{steps}] Visualizing")
            self.visualize(vecs, clusters, labels)

        return vecs, clusters, labels, score

    @staticmethod
    def visualize(vecs, clusters, labels):
        pca = UMAP(n_components=2).fit_transform(vecs)
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
                label=f"{labels[i]}",
            )
            for i in range(len(labels))
        ]
        plt.legend(handles=handles)
        plt.show()
