from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP

from search_clustering.labeling import Labeling
from search_clustering.preprocessing import Preprocessing
from search_clustering.spatial.clustering import SpatialClustering
from search_clustering.spatial.embedding import Embedding
from search_clustering.spatial.reduction import Reduction


class SpatialPipeline:
    """Pipeline Preprocessing, Embedding, Clustering, Labeling, and
    Visualization."""

    def __init__(
        self,
        preprocessing: Preprocessing,
        embedding: Embedding,
        reduction: Union[Reduction, List[Reduction]],
        clustering: SpatialClustering,
        labeling: Labeling,
    ):
        self.preprocessing = preprocessing
        self.embedding = embedding
        self.reduction = reduction if isinstance(reduction, list) else [reduction]
        self.clustering = clustering
        self.labeling = labeling

    def run(
        self, docs: List[dict], visualize=True, verbose=True
    ) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
        self.verbose = verbose
        steps = 5 + visualize

        self.print(f"[1/{steps}] Preprocessing")
        docs = self.preprocessing.transform(docs)

        self.print(f"[2/{steps}] Embedding")
        vecs = self.embedding.transform(docs)

        self.print(f"[3/{steps}] Reducing Dimensionality")
        for reduction in self.reduction:
            vecs = reduction.transform(vecs)

        self.print(f"[4/{steps}] Clustering")
        clusters, score = self.clustering.fit_predict(vecs)

        self.print(f"[5/{steps}] Labeling")
        labels = self.labeling.fit_predict(docs, clusters)

        if visualize:
            print(f"[6/{steps}] Visualizing")
            self.visualize(vecs, clusters, labels)

        return vecs, clusters, labels, score

    @staticmethod
    def visualize(vecs, clusters, labels):
        vecs = UMAP(n_components=2).fit_transform(vecs)
        colors = [f"C{i}" for i in range(len(labels))]
        counts = [len(clusters[clusters == c]) for c in sorted(set(clusters))]
        if -1 in clusters:
            colors[-1] = "black"
            counts.append(counts.pop(0))

        plt.scatter(vecs[:, 0], vecs[:, 1], color=[colors[c] for c in clusters])

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

    def print(self, msg: str) -> None:
        if self.verbose:
            print(msg)
