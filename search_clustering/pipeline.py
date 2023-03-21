from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP

from search_clustering.clustering.spatial import SpatialClustering
from search_clustering.clustering.temporal import TemporalClustering
from search_clustering.embedding import Embedding
from search_clustering.labeling import Labeling
from search_clustering.preprocessing import Preprocessing
from search_clustering.reduction import Reduction


class Pipeline(ABC):
    verbose = True

    @abstractmethod
    def fit_transform(
        self, docs: List[dict], visualize: bool, verbose: bool
    ) -> Tuple[List[dict], np.ndarray, List[str]]:
        raise NotImplementedError

    @staticmethod
    def visualize(vecs, clusters, labels, title=""):
        fig = plt.figure(figsize=(4, 4))
        vecs = UMAP(n_components=2).fit_transform(vecs)
        colors = [f"C{i}" for i in range(max(clusters) + 1)] + ["black"]
        counts = [len(clusters[clusters == c]) for c in sorted(set(clusters))]
        if -1 in clusters:
            counts.append(counts.pop(0))

        plt.scatter(
            vecs[:, 0],
            vecs[:, 1],
            color=[colors[c] for c in clusters],
            alpha=0.75,
            s=10,
        )

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
        # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), handles=handles)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def print(self, msg: str) -> None:
        if self.verbose:
            print(msg)


class SpatialPipeline(Pipeline):
    """Pipeline Preprocessing, Embedding, SpatialClustering, Labeling, and
    Visualization."""

    def __init__(
        self,
        preprocessing: Preprocessing,
        embedding: Embedding,
        reduction: Union[Reduction, List[Reduction]],
        clustering: Union[SpatialClustering, TemporalClustering],
        labeling: Labeling,
    ):
        self.preprocessing = preprocessing
        self.embedding = embedding
        self.reduction = reduction if isinstance(reduction, list) else [reduction]
        self.clustering = clustering
        self.labeling = labeling

    def fit_transform(
        self, docs: List[dict], visualize=True, verbose=True, title=""
    ) -> Tuple[List[dict], np.ndarray, List[str]]:
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
        if isinstance(self.clustering, SpatialClustering):
            clusters, score = self.clustering.fit_predict(vecs)
        else:
            clusters = self.clustering.fit_predict(docs)

        self.print(f"[5/{steps}] Labeling")
        labels = self.labeling.fit_predict(docs, clusters)

        if visualize:
            print(f"[6/{steps}] Visualizing")
            self.visualize(vecs, clusters, labels, title)

        return docs, clusters, labels


class TemporalPipeline(Pipeline):
    """Pipeline Preprocessing, TemporalClustering, and Labeling."""

    def __init__(
        self,
        preprocessing: Preprocessing,
        clustering: TemporalClustering,
        labeling: Labeling,
    ):
        self.preprocessing = preprocessing
        self.clustering = clustering
        self.labeling = labeling

    def fit_transform(
        self, docs: List[dict], visualize=True, verbose=True
    ) -> Tuple[List[dict], np.ndarray, List[str]]:
        self.verbose = verbose

        self.print(f"[1/3] Preprocessing")
        docs = self.preprocessing.transform(docs)

        self.print(f"[2/3] Clustering")
        clusters = self.clustering.fit_predict(docs)

        self.print(f"[3/3] Labeling")
        labels = self.labeling.fit_predict(docs, clusters)

        return docs, clusters, labels
