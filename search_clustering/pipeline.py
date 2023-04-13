from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP

from search_clustering.clustering.knn import KNNClustering
from search_clustering.clustering.temporal import TemporalClustering
from search_clustering.embedding import Embedding
from search_clustering.labeling import Labeling
from search_clustering.preprocessing import Preprocessing
from search_clustering.reduction import Reduction


class Pipeline(ABC):
    verbose = True

    @abstractmethod
    def fit_transform(self, docs: List[dict], visualize: bool, verbose: bool):
        raise NotImplementedError

    @abstractmethod
    def visualize(
        self, vecs: np.ndarray, clusters: np.ndarray, labels: list, title: str = ""
    ):
        raise NotImplementedError

    def print(self, msg: str) -> None:
        if self.verbose:
            print(msg)


class KNNPipeline(Pipeline):
    """Pipeline Preprocessing, Embedding, KNNClustering, Labeling, and
    Visualization."""

    def __init__(
        self,
        preprocessing: Union[Preprocessing, List[Preprocessing]],
        embedding: Embedding,
        reduction: Reduction,
        clustering: Union[KNNClustering, TemporalClustering],
        labeling: Labeling,
    ):
        self.preprocessing = (
            preprocessing if isinstance(preprocessing, list) else [preprocessing]
        )
        self.embedding = embedding
        self.reduction = reduction
        self.clustering = clustering
        self.labeling = labeling

    def fit_transform(
        self,
        docs: List[dict],
        visualize=True,
        verbose=True,
        title="",
        legend: bool = True,
        query="",
    ) -> Tuple[List[dict], np.ndarray, List[str], float]:
        self.verbose = verbose
        steps = 5 + visualize

        self.print(f"[1/{steps}] Preprocessing")
        for preprocessing in self.preprocessing:
            docs = preprocessing.transform(docs)

        self.print(f"[2/{steps}] Embedding")
        vecs = self.embedding.transform(docs)

        self.print(f"[3/{steps}] Reducing Dimensionality")
        vecs = self.reduction.transform(vecs)

        self.print(f"[4/{steps}] Clustering")
        if isinstance(self.clustering, KNNClustering):
            clusters, score = self.clustering.fit_predict(vecs)
        else:
            clusters = self.clustering.fit_predict(docs)

        self.print(f"[5/{steps}] Labeling")
        labels = self.labeling.fit_predict(docs, clusters, query)

        if visualize:
            print(f"[6/{steps}] Visualizing")
            self.visualize(vecs, clusters, labels, title, legend)

        return docs, clusters, labels, score

    def visualize(
        self,
        vecs: np.ndarray,
        clusters: np.ndarray,
        labels: list,
        title: str = "",
        legend: bool = True,
    ):
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

        handles = [
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
        if legend:
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), handles=handles)
        plt.title(title)
        plt.show()


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
        self,
        docs: List[dict],
        visualize: bool = True,
        verbose: bool = True,
        title: str = "",
        add_xticks: bool = False,
        query: str = "",
    ) -> Tuple[List[dict], np.ndarray, List[str]]:
        self.verbose = verbose
        steps = 3 + visualize

        self.print(f"[1/{steps}] Preprocessing")
        docs = self.preprocessing.transform(docs)

        self.print(f"[2/{steps}] Clustering")
        clusters, hist = self.clustering.fit_predict(docs)

        self.print(f"[3/{steps}] Labeling")
        labels = self.labeling.fit_predict(docs, clusters, query)

        if visualize:
            self.print(f"[4/{steps}] Visualizing")
            self.visualize(hist, clusters, labels, title, add_xticks)

        return docs, clusters, labels

    def _truncate_label(self, label: str):
        if " -" in label:
            return label[: label.find(" -")]
        if " (" in label:
            return label[: label.find(" (")]
        return label

    def visualize(
        self, hist: np.ndarray, clusters: np.ndarray, labels: list, title: str = "", add_xticks: bool = False
    ):
        bins = len(hist)
        colors = ["C0" for _ in range(bins)]

        timestamps = [self._truncate_label(label) for label in labels]

        label_next_bin = True
        xticks = ["" for _ in range(bins)]

        color = 0
        bin_sum = 0

        for i in range(len(colors)):
            if label_next_bin and hist[i] > 0 and color < len(timestamps):
                xticks[i] = timestamps[color]
                label_next_bin = False

            bin_sum += hist[i]
            colors[i] = f"C{color}"
            if bin_sum >= len(clusters[clusters == color]):
                color += 1
                bin_sum = 0
                label_next_bin = True

        plt.figure(figsize=(5, 4))
        plt.bar(range(bins), hist, color=colors, label=labels[0])
        for i in range(1, len(labels)):
            plt.bar(0, 0, color=f"C{i}", label=labels[i])

        if add_xticks:
            plt.xticks(range(bins), xticks, rotation=90)
        else:
            plt.xticks([])

        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()
