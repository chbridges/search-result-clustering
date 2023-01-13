from datetime import timedelta
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from search_clustering.clustering._base import Clustering


class TemporalClustering(Clustering):
    def __init__(
        self,
        key: str = "publication_date",
        target_bins: int = 10,
        average: str = "mean",
        window_size: Optional[Union[int, str]] = "auto",
    ) -> None:
        if target_bins < 2:
            raise ValueError("Parameter 'target_bins' must be greater than 1")

        if average == "mean":
            self.average = np.mean
        elif average == "median":
            self.average = np.median
        else:
            raise ValueError("Parameter 'avg' must be 'mean' or 'median'")

        if window_size and (
            (isinstance(window_size, str) and window_size != "auto")
            or (isinstance(window_size, int) and window_size < 2)
        ):
            raise ValueError("Parameter 'window_size' must be greater than 1 or 'auto'")

        self.key = key
        self.target_bins = target_bins
        self.window_size = window_size

    def fit_predict(self, docs: List[dict]) -> np.ndarray:
        # Construct time histogram
        timestamps = [doc["_source"][self.key] for doc in docs]
        df = pd.DataFrame(pd.to_datetime(timestamps), columns=["date"])

        first_day = df.date.min()
        last_day = df.date.max()
        timespan = last_day - first_day
        bins = self.target_bins if not self.window_size else 100
        interval = timespan / bins

        hist = list(pd.np.histogram(df["date"].astype(int), bins=bins, density=False))

        # Find time bins with above-average change of n_documents
        diff = abs(np.diff(hist[0]))

        if not self.window_size:
            sign_changes = np.where(diff - np.mean(diff) > 0)[0] + 1
        else:
            sign_changes = np.arange(bins)

        boundaries = np.hstack(
            (
                np.datetime64(first_day),
                first_day + sign_changes * interval,
                np.datetime64(last_day + timedelta(1)),
            )
        )

        # Assign documents to temporal clusters
        df["cluster"] = 0

        for i in range(len(boundaries) - 1):
            hits = df[df.date >= boundaries[i]][df.date < boundaries[i + 1]]
            df["cluster"][hits.index] = i

        # If window_size is set, merge clusters until n_bins <= target_bins
        window_size = 2 if self.window_size == "auto" else self.window_size

        if isinstance(window_size, int):
            while df["cluster"].unique().shape[0] > self.target_bins:
                merged_clusters = self.merge_clusters(df["cluster"].copy(), window_size)
                if (df["cluster"] == merged_clusters).all():
                    if self.window_size == "auto":
                        window_size += 1
                    else:
                        print(
                            f"Warning: 'target_bins' = {self.target_bins} not reached with 'window_size' = {self.window_size}"
                        )
                        break
                df["cluster"] = merged_clusters

        return self.remove_empty_clusters(df["cluster"]).to_numpy()

    def merge_clusters(self, clusters: pd.Series, window_size: int) -> pd.Series:
        min_cluster = -1
        min_size = np.inf

        for i in range(clusters.max() - window_size + 2):
            sizes = np.zeros(window_size)
            for j in range(window_size):
                sizes[j] = len(clusters[clusters == i + j])

            # Merge only adjacent clusters where at least 2 clusters contain documents
            if np.count_nonzero(sizes) > 1:
                if sum(sizes) < min_size:
                    min_size = sum(sizes)
                    min_cluster = i

        if min_cluster == -1:
            return clusters

        for i in range(min_cluster + 1, min_cluster + window_size):
            clusters[clusters == i] = min_cluster

        # Shift temporal clusters to the left to close resulting gap
        clusters[clusters > min_cluster] -= window_size

        return clusters

    def remove_empty_clusters(
        self, clusters: pd.Series, min_cluster: int = 0
    ) -> pd.Series:
        """Shift cluster numbers to remove gaps and produce a contiguous
        sequence of integers."""
        mapping = dict()
        unique = sorted(clusters.unique())

        for i in range(min_cluster, len(unique)):
            mapping[unique[i]] = i

        for i in mapping.keys():
            clusters[clusters == i] = mapping[i]

        return clusters
