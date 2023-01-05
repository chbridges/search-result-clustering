from typing import List, Optional

import numpy as np
import pandas as pd


class TemporalClustering:
    def __init__(
        self,
        key: str = "publication_date",
        avg: str = "mean",
        target_bins: int = 10,
        window_size: Optional[int] = None,
    ) -> None:
        if avg == "mean":
            self.avg = np.mean
        elif avg == "median":
            self.avg = np.median
        else:
            raise ValueError("Parameter 'avg' must be 'mean' or 'median'")

        if target_bins < 2:
            raise ValueError("Parameter 'target_bins' must be greater than 1")

        if window_size and window_size < 2:
            raise ValueError("Parameter 'window_size' must be greater than 1")

        self.key = key
        self.target_bins = target_bins
        self.window_size = window_size

    def fit_predict(self, docs: List[dict]) -> np.ndarray:
        # Construct time histogram
        timestamps = [doc["_source"]["key"] for doc in docs]
        df = pd.DataFrame(pd.to_datetime(timestamps), columns=["date"])

        first_day = df.date.min()
        last_day = df.date.max()
        timespan = last_day - first_day
        bins = self.target_bins if not self.window_size else 100
        interval = timespan / bins

        hist = list(pd.np.histogram(df["date"].astype(int), bins=bins, density=False))

        # Find time bins with above-average change of n_documents
        diff = abs(self.avg(hist[0]))
        sign_changes = np.where(diff - np.mean(diff) > 0)[0] + 1
        boundaries = np.hstack(
            (
                np.datetime64(first_day),
                first_day + sign_changes * interval,
                np.datetime64(last_day),
            )
        )

        # Assign documents to temporal clusters
        df["cluster"] = 0

        for i in range(len(boundaries) - 1):
            hits = df[df.date >= boundaries[i]][df.date < boundaries[i + 1]]
            df["cluster"][hits.index] = i

        # If window_size is set, merge clusters until n_bins <= target_bins
        while df["cluster"].unique().shape[0] > self.target_bins:
            merged_clusters = self.merge_clusters(df["cluster"].copy())
            if (df["cluster"] == merged_clusters).all():
                print(
                    f"Warning: 'target_bins' = {self.target_bins} not reached with 'window_size' = {self.window_size}"
                )
                break
            df["cluster"] = merged_clusters

        return self.remove_empty_clusters(df["cluster"])

    def merge_clusters(self, clusters: pd.Series) -> pd.Series:
        if not self.window_size:
            return clusters

        min_cluster = -1
        min_size = np.inf

        for i in range(clusters.max() - self.window_size + 2):
            sizes = np.zeros(self.window_size)
            for j in range(self.window_size):
                sizes[j] = len(clusters[clusters == i + j])
            # Merge only adjacent clusters where at least 2 clusters contain documents
            if np.count_nonzero(sizes) > 1:
                if sum(sizes) < min_size:
                    min_size = sum(sizes)
                    min_cluster = i

        if min_cluster == -1:
            return clusters

        for i in range(min_cluster + 1, min_cluster + self.window_size):
            clusters[clusters == i] = min_cluster

        # Shift temporal clusters to the left to close resulting gap
        clusters[clusters > min_cluster] -= self.window_size

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
