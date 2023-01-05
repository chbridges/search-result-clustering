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
        self.key = key
        self.target_bins = target_bins
        self.window_size = window_size

        if avg == "mean":
            self.avg = np.mean
        elif avg == "median":
            self.avg = np.median
        else:
            raise ValueError("Parameter 'avg' must be 'mean' or 'median'")

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
            (first_day, first_day + sign_changes * interval, last_day)
        )

        # Assign documents to temporal clusters
        clusters = np.zeros(len(docs), dtype=int)

        for i in range(len(boundaries) - 1):
            hits = df[df.date >= boundaries[i]][df.date < boundaries[i + 1]]
            clusters[hits.index] = i

        return self.merge_clusters(docs, clusters)

    def merge_clusters(self, docs: List[dict], clusters: np.ndarray) -> np.ndarray:
        if not self.window_size:
            return clusters

        raise NotImplementedError
