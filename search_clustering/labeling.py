import importlib
from abc import ABC
from collections import Counter
from copy import copy
from typing import List

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer


class Labeling(ABC):
    """Compute labels for pairs of documents and clusters."""

    def fit_predict_cluster(self, docs: List[dict]) -> str:
        raise NotImplementedError

    def fit_predict(
        self, docs: List[dict], clusters: np.ndarray, query: str = ""
    ) -> List[str]:
        self.query = query
        cluster_indices = [np.where(clusters == i) for i in range(max(clusters) + 1)]
        clustered_docs = [[docs[i] for i in cluster[0]] for cluster in cluster_indices]
        labels = [
            f"{self.fit_predict_cluster(cluster)} ({len(cluster)})"
            for cluster in clustered_docs
        ]
        if -1 in clusters:

            labels.append(f"other ({Counter(clusters)[-1]})")
        return labels


class DummyLabeling(Labeling):
    """Return empty strings."""

    def fit_predict_cluster(self, docs: List[dict]) -> str:
        return ""


class CountLabeling(Labeling):
    """Return cluster sizes."""

    def fit_predict_cluster(self, docs: List[dict]) -> str:
        return str(len(docs))


class Topically(Labeling):
    def fit_predict(
        self, docs: List[dict], clusters: np.ndarray, query: str = ""
    ) -> List[str]:
        import topically

        counts = [len(np.where(clusters == i)[0]) for i in range(max(clusters) + 1)]
        app = topically.Topically("api-key")
        titles = [doc["_source"]["title"] for doc in docs]
        _, topic_names = app.name_topics((titles, clusters))
        labels = [
            f"{topic_names[i]} ({counts[i]})"
            for i in range(max(topic_names.keys()) + 1)
        ]
        if -1 in topic_names.keys():
            counts.append(len(np.where(clusters == -1)[0]))
            labels.append(f"other ({counts[-1]})")

        return labels


class FrequentPhrases(Labeling):
    def __init__(
        self,
        language: str = "en",
        column: str = "title",
        n_phrases: int = 3,
        n_gram_max: int = 6,
        n_candidates: int = 12,
    ) -> None:
        self.stopwords = importlib.import_module(f"spacy.lang.{language}").STOP_WORDS
        self.column = column
        self.n_phrases = n_phrases
        self.n_gram_max = n_gram_max
        self.n_candidates = n_candidates

    def fit_predict_cluster(self, docs: List[dict]) -> str:
        token_pattern = r"(?u)\b\w+\b"  # nosec
        vectorizer = CountVectorizer(
            ngram_range=(1, self.n_gram_max), token_pattern=token_pattern
        )
        titles = [doc["_source"][self.column] for doc in docs]

        counts_per_doc = vectorizer.fit_transform(titles)
        counts = np.sum(counts_per_doc, axis=0).getA1()
        argsort_desc = np.argsort(counts)[::-1]

        vocabulary = vectorizer.get_feature_names_out()
        frequent_phrases: List[str] = []

        for i in argsort_desc:
            if len(frequent_phrases) == self.n_candidates:
                break
            phrase = vocabulary[i]
            tokens = word_tokenize(phrase)
            if tokens[0] not in self.stopwords and tokens[-1] not in self.stopwords:
                frequent_phrases.append(phrase)

        if len(frequent_phrases) == 0:
            return vocabulary[argsort_desc[0]]

        cleaned_phrases = self.clean_labels(frequent_phrases)
        cleaned_sorted = [p for p in frequent_phrases if p in cleaned_phrases]

        return ", ".join(cleaned_sorted[: self.n_phrases])

    def clean_labels(self, labels: List[str]) -> List[str]:
        """Merge labels and return list without duplicates."""
        merged_labels = copy(labels)

        for i in range(len(labels)):
            label_i = labels[i]
            idx_i = merged_labels.index(label_i)

            for j in range(len(labels)):
                label_j = labels[j]
                if i != j and label_i in label_j:
                    merged_labels.pop(idx_i)
                    if label_j in merged_labels:
                        idx_j = merged_labels.index(label_j)
                        merged_labels.insert(0, merged_labels.pop(idx_j))
                    break

        return [label for label in merged_labels if label != self.query and label not in self.query.split(" ")]


class TemporalLabeling(Labeling):
    def __init__(self, format="%d.%m.%Y", query="") -> None:  # %x
        self.format = format

    def fit_predict_cluster(self, docs: List[dict]) -> str:
        timestamps = [doc["_source"]["publication_date"][:10] for doc in docs]
        df = pd.DataFrame(pd.to_datetime(timestamps), columns=["date"])
        df_min = df.min()[0]
        df_max = df.max()[0]
        if df_min == df_max:
            return self.date_to_str(df_max)
        return f"{self.date_to_str(df_min)} - {self.date_to_str(df_max)}"

    def date_to_str(self, timestamp: pd.Timestamp):
        return timestamp.strftime(self.format)
