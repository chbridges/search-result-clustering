from abc import ABC
from string import punctuation
from typing import List

import numpy as np
import pandas as pd
import topically
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LsiModel
from gensim.models.basemodel import BaseTopicModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Labeling(ABC):
    """Compute labels for pairs of documents and clusters."""

    def fit_predict_cluster(self, docs: List[dict]) -> str:
        raise NotImplementedError

    def fit_predict(self, docs: List[dict], clusters: np.ndarray) -> List[str]:
        cluster_indices = [np.where(clusters == i) for i in range(max(clusters) + 1)]
        clustered_docs = [[docs[i] for i in cluster[0]] for cluster in cluster_indices]
        labels = [self.fit_predict_cluster(cluster) for cluster in clustered_docs]
        if -1 in clusters:
            labels.append("other")
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
    def fit_predict(self, docs: List[dict], clusters: np.ndarray) -> List[str]:
        app = topically.Topically("Xp3NNCa65nRi8unD4lxtLSWyvng9ZVokoowqzZV5")
        titles = [doc["_source"]["title"] for doc in docs]
        _, topic_names = app.name_topics((titles, clusters))
        labels = [topic_names[i] for i in range(max(topic_names.keys()))]
        if -1 in topic_names.keys():
            labels.append("other")
        return labels


class TopicModeling(Labeling):
    """Return topics using Latent Semantic Indexing."""

    model: BaseTopicModel

    def tokenize(self, doc: str) -> List[str]:
        tokens = word_tokenize(doc)
        tokens = [t for t in tokens if t not in stopwords.words("german")]
        return [t for t in tokens if t not in punctuation]

    def fit_predict_cluster(self, docs: List[dict]) -> str:
        corpus_str = [self.tokenize(doc["snippet"]) for doc in docs]
        id2word = Dictionary(corpus_str)
        corpus_vec = [id2word.doc2bow(doc) for doc in corpus_str]
        topic_vecs = self.model(corpus_vec, id2word=id2word).get_topics()
        argmax = np.argmax(topic_vecs, axis=1)
        return ", ".join(set([id2word[topic] for topic in argmax]))


class LSI(TopicModeling):
    def __init__(self) -> None:
        self.model = LsiModel


class LDA(TopicModeling):
    def __init__(self) -> None:
        self.model = LdaModel


class TemporalLabeling(Labeling):
    def __init__(self, format="%x") -> None:
        self.format = format

    def fit_predict_cluster(self, docs: List[dict]) -> str:
        timestamps = [doc["_source"]["publication_date"][:10] for doc in docs]
        df = pd.DataFrame(pd.to_datetime(timestamps), columns=["date"])
        if df.min() == df.max():
            return self.date_to_str(df.min())
        return f"{self.date_to_str(df.min())} - {self.date_to_str(df.max())}"

    def date_to_str(self, timestamp: pd.Timestamp):
        return timestamp.strftime(self.format)
