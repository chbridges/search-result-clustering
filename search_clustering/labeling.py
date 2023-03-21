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
from sklearn.feature_extraction.text import CountVectorizer


class Labeling(ABC):
    """Compute labels for pairs of documents and clusters."""

    def fit_predict_cluster(self, docs: List[dict]) -> str:
        raise NotImplementedError

    def fit_predict(self, docs: List[dict], clusters: np.ndarray) -> List[str]:
        cluster_indices = [np.where(clusters == i) for i in range(max(clusters) + 1)]
        clustered_docs = [[docs[i] for i in cluster[0]] for cluster in cluster_indices]
        labels = [
            f"{self.fit_predict_cluster(cluster)} ({len(cluster)})"
            for cluster in clustered_docs
        ]
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
        counts = [len(np.where(clusters == i)[0]) for i in range(max(clusters) + 1)]
        app = topically.Topically("Xp3NNCa65nRi8unD4lxtLSWyvng9ZVokoowqzZV5")
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
        self, n: int = 3, column: str = "title", language: str = "german"
    ) -> None:
        self.n = n
        self.column = column
        self.stopwords = stopwords.words(language)

    def fit_predict_cluster(self, docs: List[dict]) -> str:
        token_pattern = r"(?u)\b\w+\b"  # nosec
        vectorizer = CountVectorizer(ngram_range=(1, 3), token_pattern=token_pattern)
        titles = [doc["_source"][self.column] for doc in docs]

        counts_per_doc = vectorizer.fit_transform(titles)
        counts = np.sum(counts_per_doc, axis=0).getA1()
        argsort_desc = np.argsort(counts)[::-1]

        vocabulary = vectorizer.get_feature_names_out()
        frequent_phrases: List[str] = []

        for i in argsort_desc:
            if len(frequent_phrases) == self.n:
                break
            phrase = vocabulary[i]
            tokens = word_tokenize(phrase)
            if tokens[0] not in self.stopwords and tokens[-1] not in self.stopwords:
                frequent_phrases.append(phrase)

        if len(frequent_phrases) == 0:
            return vocabulary[argsort_desc[0]]

        return ", ".join(frequent_phrases)


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
    def __init__(self, format="%d.%m.%Y") -> None:  # %x
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
