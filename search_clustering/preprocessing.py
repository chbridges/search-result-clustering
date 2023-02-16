import multiprocessing
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from bertopic import BERTopic
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from keybert import KeyBERT
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer


class Preprocessing(ABC):
    """Perform text preprocessing before vector embeddings."""

    @abstractmethod
    def transform(self, docs: List[dict]) -> List[dict]:
        raise NotImplementedError


class DummyPreprocessor(Preprocessing):
    """Do nothing."""

    def transform(self, docs: List[dict]) -> List[dict]:
        return docs


class StopWordRemoval(Preprocessing):
    """Remove stopwords."""

    def __init__(self, language="german"):
        self.word_list = stopwords.words(language)

    def transform(self, docs: List[dict]) -> List[dict]:
        tokens = [word_tokenize(doc["snippet"]) for doc in docs]
        tokens = [t for t in tokens if t not in self.word_list]
        for i in range(len(docs)):
            docs[i]["snippet"] = " ".join(tokens[i])
        return docs


class ParagraphSplitter(Preprocessing):
    """Split body into paragraphs."""

    def __init__(self, include_title: bool = True):
        self.include_title = include_title

    def add_paragraphs_and_weights(self, doc: dict) -> dict:
        body = doc["_source"]["body"]
        if self.include_title:
            body = doc["_source"]["title"] + "\n\n" + body

        body_length = len(word_tokenize(body))
        paragraphs = [p for p in body.split("\n") if p != ""]
        weights = [len(word_tokenize(p)) / body_length for p in paragraphs]

        doc["_source"]["paragraphs"] = paragraphs
        doc["_source"]["weights"] = weights

        return doc

    def transform(self, docs: List[dict]) -> List[dict]:
        with multiprocessing.pool.ThreadPool() as pool:
            return list(pool.imap(self.add_paragraphs_and_weights, docs, chunksize=8))


class ParagraphKeyphraseExtractor(Preprocessing):
    """Assign a topic to each paragraph in the document."""

    def __init__(
        self,
        query: Optional[str] = None,
        include_title: bool = True,
        n_gram_range: tuple = (1, 3),
    ) -> None:
        self.query = [query] if query else None
        self.splitter = ParagraphSplitter(include_title)
        self.model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")
        self.vectorizer = CountVectorizer(
            stop_words=stopwords.words("german"), n_gram_range=n_gram_range
        )

    def add_topics(self, doc: dict) -> dict:
        title = doc["_source"]["title"]

        keywords = self.model.extract_keywords(
            doc["_source"]["paragraphs"],
            vectorizer=self.vectorizer,
            seed_keywords=self.query,
        )

        doc["_source"]["topics"] = ". ".join(
            [title] + [", ".join([kw[0] for kw in kw_i]) for kw_i in keywords]
        )

        return doc

    def transform(self, docs: List[dict]) -> List[dict]:
        docs = self.splitter.transform(docs)

        with multiprocessing.pool.ThreadPool() as pool:
            return list(pool.imap(self.add_topics, docs, chunksize=8))


class ParagraphTopicPreprocessor(Preprocessing):
    """Assign a topic to each paragraph in the document."""

    def __init__(self) -> None:
        vectorizer = CountVectorizer(
            stop_words=stopwords.words("german"), ngram_range=(1, 3)
        )
        self.topic_model = BERTopic(
            language="multilingual", vectorizer_model=vectorizer
        )

    def add_topics(self, doc: dict) -> dict:
        title = [doc["_source"]["title"]]
        paragraphs = title + doc["_source"]["body"].split("\n\n")
        tagged_paragraphs = [
            TaggedDocument(paragraph, [i]) for i, paragraph in enumerate(paragraphs)
        ]
        doc2vec = Doc2Vec(tagged_paragraphs, seed=42, workers=1)
        embeddings = np.array([doc2vec[i] for i in range(len(tagged_paragraphs))])
        topics, probs = self.topic_model.fit_transform(
            paragraphs, embeddings=embeddings
        )
        doc["_source"]["topics"] = topics
        return doc

    def transform(self, docs: List[dict]) -> List[dict]:
        with multiprocessing.pool.ThreadPool() as pool:
            return list(pool.imap(self.add_topics, docs, chunksize=8))
