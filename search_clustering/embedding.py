import multiprocessing
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import spacy
from flair.data import Sentence
from flair.embeddings import (
    SentenceTransformerDocumentEmbeddings,
    TransformerDocumentEmbeddings,
)
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedding(ABC):
    """Embed input documents in vectors."""

    @abstractmethod
    def transform(self, docs: List[dict]) -> np.ndarray:
        raise NotImplementedError


class TransformerEmbedding(Embedding):
    """Embed input documents in transformer document embeddings."""

    embedding_model: Union[
        SentenceTransformerDocumentEmbeddings, TransformerDocumentEmbeddings
    ]

    def __init__(self, key: str = "body") -> None:
        self.key = key

    def tokenize(self, doc):
        return Sentence(doc["_source"][self.key])

    def transform(self, docs: List[dict]) -> np.ndarray:
        with multiprocessing.pool.ThreadPool() as pool:
            tokenized_docs = list(pool.imap(self.tokenize, docs, chunksize=8))
        self.embedding_model.embed(tokenized_docs)
        return np.vstack([doc.embedding for doc in tokenized_docs])


class DistilBERT(TransformerEmbedding):
    """Embed input documents in DistilBERT document embeddings and crash my
    laptop."""

    def __init__(self, key: str = "body") -> None:
        super().__init__(key)
        self.embedding_model = TransformerDocumentEmbeddings(
            "bert-base-german-cased", fine_tune=False
        )


class SentenceMiniLM(TransformerEmbedding):
    """Embed input documents in sentence MiniLM document embeddings."""

    def __init__(self, key: str = "body") -> None:
        super().__init__(key)
        self.embedding_model = SentenceTransformerDocumentEmbeddings(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )


class ParagraphPoolEmbeddings(Embedding):
    """Average pool paragraph-wise embeddings.

    Needs to follow ParagraphSplitter.
    """

    def __init__(self, weighted: bool = True) -> None:
        self.weighted = weighted
        self.embedding_model = SentenceTransformerDocumentEmbeddings(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

    def embed(self, doc: dict) -> np.ndarray:
        paragraphs = doc["_source"]["paragraphs"]

        if self.weighted:
            weights = np.ndarray(doc["_source"]["weights"])
        else:
            weights = [1 / len(paragraphs) for _ in paragraphs]

        tokenized_paragraphs = [Sentence(p) for p in paragraphs]
        tokenized_paragraphs = self.embedding_model.embed(tokenized_paragraphs)
        embeddings = [p.embedding for p in tokenized_paragraphs]
        weighted_embeddings = weights[:, np.newaxis] * embeddings
        return np.sum(weighted_embeddings, axis=0)

    def transform(self, docs: List[dict]) -> np.ndarray:
        with multiprocessing.pool.ThreadPool() as pool:
            embeddings = list(pool.imap(self.embed, docs, chunksize=8))
        return np.vstack(embeddings)


class Snippet2Vec(Embedding):
    """Embed input snippets in Doc2Vec vectors."""

    def transform(self, docs: List[dict]) -> np.ndarray:
        snippets = [doc["snippet"] for doc in docs]
        tagged_snippets = [TaggedDocument(doc, [i]) for i, doc in enumerate(snippets)]
        model = Doc2Vec(tagged_snippets, seed=42, workers=1)
        return np.array([model[i] for i in range(len(tagged_snippets))])


class Tfidf(Embedding):
    """Embed input snippets in TF-IDF vectors."""

    def transform(self, docs: List[dict]) -> np.ndarray:
        snippets = [doc["snippet"] for doc in docs]
        return TfidfVectorizer().fit_transform(snippets).todense()


class Nefidf(Embedding):
    """Embed named entities in TF-IDF vectors."""

    def transform(self, docs: List[dict]) -> np.ndarray:
        snippets = [doc["_source"]["body"] for doc in docs]
        ner = spacy.load("en_core_web_sm")
        entities = [" ".join(map(str, ner(snippet).ents)) for snippet in snippets]
        return TfidfVectorizer().fit_transform(entities).todense()
