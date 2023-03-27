import multiprocessing
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
from flair.data import Sentence
from flair.embeddings import (
    DocumentPoolEmbeddings,
    SentenceTransformerDocumentEmbeddings,
    TransformerDocumentEmbeddings,
)
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
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
        if "embedding" in docs[0]["_source"].keys():
            return np.vstack([doc["_source"]["embedding"] for doc in docs])

        with multiprocessing.pool.ThreadPool() as pool:
            tokenized_docs = list(pool.imap(self.tokenize, docs, chunksize=8))
        self.embedding_model.embed(tokenized_docs)

        embeddings = np.vstack([doc.embedding for doc in tokenized_docs])
        for i in range(len(docs)):
            docs[i]["_source"]["embedding"] = embeddings[i]
        return embeddings


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
            weights = np.array(doc["_source"]["weights"])
        else:
            weights = [1 / len(paragraphs) for _ in paragraphs]

        tokenized_paragraphs = [Sentence(p) for p in paragraphs]
        tokenized_paragraphs = self.embedding_model.embed(tokenized_paragraphs)
        embeddings = np.vstack([p.embedding for p in tokenized_paragraphs])
        weighted_embeddings = weights[:, np.newaxis] * embeddings
        return np.sum(weighted_embeddings, axis=0)

    def transform(self, docs: List[dict]) -> np.ndarray:
        with multiprocessing.pool.ThreadPool() as pool:
            embeddings = list(pool.imap(self.embed, docs, chunksize=8))
        return np.vstack(embeddings)


class Col2Vec(Embedding):
    """Embed input snippets in Doc2Vec vectors."""

    def __init__(self, column: str, dim: int = 100) -> None:
        self.column = column
        self.dim = dim

    def transform(self, docs: List[dict]) -> np.ndarray:
        docs_col = [doc["_source"][self.column] for doc in docs]
        tagged_snippets = [
            TaggedDocument(word_tokenize(doc), [i]) for i, doc in enumerate(docs_col)
        ]
        model = Doc2Vec(tagged_snippets, vector_size=self.dim, seed=42, workers=1)
        return np.array([model[i] for i in range(len(tagged_snippets))])


class Tfidf(Embedding):
    """Embed input column in TF-IDF vectors."""

    def __init__(self, column: str) -> None:
        self.column = column

    def transform(self, docs: List[dict]) -> np.ndarray:
        snippets = [doc["_source"][self.column] for doc in docs]
        return TfidfVectorizer().fit_transform(snippets).todense()
