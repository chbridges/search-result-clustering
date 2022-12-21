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

    def tokenize(self, doc):
        return Sentence(doc["_source"]["body"])

    def transform(self, docs: List[dict]) -> np.ndarray:
        pool = multiprocessing.Pool()
        tokenized_docs = list(pool.map(self.tokenize, docs))
        self.embedding_model.embed(tokenized_docs)
        return np.vstack([doc.embedding for doc in tokenized_docs])


class DistilBERT(TransformerEmbedding):
    """Embed input documents in DistilBERT document embeddings and crash my
    laptop."""

    def __init__(self) -> None:
        self.embedding_model = TransformerDocumentEmbeddings(
            "bert-base-german-cased", fine_tune=False
        )


class SentenceMiniLM(TransformerEmbedding):
    """Embed input documents in sentence MiniLM document embeddings."""

    def __init__(self) -> None:
        self.embedding_model = SentenceTransformerDocumentEmbeddings(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )


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
