from abc import ABC, abstractmethod
from typing import List

import numpy as np
import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedding(ABC):
    """Embed input documents in vectors."""

    @abstractmethod
    def transform(self, docs: List[str]) -> List[np.ndarray]:
        raise NotImplementedError


class Snippet2Vec(Embedding):
    """Embed input snippets in Doc2Vec vectors."""

    def transform(self, docs: dict) -> List[np.ndarray]:
        snippets = [doc["snippet"] for doc in docs]
        tagged_snippets = [TaggedDocument(doc, [i]) for i, doc in enumerate(snippets)]
        model = Doc2Vec(tagged_snippets)
        return np.array([model[i] for i in range(len(tagged_snippets))])


class Tfidf(Embedding):
    """Embed input snippets in TF-IDF vectors."""

    def transform(self, docs: dict) -> List[np.ndarray]:
        snippets = [doc["snippet"] for doc in docs]
        return TfidfVectorizer().fit_transform(snippets).todense()


class Nefidf(Embedding):
    """Embed named entities in TF-IDF vectors."""

    def transform(self, docs: dict) -> List[np.ndarray]:
        snippets = [doc["_source"]["introduction"] for doc in docs]
        ner = spacy.load("en_core_web_sm")
        entities = [" ".join(map(str, ner(snippet).ents)) for snippet in snippets]
        return TfidfVectorizer().fit_transform(entities).todense()
