from abc import ABC, abstractmethod
from typing import List

import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


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

    def __init__(self, word_list="nltk"):
        if word_list == "nltk":
            self.word_list = stopwords.words("english")
        elif word_list == "spacy":
            self.word_list = list(spacy.load("en_core_web_sm").Defaults.stop_words)

    def transform(self, docs: List[dict]) -> List[dict]:
        tokens = [word_tokenize(doc["snippet"]) for doc in docs]
        tokens = [t for t in tokens if t not in self.word_list]
        for i in range(len(docs)):
            docs[i]["snippet"] = " ".join(tokens[i])
        return docs
