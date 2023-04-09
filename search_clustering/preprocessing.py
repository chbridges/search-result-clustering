import importlib
import multiprocessing
from abc import ABC, abstractmethod
from string import punctuation
from typing import List

import numpy as np
import pke
import spacy
from flair.data import Sentence
from flair.models import SequenceTagger
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LsiModel
from gensim.models.basemodel import BaseTopicModel
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


class ColumnMerger(Preprocessing):
    """Merge multiple columns."""

    def __init__(self, columns: List[str], sep: str = ". ") -> None:
        self.columns = columns
        self.sep = sep

    def transform(self, docs: List[dict]) -> List[dict]:
        for i in range(len(docs)):
            source = docs[i]["_source"]
            merged = self.sep.join([source[col] for col in self.columns])
            docs[i]["_source"]["merged"] = merged
        return docs


class ListJoiner(Preprocessing):
    """Concatenate a nested list to a flat list of strings."""

    def __init__(self, column: str, sep: str = ". ") -> None:
        self.column = column
        self.sep = sep

    def transform(self, docs: List[dict]) -> List[dict]:
        for i in range(len(docs)):
            joined = self.sep.join(docs[i]["_source"][self.column])
            docs[i]["_source"]["joined"] = joined
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
        language: str = "en",
        max_phrases: int = 5,
        sep: str = ", ",
        extractor=pke.unsupervised.SingleRank,
    ) -> None:
        self.language = language
        self.max_phrases = max_phrases
        self.sep = sep
        self.extractor = extractor

    def add_keyphrases(self, doc: dict) -> dict:
        if "keyphrases" in doc["_source"].keys():
            return doc

        extractor = self.extractor()

        keyphrases = []
        for paragraph in doc["_source"]["paragraphs"]:
            extractor.load_document(input=paragraph, language=self.language)
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases_p = extractor.get_n_best(n=self.max_phrases)
            joined = self.sep.join([keyphrase[0] for keyphrase in keyphrases_p])
            keyphrases.append(joined)
        doc["_source"]["keyphrases"] = keyphrases

        return doc

    def transform(self, docs: List[dict]) -> List[dict]:
        with multiprocessing.pool.ThreadPool() as pool:
            return list(pool.imap(self.add_keyphrases, docs, chunksize=8))


class ParagraphTopicModeling(Preprocessing):
    """Return topics using Latent Semantic Indexing."""

    def __init__(
        self, language: str = "en", top_n: int = 3, model: BaseTopicModel = LdaModel
    ) -> None:
        self.stopwords = importlib.import_module(f"spacy.lang.{language}").STOP_WORDS
        self.top_n = top_n
        self.model = model

    def tokenize(self, doc: str) -> List[str]:
        tokens = word_tokenize(doc.lower())
        tokens = [t for t in tokens if t not in self.stopwords]
        return [t for t in tokens if t not in punctuation]

    def add_topics(self, doc: dict) -> dict:
        if "topics" in doc["_source"].keys():
            return doc

        paragraphs = doc["_source"]["paragraphs"]
        corpus_str = [self.tokenize(paragraph) for paragraph in paragraphs]
        id2word = Dictionary(corpus_str)
        bows = [id2word.doc2bow(doc) for doc in corpus_str]
        model = self.model(bows, id2word=id2word, num_topics=len(paragraphs))
        doc["_source"]["topics"] = []
        for i in range(len(paragraphs)):
            topic_id = model.get_document_topics(bows[i])[0][0]
            topics = [
                topic[0] for topic in model.show_topic(topic_id) if len(topic[0]) > 1
            ]
            doc["_source"]["topics"].append(", ".join(topics[: self.top_n]))
        return doc

    def transform(self, docs: List[dict]) -> List[dict]:
        for i in range(len(docs)):
            docs[i] = self.add_topics(docs[i])
        return docs


class NER(Preprocessing):
    """Extract named entities from a given column."""

    def __init__(self, column: str = "body") -> None:
        self.column = column
        self.tagger = spacy.load("xx_ent_wiki_sm")

    def transform(self, docs: List[dict]) -> List[dict]:
        for i in range(len(docs)):
            tagged_doc = self.tagger(docs[i]["_source"][self.column])
            entities = [ent.text for ent in tagged_doc.ents]
            docs[i]["_source"]["entities"] = ", ".join(entities)
        return docs
