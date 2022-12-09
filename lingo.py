import re
import string
from collections import Counter
from typing import List

import numpy as np
from elastic import Elastic
from gensim.corpora import Dictionary
from gensim.models import LsiModel
from nltk.tokenize import word_tokenize


def tokenize(self, snippet):
    raise NotImplementedError("do not use for now")
    tokens = word_tokenize(snippet.lower())
    return [t for t in tokens if t.strip(string.punctuation) != ""]


class AbstractTopics:
    def _vectorize(self, snippets: List[str]):
        corpus_str = [word_tokenize(snippet) for snippet in snippets]
        id2word = Dictionary(corpus_str)
        corpus_vec = [id2word.doc2bow(doc) for doc in corpus_str]
        return corpus_vec, id2word

    def find_topics_lsi(self, snippets: List[str]):
        corpus, id2word = self._vectorize(snippets)
        lsi = LsiModel(corpus, id2word=id2word)


class FrequentPhrases:
    def _create_suffix_array(self, tokens):
        return [
            rank for suffix, rank in sorted((tokens[i:], i) for i in range(len(tokens)))
        ]

    def _find_longest_common_prefixes(self, tokens, suffix_array):
        lcp = [0] * len(suffix_array)

        for i in range(len(suffix_array) - 1):
            suffix1 = tokens[suffix_array[i] :]
            suffix2 = tokens[suffix_array[i + 1] :]

            for j in range(min(len(suffix1), len(suffix2))):
                if suffix1[j] != suffix2[j]:
                    break
            else:
                j += 1

            lcp[i] = j

        return lcp

    def extract(self, snippet):
        tokens = tokenize(snippet)
        array = self._create_suffix_array(tokens)
        lcp = self._find_longest_common_prefixes(tokens, array)
        indices = np.where(lcp)[0]
        phrases = []

        for i in indices:
            starting_idx = array[i]
            phrase = tokens[starting_idx : (starting_idx + lcp[i])]
            phrases.append(" ".join(phrase))

        return Counter(phrases)


def pipe(idx=0):
    es = Elastic()
    fs = FrequentPhrases()
    snippet = es.sample_snippet("data mining", idx=idx)
    print(snippet)
    return fs.extract(snippet)
