import re
from abc import ABC, abstractmethod

from elasticsearch import Elasticsearch
from opensearchpy import OpenSearch


class Client(ABC):
    client = None

    @abstractmethod
    def __init__(self, url="http://localhost:9200"):
        raise NotImplementedError

    def _add_snippet(self, hits: list):
        for hit in hits:
            snippet = ""
            for highlights in hit["highlight"].values():
                snippet = snippet + " ".join(highlights)
            hit["snippet"] = re.sub(r"</?em>", "", snippet)
        return hits

    def search(self, query: str, index="plos_intros", field="introduction"):
        response = self.client.search(
            index=index,
            query={"match_phrase": {field: query}},
            highlight={"fragment_size": 100, "fields": {field: {}}},
            size=100,
        )

        hits = response["hits"]["hits"]
        return self._add_snippet(hits)

    def count(self, query: str, index="test", field="content"):
        return self.client.count(index=index, query={"match_phrase": {field: query}})

    def get_snippets(self, query: str, index="test", field="content"):
        hits = self.search(query, index, field)
        return [(hit["_id"], hit["snippet"]) for hit in hits]


class ElasticClient(Client):
    def __init__(self, url="http://localhost:9200"):
        self.client = Elasticsearch(url)


class OpenClient(Client):
    def __init__(self, url="http://localhost:9200"):
        self.client = OpenSearch(url)
