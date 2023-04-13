import re

from elasticsearch import Elasticsearch
from opensearchpy import OpenSearch


class ElasticClient:
    def __init__(self, url: str = "http://localhost:9200"):
        self.url = url
        self.client = Elasticsearch(url)

    def search(self, query: str, index="faz", field="body", size=10_000):
        response = self.client.search(
            index=index,
            query={"match_phrase": {field: query}},
            highlight={"fragment_size": 100, "fields": {field: {}}},
            size=size,
            request_timeout=120,
        )

        hits = response["hits"]["hits"]
        return self._add_snippet(hits)

    def count(self, query: str, index="faz", field="body"):
        return self.client.count(index=index, query={"match_phrase": {field: query}})

    def get_snippets(self, query: str, index="faz", field="body"):
        hits = self.search(query, index, field)
        return [(hit["_id"], hit["snippet"]) for hit in hits]

    def _add_snippet(self, hits: list):
        for hit in hits:
            snippet = ""
            for highlights in hit["highlight"].values():
                snippet = snippet + " ".join(highlights)
            hit["snippet"] = re.sub(r"</?em>", "", snippet)
        return hits


class OpenClient:
    """Opensearch-py does not seem to support highlighting."""

    def __init__(self, url="http://localhost:9200"):
        self.client = OpenSearch(url)

    def search(self, query: str, index="plos_intros", field="introduction"):
        response = self.client.search(
            index=index,
            body={"match_phrase": {field: query}},
            size=100,
        )

        return response["hits"]["hits"]

    def count(self, query: str, index="test", field="content"):
        return self.client.count(index=index, body={"match_phrase": {field: query}})
