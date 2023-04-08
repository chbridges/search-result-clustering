from typing import List

import numpy as np

from search_clustering.client import ElasticClient
from search_clustering.presets import get_demo_preset

es = ElasticClient("http://localhost:9200")
pipe_knn, pipe_temp, pipe_none = get_demo_preset()

commands: str = """
    query [-t] <q>: query and cluster documents for query q, -t for temporal clustering
    cluster [-t] <c>: hierarchically cluster the cluster with index c, -t for temporal clustering
    list <c>: list the documents in the cluster with index c
    show <d>: show the document with ID d
    quit: self-explanatory
    """

cache: dict = {"D_q": None, "clusters": None, "labels": None, "query": None}


def filter_cache(c: str):
    if cache["D_q"] == None:
        print("Query documents first")
    return [cache["D_q"][i] for i in np.where(cache["clusters"] == int(c))[0]]


def list_cluster(D_c: List[dict]) -> None:
    for d in D_c:
        print(d["_id"], d["_source"]["title"])


def print_clusters() -> None:
    print("Clusters:")
    for i, label in enumerate(cache["labels"]):
        print(i, label)


def show(_id: str) -> None:
    for doc in cache["D_q"]:
        if doc["_id"] == _id:
            print(doc["_source"]["title"])
            print(doc["_source"]["url"], end="\n\n")
            print(doc["_source"]["body"])
            return
    print("Invalid ID")


def cluster_knn(D: List[dict]) -> None:
    if len(D) >= 8:
        docs, clusters, labels, _ = pipe_knn.fit_transform(
            D, visualize=False, query=cache["query"]
        )
    elif len(D) > 0:
        docs, clusters, labels, _ = pipe_none.fit_transform(
            D, visualize=False, query=cache["query"]
        )
    else:
        return
    cache["D_q"] = docs
    cache["clusters"] = clusters
    cache["labels"] = labels
    print_clusters()


def cluster_temp(D) -> None:
    raise NotImplementedError


def query(q, temporal=False) -> None:
    cache["query"] = q
    D_q = es.search(q, size=100)
    print("Retrieved", len(D_q), "results")
    if len(D_q) == 0:
        return
    if not temporal:
        cluster_knn(D_q)
    else:
        cluster_temp(D_q)


def parse_input(inp_str: str = ""):
    if inp_str == "":
        return
    inp = inp_str.split(" ")

    func = inp.pop(0)

    if len(inp) == 0 and func != "quit":
        func = "error"

    if func == "help":
        print(commands)

    elif func == "query":
        if len(inp) > 2 and inp[0] == "-t":
            query(" ".join(inp[1:]), True)
        if inp[0] != "-t":
            query(" ".join(inp))

    elif func == "cluster":
        if len(inp) == 1 and inp[0] != "-t":
            D_c = filter_cache(inp[0])
            cluster_knn(D_c)
        elif len(inp) == 2 and inp[0] == "-t":
            D_c = filter_cache(inp[0])
            cluster_temp(D_c)

    elif func == "list":
        if len(inp) == 1:
            D_c = filter_cache(inp[0])
            list_cluster(D_c)

    elif func == "show":
        if len(inp) == 1:
            show(inp[0])

    elif func == "quit":
        exit()

    elif func == "cache":
        print(cache)

    else:
        print("Invalid command (Type 'help' to see all available commands)")


if __name__ == "__main__":
    print("Type 'help' to see all available commands")
    while True:
        parse_input(input("> "))
