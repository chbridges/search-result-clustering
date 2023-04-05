from typing import TypedDict, Union

from sklearn.model_selection import ParameterGrid

from search_clustering.client import *
from search_clustering.clustering.knn import *
from search_clustering.clustering.temporal import *
from search_clustering.embedding import *
from search_clustering.labeling import *
from search_clustering.pipeline import *
from search_clustering.preprocessing import *
from search_clustering.reduction import *


class Params(TypedDict):
    preprocessing: Union[Preprocessing, List[Preprocessing]]
    embedding: Embedding
    reduction: Reduction
    clustering: Union[KNNClustering, TemporalClustering]
    labeling: Labeling


def make_identifier(params):
    ordering = ["preprocessing", "embedding", "reduction", "clustering", "labeling"]
    classes = {k: str(v.__class__) for k, v in params.items()}
    classes_trunc = {k: v[v.rfind(".") + 1 : -2] for k, v in classes.items()}

    # Handle some important input arguments
    reduction = classes_trunc["reduction"]
    n_components = params["reduction"].n_components
    classes_trunc["reduction"] = f"{reduction}{n_components}"

    classes_sorted = [classes_trunc[ordering[i]] for i in range(len(ordering))]
    # classes_no_dummy = [c for c in classes_sorted if not "Dummy" in c]

    return "_".join(classes_sorted)


def make_pipelines(params: Union[Params, dict]):
    grid = ParameterGrid(params)
    pipelines = {}

    for params in grid:
        identifier = make_identifier(params)
        pipelines[identifier] = KNNPipeline(**params)

    return pipelines


params_odp = {
    "preprocessing": [ColumnMerger(["title", "snippet"])],
    "embedding": [Col2Vec("merged"), SentenceMiniLM("merged")],
    "reduction": [DummyReduction(), Umap(8)],
    "clustering": [KMeans(), HierarchicalClustering(), DBSCAN(), HDBSCAN()],
    "labeling": [FrequentPhrases("english")],
}

params_odp_caching = {
    "preprocessing": [ColumnMerger(["title", "snippet"])],
    "embedding": [SentenceMiniLM("merged", use_cache=True)],
    "reduction": [Umap(8)],
    "clustering": [DummyClustering(), DBSCAN()],
    "labeling": [FrequentPhrases("english")],
}

params_odp_densmap = {
    "preprocessing": [ColumnMerger(["title", "snippet"])],
    "embedding": [Col2Vec("merged"), SentenceMiniLM("merged")],
    "reduction": [Umap(8, densmap=True)],
    "clustering": [HDBSCAN()],
    "labeling": [FrequentPhrases("english")],
}

odp_doc2vec = KNNPipeline(
    ColumnMerger(["title", "snippet"]),
    Col2Vec("merged"),
    DummyReduction(),
    KMeans(),
    FrequentPhrases("english"),
)

odp_kmeans = KNNPipeline(
    ColumnMerger(["title", "snippet"]),
    SentenceMiniLM("merged", use_cache=True),
    Umap(8),
    KMeans(),
    FrequentPhrases("english"),
)

odp_dbscan = KNNPipeline(
    ColumnMerger(["title", "snippet"]),
    SentenceMiniLM("merged", use_cache=True),
    Umap(8),
    DBSCAN(),
    FrequentPhrases("english"),
)

odp_hdbscan = KNNPipeline(
    ColumnMerger(["title", "snippet"]),
    SentenceMiniLM("merged", use_cache=True),
    Umap(8),
    HDBSCAN(),
    FrequentPhrases("english", n_phrases=2),
)
