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
    classes_no_dummy = [c for c in classes_sorted if not "Dummy" in c]

    return "_".join(classes_no_dummy)


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
    "reduction": [DummyReduction(), Umap(32), Umap(16), Umap(8)],
    "clustering": [KMeans(), HierarchicalClustering(), OPTICS(), HDBSCAN()],
    "labeling": [FrequentPhrases("english")],
}

params_test = {
    "preprocessing": [ColumnMerger(["title", "snippet"])],
    "embedding": [Col2Vec("merged")],
    "reduction": [Umap(8)],
    "clustering": [KMeans(), HDBSCAN()],
    "labeling": [FrequentPhrases("english")],
}
