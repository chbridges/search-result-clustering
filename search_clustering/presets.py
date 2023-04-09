from typing import Optional, Tuple, TypedDict, Union

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


def get_demo_preset() -> Tuple[KNNPipeline, Optional[TemporalPipeline], KNNPipeline]:
    pipe_knn = KNNPipeline(
        preprocessing=[
            ParagraphSplitter(),
            ParagraphTopicModeling("de"),
        ],
        embedding=PooledEmbeddings("topics"),
        reduction=Umap(),
        clustering=HDBSCAN(),
        labeling=FrequentPhrases(language="de", n_phrases=2),
    )
    pipe_temp = None
    pipe_none = KNNPipeline(
        DummyPreprocessor(),
        Col2Vec("body"),
        DummyReduction(),
        DummyClustering(),
        FrequentPhrases(),
    )
    return pipe_knn, pipe_temp, pipe_none


def get_odp_preset(model="") -> KNNPipeline:
    clustering: KNNClustering

    if model == "Doc2Vec":
        return KNNPipeline(
            ColumnMerger(["title", "snippet"]),
            Col2Vec("merged"),
            DummyReduction(),
            KMeans(),
            FrequentPhrases("english"),
        )
    elif model == "K-Means":
        clustering = KMeans()
    elif model == "Hierarchical":
        clustering = HierarchicalClustering()
    elif model == "DBSCAN":
        clustering = DBSCAN()
    elif model == "HDBSCAN":
        clustering = HDBSCAN()
    else:
        raise ValueError("Presets: Doc2Vec, K-Means, DBSCAN, HDBSCAN")
    return KNNPipeline(
        ColumnMerger(["title", "snippet"]),
        SentenceMiniLM("merged", use_cache=True),
        Umap(8),
        clustering,
        FrequentPhrases("english"),
    )


def get_odp_params(model=""):
    if model == "all":
        return {
            "preprocessing": [ColumnMerger(["title", "snippet"])],
            "embedding": [Col2Vec("merged"), SentenceMiniLM("merged")],
            "reduction": [DummyReduction(), Umap(8)],
            "clustering": [KMeans(), HierarchicalClustering(), DBSCAN(), HDBSCAN()],
            "labeling": [FrequentPhrases("english")],
        }

    if model == "caching":
        return {
            "preprocessing": [ColumnMerger(["title", "snippet"])],
            "embedding": [SentenceMiniLM("merged", use_cache=True)],
            "reduction": [Umap(8)],
            "clustering": [DummyClustering(), DBSCAN()],
            "labeling": [FrequentPhrases("english")],
        }

    if model == "densmap":
        return {
            "preprocessing": [ColumnMerger(["title", "snippet"])],
            "embedding": [Col2Vec("merged"), SentenceMiniLM("merged")],
            "reduction": [Umap(8, densmap=True)],
            "clustering": [HDBSCAN()],
            "labeling": [FrequentPhrases("english")],
        }

    raise ValueError("Presets: all, caching, densmap")
