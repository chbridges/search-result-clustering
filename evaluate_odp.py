import json
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from search_clustering.pipeline import KNNPipeline
from search_clustering.presets import get_odp_params, make_pipelines
from search_clustering.utils.odp_239 import (
    DEFAULT_EMBEDDINGS,
    align_clusters_by_label,
    create_odp239_splits,
    embed_odp239_labels_in_splits,
    embed_target_name,
    read_odp239_to_df,
    subtopic_recall,
)


def evaluate(data: dict, params: dict):
    pipelines = make_pipelines(params)

    results: Dict[str, List[str]] = {
        "id": [],
        "silhouette": [],
        "ari": [],
        "recall": [],
        "outliers": [],
        "time": [],
    }

    for identifier, pipeline in tqdm(pipelines.items(), desc="Pipelines"):
        silhouette_pipe = []
        ari_pipe = []
        recall_pipe = []
        outliers_pipe = []
        time_pipe = []

        for cat in data.keys():
            start_time = datetime.now()
            _, clusters, labels, silhouette = pipeline.fit_transform(
                data[cat]["data"], verbose=False, visualize=False
            )

            delta = datetime.now() - start_time
            ari = adjusted_rand_score(data[cat]["target"], clusters)

            label_embeddings = np.vstack(
                [embed_target_name(label, DEFAULT_EMBEDDINGS).cpu() for label in labels]
            )
            aligned_clusters = align_clusters_by_label(
                data[cat]["target_embeddings"],
                label_embeddings,
                clusters,
                n_neighbors=1,
            )
            recall = subtopic_recall(data[cat]["target"], aligned_clusters)
            outliers = len(clusters[clusters == -1]) / len(clusters)

            silhouette_pipe.append(silhouette)
            ari_pipe.append(ari)
            recall_pipe.append(recall)
            outliers_pipe.append(outliers)
            time_pipe.append(delta.total_seconds())

        def make_str(values: List[float]) -> str:
            mean = round(float(np.mean(values)), 2)
            std = round(float(np.std(values)), 2)
            return f"{mean} ± {std}"

        results["id"].append(identifier)
        results["silhouette"].append(make_str(silhouette_pipe))
        results["ari"].append(make_str(ari_pipe))
        results["recall"].append(make_str(recall_pipe))
        results["outliers"].append(make_str(outliers_pipe))
        results["time"].append(make_str(time_pipe))

    return results


def evaluate_detailed(data: dict, pipeline: KNNPipeline) -> pd.DataFrame:
    category = []
    support = []
    n_clusters = []
    ari = []
    recall = []

    for cat in data.keys():
        _, clusters, labels, _ = pipeline.fit_transform(
            data[cat]["data"], verbose=False, visualize=False
        )

        label_embeddings = np.vstack(
            [embed_target_name(label, DEFAULT_EMBEDDINGS).cpu() for label in labels]
        )
        aligned_clusters = align_clusters_by_label(
            data[cat]["target_embeddings"],
            label_embeddings,
            clusters,
            n_neighbors=1,
        )

        category.append(cat)
        support.append(len(data[cat]["data"]))
        n_clusters.append(max(clusters))
        ari.append(adjusted_rand_score(data[cat]["target"], clusters))
        recall.append(subtopic_recall(data[cat]["target"], aligned_clusters))

    return pd.DataFrame(
        {
            "category": category,
            "support": support,
            "n_clusters": n_clusters,
            "ari": ari,
            "recall": recall,
        }
    )


def plot_correlations(df: pd.DataFrame, title: str = ""):
    df["n_clusters"] /= df["n_clusters"].max()
    corr = df.corr()["support"]
    x = np.linspace(df["support"].min(), df["support"].max(), num=2).reshape(-1, 1)
    lr = Lasso()

    fig, ax = plt.subplots(figsize=(5, 5))

    labels = {"n_clusters": "#Clusters", "ari": "ARI", "recall": "Recall"}

    for col in corr.index[1:]:
        lr.fit(df["support"].to_numpy().reshape(-1, 1), df[col])
        ax.scatter(df["support"], df[col], alpha=0.75)
        ax.plot(
            x,
            lr.predict(x),
            linestyle="dashed",
            label=rf"{labels[col]}, $\rho = {corr[col]:.4f}$",
        )

    ax.set_xlabel("Number of Documents")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.set_title(title)
    fig.show()


def read_results(filename: str = f"evaluation_odp.json") -> pd.DataFrame:
    with open(f"results/{filename}", "r") as f:
        results = json.loads(f.read())

    # preprocessing and labeling equal for all pipelines, remove
    for i in range(len(results["id"])):
        id_i = results["id"][i]
        results["id"][i] = id_i[id_i.find("_") + 1 : id_i.rfind("_")]

    df = pd.DataFrame(results)

    # use pipeline parameters as hierarchical multindex
    split = df["id"].str.split("_")
    split = pd.DataFrame(
        split.to_list(), columns=["embedding", "reduction", "clustering"]
    )
    df.index = pd.MultiIndex.from_frame(split[["embedding", "clustering", "reduction"]])
    df = df.sort_index(level=["embedding"])

    return df.drop(columns=["id"])


if __name__ == "__main__":
    df = read_odp239_to_df()
    data = create_odp239_splits(df)
    data = embed_odp239_labels_in_splits(data)

    results = evaluate(data, get_odp_params("all"))

    with open("results/evaluation_odp_gpu.json", "w") as f:
        f.write(json.dumps(results, indent=2))
