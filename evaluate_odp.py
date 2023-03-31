import json
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from search_clustering.presets import make_pipelines, params_odp
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

    results: Dict[str, List[float]] = {
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
            time_pipe.append(delta.seconds)

        results["id"].append(identifier)
        results["silhouette"].append(float(np.mean(silhouette_pipe)))
        results["ari"].append(np.mean(ari_pipe))
        results["recall"].append(np.mean(recall_pipe))
        results["outliers"].append(np.mean(outliers_pipe))
        results["time"].append(np.mean(time_pipe))

        return results


def read_results(filename: str = "evaluation_odp.json") -> pd.DataFrame:
    with open(filename, "r") as f:
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
    df.sort_index(level=["embedding"])

    return df.drop(columns=["id"])


if __name__ == "__main__":
    df = read_odp239_to_df()
    data = create_odp239_splits(df)
    data = embed_odp239_labels_in_splits(data)

    results = evaluate(data, params_odp)

    with open("evaluation_odp.json", "w") as f:
        f.write(json.dumps(results, indent=2))
