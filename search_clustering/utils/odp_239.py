from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, WordEmbeddings
from torch import Tensor, mean, stack

DEFAULT_PATH = str(Path(__file__).parent / "datasets/odp-239")


def split_target_name(target_name: str) -> str:
    return " ".join(target_name.split("_"))


def strip_subtopic_id(subtopic_id: str) -> str:
    return subtopic_id[: subtopic_id.find(".")]


def load_odp239(
    path: str = DEFAULT_PATH, return_topic_ids: bool = True
) -> Dict[str, Union[list, dict]]:
    """Returns 1 dataset per category in the ODP-239 dataset. Each dataset is a
    dict of keys data, target, target_names according to sklearn dataset
    conventions, but the dtypes are not numpy arrays: data is a list of dict
    structured like Elasticsearch results, target is a list of int,
    target_names is a dict with subtopic IDs as keys and (topic, subtopic)
    tuples as values.

    :param path: Path to unpacked (?) ODP-239 dataset, ../datasets/odp-239 by default
    :type path: str
    :param topic_ids: strips subtopic IDs to topic IDs in data[target] to allow coarse-grained evaluation
    :type return_topic_ids: boolean
    :returns: Dictionary of category-wise dataset splits with keys data, target, target_names
    :rtype: Dict[str, Union[list, Dict[str, Tuple[str, str]]]]
    """
    # Load and merge all relevant ODP-239 data in dataframe
    docs = pd.read_csv(f"{path}/docs.txt", sep="\t", dtype=str).rename(
        columns={"ID": "docID"}
    )
    docs.drop(columns=["url"], inplace=True)

    subtopics = pd.read_csv(f"{path}/subTopics.txt", sep="\t", dtype=str).rename(
        columns={"ID": "subTopicID"}
    )
    subtopics[["category", "topic", "subTopic"]] = subtopics["description"].str.split(
        " > ", expand=True
    )
    subtopics.drop(columns=["description"], inplace=True)

    strel = pd.read_csv(f"{path}/STRel.txt", sep="\t", dtype=str)
    df = strel.merge(docs, on="docID").merge(subtopics, on="subTopicID")

    # Split dataframe into multiple dataframes, 1 for each category (Arts, Business, ...)
    df_splits: Dict[str, Union[list, dict]] = {}

    for category in df["category"].unique():
        df_cat = df[df["category"] == category]

        # Features: title and snippet
        data = [
            {
                "_id": row["docID"],
                "_source": {"title": row["title"], "snippet": row["snippet"]},
            }
            for _, row in df_cat.iterrows()
        ]

        # Target: topic ID or subtopic ID
        target = [row["subTopicID"] for _, row in df_cat.iterrows()]
        if return_topic_ids:
            target = [strip_subtopic_id(t) for t in target]

        # Target names: topic and subtopic, always uses subtopic IDs as keys to allow multiple mappings for each topic
        subtopics_cat = df_cat[["subTopicID", "topic", "subTopic"]].drop_duplicates()
        subtopics_cat["topic"] = subtopics_cat["topic"].apply(split_target_name)
        subtopics_cat["subTopic"] = subtopics_cat["subTopic"].apply(split_target_name)

        target_names = {
            row["subTopicID"]: (row["topic"], row["subTopic"])
            for _, row in subtopics_cat.iterrows()
        }

        # Return splits like sklearn toy datasets
        df_splits[category] = {
            "data": data,
            "target": target,
            "target_names": target_names,
        }

    return df_splits


def embed_odp239(
    data: dict,
    embeddings: Optional[DocumentPoolEmbeddings] = None,
    return_topic_ids: bool = True,
) -> Dict[str, Union[list, Dict[str, tuple], Dict[str, Tensor]]]:
    """Add new field target_embeddings to the data splits returned by
    load_odp239 that maps topic embeddings to topic IDs.

    :param data: Dictionary of category-wise dataset splits as return by load_odp239
    :type data: Dict[str, Union[list, Dict[str, Tuple[str, str]]]]
    :param embeddings: FlairNLP DocumentPoolEmbeddings object
    :type embeddings: flair.embeddings.DocumentPoolEmbeddings
    :param return_topic_ids: strips subtopic IDs to topic IDs in data[target_embeddings] values to allow coarse-grained evaluation
    :type return topic_ids: boolean
    :returns: Dictionary of category-wise dataset splits with inverse mapping for kNN evaluation
    :rtype: Dict[str, Union[list, Dict[str, tuple], Dict[str, Tensor]]]
    """
    if embeddings is None:
        embeddings = DocumentPoolEmbeddings(WordEmbeddings("en-glove"))

    embedding_cache = {}

    for category in data.keys():
        target_names = data[category]["target_names"]

        for subtopic_id, topic_subtopic in target_names.items():
            if return_topic_ids:
                subtopic_id = strip_subtopic_id(subtopic_id)

            for t in topic_subtopic:
                if t not in embedding_cache:
                    sentence = Sentence(t)
                    embeddings.embed(sentence)
                    embedding_cache[t] = sentence.embedding

            topic, subtopic = topic_subtopic
            target_name_embedding = mean(
                stack([embedding_cache[topic], embedding_cache[subtopic]]), 0
            )
            data[category]["target_embeddings"][target_name_embedding] = subtopic_id

    return data


if __name__ == "__main__":
    data = load_odp239()
    data = embed_odp239(data)