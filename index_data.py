import re
import sys
from datetime import datetime
from itertools import chain

import ijson
from elasticsearch import Elasticsearch
from tqdm import tqdm


def chunk_generator(lst, n):
    """Yield successive n-sized chunks from lst.

    https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def strip_time(time: datetime) -> str:
    time = str(time)
    return time[: time.rfind(".")]


if __name__ == "__main__":
    try:
        chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 512
        index = re.search(r"\w+$", sys.argv[1][:-12]).string
        json_file = open(sys.argv[1], "r")
    except:
        print("Usage: python index_data.py <filepath> [<chunksize>]")
        print("Supported files:", end=" ")
        print("bild_export.json faz_export.json spiegel_export.json welt_export.json")
        exit()

    es = Elasticsearch(
        "http://localhost:9200",
        request_timeout=60,
        retry_on_timeout=True,
        max_retries=100,
    )

    docs = []

    for doc in ijson.items(json_file, f"{index}.item"):
        docs.append(doc)

    json_file.close()

    ops = list(
        chain(
            *[
                [{"index": {"_index": index, "_id": f"{index}_{i+1}"}}, doc]
                for i, doc in enumerate(docs)
            ]
        )
    )
    chunks = list(chunk_generator(ops, chunk_size * 2))

    print(f"Bulk: {len(docs)} documents ({len(chunks)} chunks of size {chunk_size})")
    print(f"Start: {strip_time(datetime.now())}\n")

    for chunk in tqdm(chunks):
        es.bulk(operations=chunk)

    print(f"\nEnd: {strip_time(datetime.now())}")
