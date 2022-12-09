import json
import sys
from datetime import datetime
from itertools import chain

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
        chunk_size = sys.argv[2] if len(sys.argv) > 2 else 512
        with open(sys.argv[1], "r") as json_file:
            data = json.loads(json_file.read())

    except:
        print("Usage: python index_data.py <filepath> [<chunksize>]")
        print("Supported files:", end=" ")
        print("bild_export.json faz_export.json spiegel_export.json welt_export.json")
        exit()

    index = list(data.keys())[0]
    docs = data[index]
    ops = list(
        chain(
            *[
                [{"index": {"_index": index, "_id": f"{index}_{i+1}"}}, doc]
                for i, doc in enumerate(docs)
            ]
        )
    )
    chunks = list(chunk_generator(ops, chunk_size * 2))

    es = Elasticsearch("http://localhost:9200")

    print(f"Bulk: {len(docs)} documents ({len(chunks)} chunks of size {chunk_size})")
    print(f"Start: {strip_time(datetime.now)}\n")

    for chunk in tqdm(chunks):
        es.bulk(operations=chunk)

    print(f"\nEnd: {strip_time(datetime.now())}")
