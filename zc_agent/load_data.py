import json

from minsearch import Index
from zc_agent.prepare_data import PROCESSED_DATA_PATH


def read_repo_data():
    with open(PROCESSED_DATA_PATH, "r", encoding="utf-8") as f_in:
        data = json.load(f_in)
    return data


def sliding_window(seq, size, step):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        batch = seq[i:i+size]
        result.append({'start': i, 'content': batch})
        if i + size > n:
            break

    return result


def chunk_documents(docs, size=2000, step=1000):
    chunks = []

    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop('content')
        chunks = sliding_window(doc_content, size=size, step=step)
        for chunk in chunks:
            chunk.update(doc_copy)
        chunks.extend(chunks)

    return chunks


def index_data(
        chunk=False,
        chunking_params=None,
    ):
    docs = read_repo_data()

    if chunk:
        if chunking_params is None:
            chunking_params = {'size': 2000, 'step': 1000}
        docs = chunk_documents(docs, **chunking_params)

    index = Index(
        text_fields=["content", "filename"],
    )

    index.fit(docs)
    return index