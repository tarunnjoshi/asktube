"""
THE ACTUAL PIPELINE - no teaching prints, no experiments, no measurements.

This is your plan, steps 2 to 9, in one place.

    URL -> transcript -> chunks -> embeddings -> FAISS -> retriever -> docs

Run:
    python steps/pipeline.py "<youtube url>" "your question"
"""

import sys

from config import TOP_K
from step2_transcript import extract_video_id
from step6_retriever import get_embeddings, get_vectorstore


def get_retriever(url_or_id: str):
    """Your steps 2-6. URL goes in, a ready retriever comes out.

    Everything expensive happens ONCE and is cached on disk after that.
    """
    video_id = extract_video_id(url_or_id)              # step 2a
    embeddings = get_embeddings()                       # step 4
    store, _ = get_vectorstore(video_id, url_or_id, embeddings)   # steps 2b,3,5
    return store.as_retriever(                          # step 6
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 12, "lambda_mult": 0.5},
    )


def retrieve(url_or_id: str, question: str):
    """Your steps 7-9. Question goes in, relevant chunks come out."""
    return get_retriever(url_or_id).invoke(question)


if __name__ == "__main__":
    url, question = sys.argv[1], sys.argv[2]

    for document in retrieve(url, question):
        print(f"--- chunk {document.metadata['chunk_index']} ---")
        print(document.page_content)
        print()
