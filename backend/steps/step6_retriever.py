"""
Step 6 - Turn the vector store into a Retriever.

Three things we do here:
  1. BUILD-OR-LOAD caching   - the payoff from Step 5. Embed once, reuse forever.
  2. as_retriever()          - collapse FAISS's many methods into one .invoke()
  3. Compare search types    - similarity vs MMR vs score threshold, measured

Run:
    python steps/step6_retriever.py "<youtube url>"
    python steps/step6_retriever.py "<youtube url>" "your question"
    python steps/step6_retriever.py "<youtube url>" "your question" --rebuild
"""

import shutil
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL, INDEX_DIR, TOP_K
from step2_transcript import extract_video_id, fetch_transcript
from step3_chunks import clean_transcript, split_into_chunks
from step5_vectorstore import build_documents

DEFAULT_QUERY = "what is the workflow to make the ad?"
OFF_TOPIC_QUERY = "how do you make a recipe for butter chicken?"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """One embeddings object, used for BOTH indexing and querying.

    @lru_cache means the model is loaded from disk ONCE per process and reused.
    Without it, a web server would reload ~90MB on every single request.

    These must always be the same model. Index with one model and query with
    another and you get silent garbage - the vectors live in different spaces.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(
    video_id: str,
    url_or_id: str,
    embeddings: HuggingFaceEmbeddings,
    rebuild: bool = False,
) -> tuple[FAISS, str]:
    """Load this video's index from disk, or build and save it if missing.

    This is THE function that makes the Chrome extension viable: the first
    question about a video pays the ~8s indexing cost, every later question
    pays microseconds.
    """
    index_path = Path(INDEX_DIR) / video_id

    if index_path.exists() and not rebuild:
        store = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,  # we wrote this file ourselves
        )
        return store, "loaded from disk (CACHE HIT)"

    # --- cache miss: do the expensive work once ---
    transcript = fetch_transcript(video_id)
    raw_text = " ".join(snippet.text for snippet in transcript)
    chunks = split_into_chunks(clean_transcript(raw_text))
    documents = build_documents(chunks, video_id)

    store = FAISS.from_documents(documents, embeddings)

    if index_path.exists():
        shutil.rmtree(index_path)
    index_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_path))

    return store, "built from scratch (CACHE MISS)"


def redundancy(documents: list[Document], embeddings: HuggingFaceEmbeddings) -> float:
    """Average pairwise cosine similarity BETWEEN the retrieved chunks.

    This measures how much the results repeat each other.
      HIGH = the chunks say the same thing = wasted context
      LOW  = the chunks cover different ground = more information per token
    """
    if len(documents) < 2:
        return 0.0
    vectors = np.array(embeddings.embed_documents([d.page_content for d in documents]))
    similarities = vectors @ vectors.T
    upper_triangle = similarities[np.triu_indices(len(documents), k=1)]
    return float(upper_triangle.mean())


def show(label: str, documents: list[Document], embeddings) -> None:
    indices = [d.metadata["chunk_index"] for d in documents]
    print(f"  {label}")
    print(f"    chunks returned : {indices}")
    print(f"    redundancy      : {redundancy(documents, embeddings):.4f}  (lower = more diverse)")
    for document in documents:
        preview = document.page_content[:95].strip().replace("\n", " ")
        print(f"      [{document.metadata['chunk_index']:2d}] {preview}...")
    print()


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python steps/step6_retriever.py "<url>" ["question"] [--rebuild]')
        sys.exit(1)

    rebuild = "--rebuild" in sys.argv
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    query = positional[1] if len(positional) > 1 else DEFAULT_QUERY

    video_id = extract_video_id(positional[0])
    embeddings = get_embeddings()

    # ---- 1. BUILD OR LOAD ----
    print("=" * 74)
    print("CACHING")
    print("=" * 74)
    started = time.perf_counter()
    store, how = get_vectorstore(video_id, positional[0], embeddings, rebuild=rebuild)
    elapsed = time.perf_counter() - started
    print(f"video    : {video_id}")
    print(f"result   : {how}")
    print(f"took     : {elapsed:.3f}s")
    print(f"vectors  : {store.index.ntotal}")
    print(f"index dir: {Path(INDEX_DIR) / video_id}")
    print()
    print("Run this script a second time - 'how' should flip to CACHE HIT.")
    print()

    # ---- 2. as_retriever() - the Runnable interface ----
    print("=" * 74)
    print("THE RETRIEVER INTERFACE")
    print("=" * 74)
    retriever = store.as_retriever(search_kwargs={"k": TOP_K})
    print(f"type      : {type(retriever).__name__}")
    print(f"is Runnable: {hasattr(retriever, 'invoke')}   <- this is what makes `|` work in Step 9")
    print()

    # ONE method. A string goes in, a list of Documents comes out.
    results = retriever.invoke(query)
    print(f'retriever.invoke("{query}")')
    print(f"  -> list[Document] of length {len(results)}")
    print()

    # ---- 3. COMPARE SEARCH STRATEGIES ----
    print("=" * 74)
    print(f'SEARCH STRATEGIES   query: "{query}"')
    print("=" * 74)

    plain = store.as_retriever(search_kwargs={"k": TOP_K})
    show("A) similarity (default) - pure nearest neighbour", plain.invoke(query), embeddings)

    # MMR: fetch_k candidates, then greedily pick k that are relevant BUT
    # unlike each other. lambda_mult 1.0 = relevance only, 0.0 = diversity only.
    mmr = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 12, "lambda_mult": 0.5},
    )
    show("B) mmr (lambda=0.5) - relevance MINUS redundancy", mmr.invoke(query), embeddings)

    # ---- 4. WHY A SCORE THRESHOLD CANNOT GUARD YOU ----
    print("=" * 74)
    print("THE SCORE THRESHOLD TRAP")
    print("=" * 74)
    for label, test_query in (("ON topic ", query), ("OFF topic", OFF_TOPIC_QUERY)):
        scored = store.similarity_search_with_score(test_query, k=TOP_K)
        cosines = [1 - distance / 2 for _, distance in scored]
        chunks_hit = [d.metadata["chunk_index"] for d, _ in scored]
        print(f"  {label} | best cosine {max(cosines):.4f} | chunks {chunks_hit}")
        print(f'            "{test_query}"')
    print()
    print("  If OFF topic scores near ON topic, NO threshold can separate them.")
    print("  The retriever ALWAYS returns k documents - it cannot say 'no match'.")
    print("  That is why the anti-hallucination guard must live in the PROMPT (Step 7).")


if __name__ == "__main__":
    main()
