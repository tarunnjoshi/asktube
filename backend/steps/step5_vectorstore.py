"""
Step 5 - Put the vectors into FAISS, save the index to disk, load it back.

Four things we prove here:
  1. A LangChain Document = text + metadata (metadata is how we cite sources)
  2. FAISS Flat returns the SAME ranking as our by-hand numpy from Step 4
  3. FAISS's "score" is a DISTANCE, not a similarity - lower is better
  4. Loading a saved index is dramatically faster than re-embedding

Run:
    python steps/step5_vectorstore.py "<youtube url>"
    python steps/step5_vectorstore.py "<youtube url>" "your question"
"""

import shutil
import sys
import time
from pathlib import Path

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL, INDEX_DIR, TOP_K
from step2_transcript import extract_video_id, fetch_transcript
from step3_chunks import clean_transcript, split_into_chunks

DEFAULT_QUERY = "what is the workflow to make the ad?"


def build_documents(chunks: list[str], video_id: str) -> list[Document]:
    """Wrap each chunk as a LangChain Document with metadata attached.

    The metadata rides along with the text through the whole pipeline. It is how
    we will later say "this answer came from chunk 7" instead of just handing the
    user an answer with no source.
    """
    return [
        Document(
            page_content=chunk,
            metadata={
                "video_id": video_id,
                "chunk_index": index,
                "chunk_chars": len(chunk),
            },
        )
        for index, chunk in enumerate(chunks)
    ]


def folder_report(path: Path) -> None:
    """Print every file FAISS wrote, and how big it is."""
    total = 0
    for file in sorted(path.iterdir()):
        size = file.stat().st_size
        total += size
        print(f"  {file.name:<20} {size / 1024:8.1f} KB")
    print(f"  {'TOTAL':<20} {total / 1024:8.1f} KB")


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python steps/step5_vectorstore.py "<youtube url>" ["question"]')
        sys.exit(1)

    query = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUERY

    # ---- rebuild chunks from Steps 2 + 3 ----
    video_id = extract_video_id(sys.argv[1])
    transcript = fetch_transcript(video_id)
    raw_text = " ".join(snippet.text for snippet in transcript)
    chunks = split_into_chunks(clean_transcript(raw_text))
    documents = build_documents(chunks, video_id)

    print(f"video {video_id}: {len(chunks)} chunks -> {len(documents)} Documents")
    print()
    print("what one Document looks like:")
    print(f"  metadata     : {documents[0].metadata}")
    print(f"  page_content : {documents[0].page_content[:80]}...")
    print()

    # ---- 1. THE EMBEDDINGS OBJECT ----
    # Same model as Step 4, but wrapped so LangChain can call it.
    # normalize_embeddings=True is NOT optional here - it is what makes the
    # distance maths below line up with cosine similarity.
    print("=" * 70)
    print("EMBEDDINGS WRAPPER")
    print("=" * 70)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    probe = embeddings.embed_query("test")
    print(f"model      : {EMBEDDING_MODEL}")
    print(f"dimensions : {len(probe)}")
    print(f"vector norm: {np.linalg.norm(probe):.4f}  (must be 1.0)")
    print()

    # ---- 2. BUILD THE INDEX ----
    print("=" * 70)
    print("BUILDING FAISS INDEX")
    print("=" * 70)
    started = time.perf_counter()
    store = FAISS.from_documents(documents, embeddings)
    build_seconds = time.perf_counter() - started

    # store.index is the RAW faiss object. Let us look inside it.
    print(f"build time    : {build_seconds:.2f}s")
    print(f"index class   : {type(store.index).__name__}   <- Flat = exact brute force")
    print(f"vectors stored: {store.index.ntotal}")
    print(f"dimensions    : {store.index.d}")
    print(f"docstore size : {len(store.docstore._dict)} documents")
    print()

    # ---- 3. SAVE TO DISK ----
    index_path = Path(INDEX_DIR) / video_id
    if index_path.exists():
        shutil.rmtree(index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    store.save_local(str(index_path))
    print("=" * 70)
    print(f"SAVED to {index_path}/")
    print("=" * 70)
    folder_report(index_path)
    print()
    print("  index.faiss = the raw float32 vectors + the search structure")
    print("  index.pkl   = the Documents and their metadata (a Python pickle)")
    print()

    # ---- 4. LOAD IT BACK - this is the payoff ----
    print("=" * 70)
    print("LOADING FROM DISK")
    print("=" * 70)
    started = time.perf_counter()
    reloaded = FAISS.load_local(
        str(index_path),
        embeddings,
        # FAISS stores the documents as a pickle, and unpickling can execute
        # arbitrary code. LangChain therefore makes you opt in explicitly.
        # Safe here because WE wrote this file seconds ago. Never pass True for
        # an index file you downloaded from someone else.
        allow_dangerous_deserialization=True,
    )
    load_seconds = time.perf_counter() - started

    print(f"load time : {load_seconds:.4f}s")
    print(f"build time: {build_seconds:.4f}s")
    print(f"SPEEDUP   : {build_seconds / max(load_seconds, 1e-9):.0f}x faster than rebuilding")
    print(f"vectors   : {reloaded.index.ntotal} (same as before)")
    print()

    # ---- 5. SEARCH ----
    print("=" * 70)
    print("SEARCH via FAISS")
    print("=" * 70)
    print(f'query: "{query}"   (k={TOP_K})')
    print()

    hits = reloaded.similarity_search_with_score(query, k=TOP_K)

    print("FAISS returns a DISTANCE, not a similarity. LOWER is better.")
    print("Our vectors are normalised, so squared-L2 and cosine are related by")
    print("    cosine = 1 - distance / 2")
    print()
    for rank, (document, distance) in enumerate(hits, start=1):
        cosine = 1 - distance / 2
        print(f"[{rank}] chunk {document.metadata['chunk_index']:2d}"
              f" | L2 distance {distance:.4f}"
              f" | cosine {cosine:.4f}")
        print(f"    {document.page_content[:140].strip()}...")
        print()

    # ---- 6. VERIFY FAISS AGREES WITH OUR BY-HAND NUMPY ----
    # If these disagree, one of them is lying and we need to know NOW,
    # before we build a retriever on top.
    print("=" * 70)
    print("VERIFICATION - FAISS vs the by-hand numpy from Step 4")
    print("=" * 70)

    chunk_vectors = np.array(embeddings.embed_documents(chunks))
    query_vector = np.array(embeddings.embed_query(query))
    manual_scores = chunk_vectors @ query_vector
    manual_ranking = list(np.argsort(manual_scores)[::-1][:TOP_K])

    faiss_ranking = [document.metadata["chunk_index"] for document, _ in hits]

    print(f"numpy by hand : {manual_ranking}")
    print(f"FAISS         : {faiss_ranking}")
    print(f"same order    : {manual_ranking == faiss_ranking}")
    print()
    print("score agreement, per chunk:")
    for (document, distance) in hits:
        index = document.metadata["chunk_index"]
        converted = 1 - distance / 2
        print(f"  chunk {index:2d}: numpy cosine {manual_scores[index]:.6f}"
              f" | from FAISS distance {converted:.6f}"
              f" | diff {abs(manual_scores[index] - converted):.2e}")


if __name__ == "__main__":
    main()
