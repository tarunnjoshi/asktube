"""
Step 4 - Turn the chunks into embeddings (vectors).

This is where AI finally enters the project. Everything before this was
string manipulation.

Three things we prove here:
  1. Our chunks actually FIT inside the model's 256-token window
  2. What a vector looks like
  3. That similar meaning really does produce nearby vectors
     (we do retrieval BY HAND here - FAISS in Step 5 just makes it fast)

Run:
    python steps/step4_embeddings.py "<youtube url>"
    python steps/step4_embeddings.py "<youtube url>" "your question here"
"""

import sys
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL
from step2_transcript import extract_video_id, fetch_transcript
from step3_chunks import clean_transcript, split_into_chunks

# EMBEDDING_MODEL now lives in config.py as EMBEDDING_MODEL

DEFAULT_QUERY = "what is the workflow to make the ad?"


def embedding_dim(model: SentenceTransformer) -> int:
    """The method was renamed in newer sentence-transformers; support both names."""
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    return model.get_sentence_embedding_dimension()


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python steps/step4_embeddings.py "<youtube url>" ["question"]')
        sys.exit(1)

    query = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUERY

    # ---- rebuild the chunks from Steps 2 + 3 ----
    video_id = extract_video_id(sys.argv[1])
    transcript = fetch_transcript(video_id)
    raw_text = " ".join(snippet.text for snippet in transcript)
    chunks = split_into_chunks(clean_transcript(raw_text))
    print(f"rebuilt {len(chunks)} chunks from video {video_id}\n")

    # ---- 1. LOAD THE MODEL ----
    # First run downloads ~90MB to ~/.cache/huggingface/ - after that it is instant.
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    started = time.perf_counter()
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"model       : {EMBEDDING_MODEL}")
    print(f"load time   : {time.perf_counter() - started:.1f}s")
    print(f"dimensions  : {embedding_dim(model)}")
    print(f"max tokens  : {model.max_seq_length}   <-- the hard ceiling")
    print()

    # ---- 2. DO OUR CHUNKS ACTUALLY FIT? ----
    # This is the check I promised in Step 3. Anything over max_seq_length is
    # SILENTLY TRUNCATED by the model - no error, just a half-blind vector.
    print("=" * 70)
    print("TOKEN CHECK - does every chunk fit in the window?")
    print("=" * 70)
    tokenizer = model.tokenizer
    token_counts = [len(tokenizer.encode(chunk)) for chunk in chunks]
    limit = model.max_seq_length

    for index, (chars, tokens) in enumerate(zip((len(c) for c in chunks), token_counts)):
        flag = "TRUNCATED" if tokens > limit else "ok"
        bar = "#" * int(30 * min(tokens, limit * 2) / (limit * 2))
        print(f"  chunk {index:2d}: {chars:5d} chars -> {tokens:4d} tokens  {bar:<30} {flag}")

    over = sum(1 for t in token_counts if t > limit)
    print()
    print(f"chars per token (avg): {sum(len(c) for c in chunks) / sum(token_counts):.2f}")
    if over:
        print(f"PROBLEM: {over}/{len(chunks)} chunks exceed {limit} tokens and will be cut.")
        print(f"         Fix = lower CHUNK_SIZE in step3_chunks.py.")
    else:
        print(f"OK: all {len(chunks)} chunks fit inside {limit} tokens.")
    print()

    # ---- 3. EMBED EVERYTHING ----
    print("=" * 70)
    print("EMBEDDING")
    print("=" * 70)
    started = time.perf_counter()
    # normalize_embeddings=True scales every vector to length 1. That makes
    # cosine similarity identical to a plain dot product - see step 4 below.
    vectors = model.encode(chunks, normalize_embeddings=True)
    elapsed = time.perf_counter() - started

    print(f"input       : {len(chunks)} chunks of text")
    print(f"output shape: {vectors.shape}   <- ({len(chunks)} chunks, {vectors.shape[1]} numbers each)")
    print(f"dtype       : {vectors.dtype}")
    print(f"time        : {elapsed:.2f}s  ({elapsed / len(chunks) * 1000:.0f}ms per chunk)")
    print(f"memory      : {vectors.nbytes / 1024:.1f} KB for the whole video")
    print()
    print("chunk 0's vector, first 8 of 384 numbers:")
    print(f"  {np.round(vectors[0][:8], 4)}")
    print(f"  length of this vector = {np.linalg.norm(vectors[0]):.4f}  (normalised to 1.0)")
    print()

    # ---- 4. RETRIEVAL, BY HAND ----
    # This is the whole point of embeddings. We embed the QUESTION the same way,
    # then measure the angle between it and every chunk.
    print("=" * 70)
    print("SEMANTIC SEARCH (done manually - no vector DB yet)")
    print("=" * 70)
    print(f'query: "{query}"')
    print()

    query_vector = model.encode([query], normalize_embeddings=True)[0]

    # Because every vector has length 1, dot product == cosine similarity.
    # One matrix multiply scores the question against all 15 chunks at once.
    scores = vectors @ query_vector

    ranked = np.argsort(scores)[::-1]

    print("all chunks, best match first:")
    for rank, index in enumerate(ranked):
        marker = "<<<" if rank < 3 else ""
        print(f"  {scores[index]:+.4f}  chunk {index:2d}  {marker}")
    print()

    print("--- TOP 3 CHUNKS ---")
    for rank, index in enumerate(ranked[:3], start=1):
        print(f"\n[{rank}] chunk {index} | similarity {scores[index]:.4f}")
        print(f"    {chunks[index][:300]}...")


if __name__ == "__main__":
    main()
