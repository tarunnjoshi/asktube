"""Shared settings for every step. One place to turn a knob.

Lives in steps/ rather than backend/ because Python puts the running script's
own folder on the import path - so every step can just `from config import ...`
with no sys.path tricks.
"""

# ---------------------------------------------------------------------------
# Step 3 - chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000       # characters, not tokens
CHUNK_OVERLAP = 200     # a CEILING - the splitter snaps to sentence boundaries

# ---------------------------------------------------------------------------
# Step 4 - embeddings
# ---------------------------------------------------------------------------
# Chosen by experiment, not by reputation. See step4b_compare_models.py.
#
#   all-MiniLM-L6-v2           trained on sentence <-> sentence (SYMMETRIC).
#                              256-token window. Truncated 1/15 chunks and
#                              ranked our chunks WRONG.
#
#   multi-qa-MiniLM-L6-cos-v1  trained on 215M question -> passage pairs
#                              (ASYMMETRIC - which is what RAG actually does).
#                              512-token window. Truncated 0/15. Ranked right.
#
# Same size, same speed, same 384 dimensions. Strictly better for our job.
EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

# ---------------------------------------------------------------------------
# Step 5/6 - vector store + retrieval
# ---------------------------------------------------------------------------
TOP_K = 4               # how many chunks the retriever hands to the LLM
INDEX_DIR = "faiss_index"   # one subfolder per video id
