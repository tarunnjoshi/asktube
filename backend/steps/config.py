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

# ---------------------------------------------------------------------------
# Step 11 - the LLM
# ---------------------------------------------------------------------------
# To use a different provider, swap the one import + one line in
# step11_chain.get_llm() - e.g. ChatOllama(model="llama3.2:3b") for local.
GEMINI_MODEL = "gemini-flash-latest"

# Low temperature because this is a FACTUAL task. We want the model to repeat
# what the transcript says, not to be creative.
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1024

# Gemini Flash is a REASONING model: by default it burns output tokens on
# internal "thinking" before it writes anything. With a small token ceiling it
# spends the whole budget thinking and returns an EMPTY string - no error.
#
# Our task needs no reasoning. The answer is already in the retrieved chunks;
# the model only has to extract it. So we turn thinking off.
LLM_THINKING_BUDGET = 0
