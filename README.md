# AskTube

Ask questions about any YouTube video and get answers grounded in what was
actually said in it. A RAG (Retrieval-Augmented Generation) system built from
first principles, plus a Chrome extension front-end.

Built step by step to understand every layer rather than gluing a tutorial
together — each stage lives in its own runnable script under `backend/steps/`.

---

## How it works

**Indexing** — runs once per video, then cached on disk:

```
YouTube URL
   |
   |  youtube-transcript-api
   v
346 caption snippets  ->  one string (11,684 chars)
   |
   |  clean out [music] tags and >> speaker markers
   |  RecursiveCharacterTextSplitter  (no AI - pure string logic)
   v
15 chunks (~950 chars, 200-char overlap)
   |
   |  multi-qa-MiniLM-L6-cos-v1  (runs locally, 22M params)
   v
15 vectors x 384 dimensions
   |
   |  FAISS IndexFlatL2
   v
faiss_index/<video_id>/{index.faiss, index.pkl}      38 KB
```

**Answering** — runs per question:

```
question -> embed -> FAISS nearest-neighbour (MMR) -> 4 relevant chunks
                                                          |
                             chunks + question + guard ---+
                                        |
                                        v
                                   LLM -> answer
```

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Transcript | `youtube-transcript-api` | LangChain's `YoutubeLoader` wraps it but hides real errors |
| Chunking | `RecursiveCharacterTextSplitter` | Sentence-aware splits, zero cost, no model needed |
| Embeddings | `multi-qa-MiniLM-L6-cos-v1` | Runs locally. Free, offline, ~50ms/chunk |
| Vector store | FAISS `IndexFlatL2` | Exact search. At 15 vectors an approximate index is slower AND worse |
| Retrieval | MMR | Measurably less redundant than plain similarity — see below |
| Orchestration | LangChain (LCEL) | Adapter layer, so components stay swappable |

---

## Two decisions made by measurement, not by default

### 1. The popular embedding model was the wrong one

`all-MiniLM-L6-v2` is the most-recommended embedding model for RAG. It was
trained for **symmetric** similarity (sentence vs sentence). RAG is
**asymmetric** — a short question against a long passage.

Benchmarked both on the same transcript:

| | `all-MiniLM-L6-v2` | `multi-qa-MiniLM-L6-cos-v1` |
|---|---|---|
| Chunks truncated | 1 / 15 (256-token window) | **0 / 15** (512-token window) |
| Top score, real query | 0.2340 | **0.4647** |
| Score spread | 0.16 | **0.40** |
| Ranked the correct chunk first | no | **yes** |

Same size, same speed, same 384 dimensions. Swapped.

### 2. A similarity threshold cannot detect off-topic questions

A retriever always returns `k` documents. It has no "no match" state. Asked
about a recipe, it returned transcript chunks about AI ads — because
*"recipe"* is genuinely close to *"instructions"* in embedding space:

```
"which free tools does he use?"               -> 0.2218   (legitimate)
"how do you make a recipe for butter chicken?" -> 0.2410   (nonsense, scored HIGHER)
```

No cutoff separates those. So the guard against hallucination lives in the
**prompt**, where the model can read the retrieved text and judge relevance —
something vector arithmetic cannot do.

---

## Setup

Requires Python 3.12 (the ML stack has no wheels for 3.14 yet).

```bash
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
```

Run the finished pipeline:

```bash
cd backend
python steps/pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID" "your question"
```

Or run any single stage to see inside it:

```bash
python steps/step2_transcript.py  "<url>"          # fetch + parse the transcript
python steps/step3_chunks.py      "<url>"          # clean + chunk, shows the overlap
python steps/step4_embeddings.py  "<url>" "<q>"    # token budget + search by hand
python steps/step5_vectorstore.py "<url>"          # build, save, reload, verify
python steps/step6_retriever.py   "<url>" "<q>"    # caching + MMR vs similarity
python steps/step10_prompt.py     "<url>" "<q>"    # the assembled prompt, before the LLM
```

Ask a question end to end:

```bash
python steps/step11_chain.py "<url>" "what is the workflow to make the ad?"
```

---

## Status

- [x] Transcript fetching, with URL-shape normalisation
- [x] Cleaning and chunking
- [x] Local embeddings
- [x] FAISS vector store with on-disk caching (2.2s cold -> 0.05s warm)
- [x] Retriever with MMR
- [x] Prompt assembly with anti-hallucination guard
- [x] LLM via HuggingFace Inference API (Llama 3.1 8B Instruct)
- [x] Single LangChain LCEL chain — `chain.invoke(question)`, 2.7-3.6s end to end
- [x] FastAPI endpoint — `POST /ask`, chains cached per video
- [x] Chrome extension (Manifest V3)

Requires a free HuggingFace token in `backend/.env` (see `.env.example`).

### Known limits

- The embedding model is English-only. Transcripts in other languages are
  fetched and answered correctly, but retrieval quality drops on long
  non-English videos. A multilingual model would need a smaller `chunk_size`.
- Timestamps are discarded during chunking, so answers cannot yet cite a
  moment in the video.

## Running it

Start the backend:

```bash
cd backend
uvicorn api:app --reload --port 8000
```

Load the extension: `chrome://extensions` -> Developer mode -> Load unpacked
-> select `extension/`. Then open any YouTube video and click the icon.
