"""
Step 12 - FastAPI server.

Wraps the chain from Step 11 in an HTTP endpoint so the Chrome extension
(which cannot run Python) can use it.

Run:
    uvicorn api:app --reload --port 8000

Then open http://localhost:8000/docs in a browser for a free UI to test it.
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# The step scripts use flat imports (`from config import ...`) because they are
# run directly. Putting steps/ on the path lets us reuse them here unchanged,
# instead of copy-pasting the pipeline into a second place.
sys.path.insert(0, str(Path(__file__).parent / "steps"))

from fastapi import FastAPI, HTTPException          # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field               # noqa: E402

from step2_transcript import extract_video_id       # noqa: E402
from step6_retriever import get_embeddings          # noqa: E402
from step11_chain import build_chain                # noqa: E402

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs ONCE when the server boots (before yield) and once on shutdown.

    We load the embedding model here so the first person to ask a question does
    not wait ~7 extra seconds for it.
    """
    get_embeddings()
    print("embedding model loaded - server ready")
    yield


app = FastAPI(
    title="AskTube",
    description="Ask questions about a YouTube video, answered from its transcript.",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS - without this, the browser silently blocks your extension.
#
# A page at chrome-extension://abc123 is a DIFFERENT ORIGIN from
# http://localhost:8000, and browsers forbid cross-origin calls unless the
# server says otherwise. That "otherwise" is this block.
#
# allow_origins=["*"] is fine for local development. For a public deployment
# you would list your extension's real origin instead.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# One chain per video, kept in memory. Building a chain means loading the FAISS
# index; doing that on every request would waste ~1s each time.
_chains: dict[str, object] = {}


class AskRequest(BaseModel):
    """The shape of an incoming request. FastAPI validates against this for free."""

    url: str = Field(..., description="YouTube URL or bare 11-char video id")
    question: str = Field(..., min_length=1, description="What to ask about the video")


class AskResponse(BaseModel):
    """The shape of what we send back."""

    video_id: str
    question: str
    answer: str
    seconds: float


@app.get("/")
def root() -> dict:
    """A friendly signpost, so visiting the bare URL is not a 404."""
    return {
        "service": "AskTube",
        "docs": "http://localhost:8000/docs",
        "endpoints": ["GET /health", "POST /ask"],
    }


@app.get("/health")
def health() -> dict:
    """A GET with no body. Used to check the server is alive."""
    return {"status": "ok", "videos_cached": len(_chains)}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """The main endpoint: a question in, a grounded answer out."""
    try:
        video_id = extract_video_id(request.url)
    except ValueError as exc:
        # 400 = "your request was wrong". 500 would mean "our server broke".
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if video_id not in _chains:
        try:
            _chains[video_id] = build_chain(request.url)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Could not index this video: {type(exc).__name__}: {exc}",
            ) from exc

    started = time.perf_counter()
    try:
        answer = _chains[video_id].invoke(request.question)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM call failed: {type(exc).__name__}: {exc}",
        ) from exc

    return AskResponse(
        video_id=video_id,
        question=request.question,
        answer=answer.strip(),
        seconds=round(time.perf_counter() - started, 2),
    )
