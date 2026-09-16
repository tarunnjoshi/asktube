"""
Step 3 - Clean the transcript, then split it into chunks.

Two jobs here:
  1. CLEAN  - strip YouTube caption junk ([music], >> speaker markers)
  2. CHUNK  - cut the text into overlapping pieces small enough for the
              embedding model to actually read

Run:
    python steps/step3_chunks.py "https://www.youtube.com/watch?v=GDxYnmBXOdk"
"""

import re
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Re-use what we already wrote in Step 2 instead of copy-pasting it.
# This works because Python puts the script's own folder (steps/) on the path.
from config import CHUNK_OVERLAP, CHUNK_SIZE
from step2_transcript import extract_video_id, fetch_transcript

# CHUNK_SIZE and CHUNK_OVERLAP now live in config.py


def clean_transcript(text: str) -> str:
    """Remove YouTube caption artifacts that carry no meaning.

    Auto-generated captions contain things we do NOT want to embed:
        [music] [applause] [laughter]   <- sound tags
        >>                              <- speaker-change markers
    """
    text = re.sub(r"\[[^\]]*\]", " ", text)   # drop anything in square brackets
    text = text.replace(">>", " ")            # drop speaker markers
    text = re.sub(r"\s+", " ", text)          # collapse runs of whitespace
    return text.strip()


def split_into_chunks(text: str) -> list[str]:
    """Cut the transcript into overlapping chunks at the nicest boundary available."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Separator PRIORITY ORDER. The splitter tries the first one; only if a
        # piece is still too big does it fall back to the next. Our transcript is
        # one long line with no newlines, so ". " is what actually does the work -
        # that keeps whole sentences together.
        separators=[".\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


def longest_overlap(first: str, second: str) -> str:
    """Longest suffix of `first` that is also a prefix of `second`.

    Only used to PROVE to ourselves that chunk_overlap is really working.
    """
    for size in range(min(len(first), len(second)), 0, -1):
        if first[-size:] == second[:size]:
            return first[-size:]
    return ""


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python steps/step3_chunks.py "<youtube url or video id>"')
        sys.exit(1)

    video_id = extract_video_id(sys.argv[1])
    transcript = fetch_transcript(video_id)
    raw_text = " ".join(snippet.text for snippet in transcript)

    # ---- 1. CLEAN ----
    clean_text = clean_transcript(raw_text)
    removed = len(raw_text) - len(clean_text)
    print("=" * 70)
    print("CLEANING")
    print("=" * 70)
    print(f"before : {len(raw_text):,} chars")
    print(f"after  : {len(clean_text):,} chars")
    print(f"removed: {removed:,} chars of junk")
    print()

    # ---- 2. CHUNK ----
    chunks = split_into_chunks(clean_text)
    sizes = [len(chunk) for chunk in chunks]

    print("=" * 70)
    print("CHUNKING")
    print("=" * 70)
    print(f"chunk_size    : {CHUNK_SIZE} (the ceiling we asked for)")
    print(f"chunk_overlap : {CHUNK_OVERLAP}")
    print(f"chunks made   : {len(chunks)}")
    print(f"sizes         : min={min(sizes)}  max={max(sizes)}  avg={sum(sizes) // len(sizes)}")
    print()

    # A chunk OVER the ceiling means the splitter could not find a break point.
    oversized = [i for i, size in enumerate(sizes) if size > CHUNK_SIZE]
    if oversized:
        print(f"WARNING: chunks over {CHUNK_SIZE} chars: {oversized}")
    else:
        print(f"OK: every chunk fits under {CHUNK_SIZE} chars")
    print()

    # ---- 3. LOOK at the first two chunks ----
    for index in (0, 1):
        print("-" * 70)
        print(f"CHUNK {index}  ({sizes[index]} chars)")
        print("-" * 70)
        print(chunks[index])
        print()

    # ---- 4. PROVE the overlap is real ----
    shared = longest_overlap(chunks[0], chunks[1])
    print("=" * 70)
    print(f"OVERLAP between chunk 0 and chunk 1: {len(shared)} chars")
    print("=" * 70)
    print(repr(shared))


if __name__ == "__main__":
    main()
