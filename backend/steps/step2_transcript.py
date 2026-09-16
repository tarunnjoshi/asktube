"""
Step 2 - Get the raw transcript for a YouTube video.

We call youtube-transcript-api directly instead of LangChain's YoutubeLoader,
because the loader hides the real errors when YouTube changes something.

Run:
    python steps/step2_transcript.py "https://www.youtube.com/watch?v=SOME_ID"
    python steps/step2_transcript.py SOME_ID
"""

import re
import sys

from youtube_transcript_api import ( YouTubeTranscriptApi, CouldNotRetrieveTranscript, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable,
)

# A YouTube video ID is ALWAYS exactly 11 chars: letters, digits, _ and -
BARE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Every URL shape YouTube uses. The Chrome extension will hand us tab.url,
# so we have to cope with all of them.
URL_PATTERNS = [
    r"(?:youtube\.com|youtube-nocookie\.com)/watch\?(?:[^ ]*&)?v=([A-Za-z0-9_-]{11})",
    r"youtu\.be/([A-Za-z0-9_-]{11})",
    r"youtube\.com/embed/([A-Za-z0-9_-]{11})",
    r"youtube\.com/shorts/([A-Za-z0-9_-]{11})",
    r"youtube\.com/live/([A-Za-z0-9_-]{11})",
]


def extract_video_id(url_or_id: str) -> str:
    """Normalise any YouTube URL (or a bare ID) down to just the 11-char ID.

    The ID is what we key our FAISS cache on later, so every URL shape for the
    same video must collapse to the same string.
    """
    text = url_or_id.strip()

    if BARE_ID.match(text):
        return text

    for pattern in URL_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    raise ValueError(f"No YouTube video ID found in: {url_or_id!r}")


def fetch_transcript(video_id: str, languages=("en", "en-US", "en-GB")):
    """Return the best caption track available, in order of preference:

      1. English as published
      2. Another language that YouTube can auto-translate into English
      3. Whatever language does exist, untranslated

    Hardcoding English made the app fail on any non-English video, which is a
    lot of YouTube. Step 3 does not care what language the text is in.
    """
    api = YouTubeTranscriptApi()

    # 1. English, the easy path
    try:
        return api.fetch(video_id, languages=list(languages))
    except NoTranscriptFound:
        pass

    # Raises TranscriptsDisabled if the video has no captions at all.
    available = api.list(video_id)

    # 2. something translatable into English
    for transcript in available:
        if transcript.is_translatable and any(
            language.language_code == "en"
            for language in transcript.translation_languages
        ):
            return transcript.translate("en").fetch()

    # 3. take the original language and carry on
    for transcript in available:
        return transcript.fetch()

    raise NoTranscriptFound(video_id, list(languages), available)


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python steps/step2_transcript.py "<youtube url or video id>"')
        sys.exit(1)

    user_input = sys.argv[1]
    video_id = extract_video_id(user_input)

    print(f"input    : {user_input}")
    print(f"video_id : {video_id}")
    print()

    try:
        transcript = fetch_transcript(video_id)
    except TranscriptsDisabled:
        sys.exit("FAILED: this video has captions turned off. Pick another video.")
    except NoTranscriptFound:
        sys.exit("FAILED: no English transcript exists for this video.")
    except VideoUnavailable:
        sys.exit("FAILED: video is private, deleted or region-blocked.")
    except CouldNotRetrieveTranscript as exc:
        sys.exit(f"FAILED ({type(exc).__name__}): {exc}")

    # ---- 1. look at the RAW shape that comes back ----
    kind = "auto-generated" if transcript.is_generated else "human-written"
    print(f"language : {transcript.language} ({transcript.language_code})")
    print(f"kind     : {kind}")
    print(f"snippets : {len(transcript)}")
    print()

    print("--- first 3 raw snippets ---")
    for snippet in list(transcript)[:3]:
        print(f"  [{snippet.start:7.2f}s +{snippet.duration:5.2f}s]  {snippet.text!r}")
    print()

    # ---- 2. flatten into ONE string - this is what Step 3 will chunk ----
    full_text = " ".join(snippet.text for snippet in transcript)

    print(f"full_text characters : {len(full_text):,}")
    print(f"full_text words      : {len(full_text.split()):,}")
    print()
    print("--- first 400 characters ---")
    print(full_text[:400])


if __name__ == "__main__":
    main()
