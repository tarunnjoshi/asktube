"""
Step 10 - Build the prompt.  (Your plan's step 10.)

Take the chunks the retriever found + the user's question, and glue them into
one message for the LLM.

NO AI HERE. This is string formatting. The only clever part is the guard.

Run:
    python steps/step10_prompt.py "<youtube url>" "your question"
"""

import sys

from langchain_core.prompts import ChatPromptTemplate

from pipeline import retrieve

# ---------------------------------------------------------------------------
# THE GUARD - this is the anti-hallucination fix from your plan.
#
# It has to live here, in the prompt, because the retriever cannot help us.
# The retriever ALWAYS returns 4 chunks, even for a question about cooking.
# Only the LLM can read those chunks and notice they are about something else.
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """You answer questions about a YouTube video, using only \
the transcript excerpts you are given.

Rules:
1. Answer ONLY from the transcript excerpts below. Never use outside knowledge.
2. Users type casually and often ungrammatically. Interpret the question \
generously: work out which topic they are asking about and answer about that \
topic. Poor wording is not a reason to refuse.
3. Refuse ONLY when the excerpts genuinely do not mention the topic at all. In \
that case reply with exactly this sentence and nothing else:
   "This is outside of the YouTube video content. Please ask something related \
to this video."
4. Never guess or invent details that are not in the excerpts.
5. Keep the answer short and direct."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INSTRUCTION),
    ("human", "TRANSCRIPT EXCERPTS:\n{context}\n\nQUESTION: {question}"),
])


def format_context(documents) -> str:
    """Turn a list of Documents into one labelled string.

    We number each excerpt so the model can refer to them, and so WE can tell
    which chunk an answer came from when something goes wrong.
    """
    return "\n\n".join(
        f"[excerpt {document.metadata['chunk_index']}]\n{document.page_content}"
        for document in documents
    )


def build_prompt(url_or_id: str, question: str):
    """Retriever -> context -> filled-in prompt. Your steps 6 to 10 in one call."""
    documents = retrieve(url_or_id, question)
    context = format_context(documents)
    return PROMPT.invoke({"context": context, "question": question})


if __name__ == "__main__":
    url, question = sys.argv[1], sys.argv[2]

    messages = build_prompt(url, question).to_messages()

    for message in messages:
        print("=" * 74)
        print(f"{message.type.upper()} MESSAGE")
        print("=" * 74)
        print(message.content)
        print()

    total = sum(len(m.content) for m in messages)
    print(f"total prompt size: {total:,} characters (~{total // 4:,} tokens)")
