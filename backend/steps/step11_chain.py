"""
Step 11 - The LLM, and the whole system as ONE LangChain chain.

This is the "last mai puri chij ko chain mai convert" part of your plan:
one .invoke(question) and everything downstream happens by itself.

Run:
    python steps/step11_chain.py "<youtube url>" "your question"
"""

import logging
import sys
import time

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from config import (
    GEMINI_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_THINKING_BUDGET,
)
from pipeline import get_retriever
from step10_prompt import PROMPT, format_context

load_dotenv()

# Google's SDK prints a long "automatic function calling is not recommended"
# notice on every call. We do not use function calling, so it is pure noise.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("google.genai.models").setLevel(logging.ERROR)


def get_llm() -> ChatGoogleGenerativeAI:
    """The LLM. LangChain's own Gemini integration - we just pass the name."""
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
        thinking_budget=LLM_THINKING_BUDGET,
    )


def build_chain(url_or_id: str):
    """Your entire plan, steps 2 through 11, as one Runnable.

    Read the `|` as "feed the output of the left into the right".

        question
           |
           +--> retriever -> format_context ---> {"context": ...}
           |                                                      \
           +--> RunnablePassthrough ----------> {"question": ...} -+-> PROMPT -> LLM -> text

    The dict at the top is a RunnableParallel: both branches receive the SAME
    input (your question) and run at the same time. RunnablePassthrough just
    hands the question along untouched, because the prompt needs it too.
    """
    retriever = get_retriever(url_or_id)

    return (
        {
            "context": retriever | format_context,   # question -> docs -> string
            "question": RunnablePassthrough(),        # question -> question
        }
        | PROMPT            # fill the template
        | get_llm()         # send to Llama 3.1 on HuggingFace
        | StrOutputParser() # pull the plain text out of the response object
    )


if __name__ == "__main__":
    url, question = sys.argv[1], sys.argv[2]

    chain = build_chain(url)

    print(f"Q: {question}")
    print("-" * 70)
    started = time.perf_counter()
    answer = chain.invoke(question)
    elapsed = time.perf_counter() - started
    print(answer.strip())
    print("-" * 70)
    print(f"({elapsed:.1f}s)")
