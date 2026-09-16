"""
Step 11 - The LLM, and the whole system as ONE LangChain chain.

This is the "last mai puri chij ko chain mai convert" part of your plan:
one .invoke(question) and everything downstream happens by itself.

Run:
    python steps/step11_chain.py "<youtube url>" "your question"
"""

import sys
import time

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from config import LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE
from pipeline import get_retriever
from step10_prompt import PROMPT, format_context

load_dotenv()


def get_llm() -> ChatHuggingFace:
    """The LLM. Runs on HuggingFace's servers - nothing loads on your laptop."""
    endpoint = HuggingFaceEndpoint(
        model=LLM_MODEL,
        task="text-generation",
        max_new_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        timeout=60,
    )
    return ChatHuggingFace(llm=endpoint)


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
