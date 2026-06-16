#!/usr/bin/env python3
"""Ask Combine-bot a question.

Retrieves the top-k chunks from the vectorstore, hands them to an LLM
(GPT-4.1 via CERN's LiteLLM gateway by default), and prints the answer
along with the sources Claude was given.

Auth:
    Set OPENAI_API_KEY to the key issued by the CERN LiteLLM gateway.

Usage:
    uv run scripts/ask.py "how do I run AsymptoticLimits with toys?"
    uv run scripts/ask.py "..." --k 12 --verbose
    uv run scripts/ask.py "..." --model gpt-4.1-mini
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from litellm import completion

DEFAULT_PERSIST = Path("vectorstore")
DEFAULT_COLLECTION = "combine"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_BASE_URL = "https://llmgw-litellm.web.cern.ch/v1"
DEFAULT_K = 8

SYSTEM_PROMPT = """You are an expert on the CMS Combine statistical analysis tool (HiggsAnalysis-CombinedLimit).

Answer the user's question using ONLY the provided context excerpts.

Rules:
- If the context is insufficient to answer, say so explicitly. Do NOT invent option names, command flags, file paths, or behaviors.
- Cite the sources you used inline with their bracket number, e.g. "use the --robustFit option [2]".
- Be precise and concise. Combine users are technical: prefer exact command syntax over hand-waving.
- If sources disagree (e.g. an old forum reply vs. current docs), trust the docs and note the discrepancy.
- When a question has both a documentation answer and a relevant forum thread, mention the forum thread as a real-world example."""


def source_tag(doc) -> str:
    md = doc.metadata
    stype = md.get("source_type", "?")
    if stype == "docs":
        nav = md.get("nav_path", "")
        return f"docs: {nav}" if nav else "docs"
    if stype == "forum":
        topic = md.get("topic_id", "")
        title = md.get("topic_title", "")
        accepted = " (accepted answer)" if md.get("is_accepted_answer") else ""
        return f"forum t/{topic}{accepted}: {title}"
    if stype == "code":
        rel = md.get("source_path", "")
        return f"code: {rel}"
    if stype == "paper":
        return "paper: Combine paper (arXiv:2404.06614v2)"
    return stype


def source_url(doc) -> str:
    md = doc.metadata
    return md.get("topic_url") or md.get("github_url") or md.get("source_path", "")


def format_context(docs) -> str:
    blocks = []
    for i, doc in enumerate(docs, 1):
        tag = source_tag(doc)
        url = source_url(doc)
        header = f"[{i}] {tag}"
        if url:
            header += f"\nURL: {url}"
        blocks.append(f"{header}\n\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def print_sources(docs, verbose: bool) -> None:
    print(f"=== Retrieved {len(docs)} chunks ===\n")
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] {source_tag(doc)}")
        url = source_url(doc)
        if url:
            print(f"    {url}")
        if verbose:
            preview = doc.page_content.strip().replace("\n", " ")
            if len(preview) > 300:
                preview = preview[:300] + " ..."
            print(f"    {preview}")
    print()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("question", help="natural-language question")
    p.add_argument("--k", type=int, default=DEFAULT_K,
                   help=f"number of chunks to retrieve (default: {DEFAULT_K})")
    p.add_argument("--persist", type=Path, default=DEFAULT_PERSIST,
                   help=f"vectorstore directory (default: {DEFAULT_PERSIST})")
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
                   help="must match build_index.py")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"LLM model id (default: {DEFAULT_MODEL})")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help=f"LiteLLM gateway base URL (default: {DEFAULT_BASE_URL})")
    p.add_argument("--no-stream", action="store_true",
                   help="don't stream the answer; wait for the full response")
    p.add_argument("--verbose", action="store_true",
                   help="print a preview of each retrieved chunk")
    p.add_argument("--quiet", action="store_true",
                   help="print the answer only; suppress sources block")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("ERROR: set OPENAI_API_KEY (the CERN LiteLLM gateway key)")
    if not args.persist.exists():
        raise SystemExit(
            f"ERROR: vectorstore not found at {args.persist}. "
            f"Run scripts/build_index.py first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
    store = Chroma(
        persist_directory=str(args.persist),
        collection_name=args.collection,
        embedding_function=embeddings,
    )
    docs = store.similarity_search(args.question, k=args.k)
    if not docs:
        print("(no chunks retrieved — is the vectorstore populated?)")
        return 1

    if not args.quiet:
        print_sources(docs, verbose=args.verbose)
        print("=== Answer ===\n")

    user_prompt = (
        "Context excerpts:\n\n"
        f"{format_context(docs)}\n\n"
        "---\n\n"
        f"Question: {args.question}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    if args.no_stream:
        response = completion(
            model=args.model, base_url=args.base_url, messages=messages
        )
        print(response.choices[0].message.content)
    else:
        stream = completion(
            model=args.model, base_url=args.base_url, messages=messages, stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
