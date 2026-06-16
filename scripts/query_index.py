#!/usr/bin/env python3
"""Retrieval-only sanity check.

Embeds a query against the vectorstore built by build_index.py and prints
the top-k chunks with their similarity score and metadata. Use this to
eyeball whether retrieval is finding relevant content before plugging in
an LLM.

Usage:
    uv run scripts/query_index.py "how do I run HybridNew"
    uv run scripts/query_index.py "asymptotic CLs limits" --k 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_PERSIST = Path("vectorstore")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "combine"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("query", help="natural-language query")
    p.add_argument("--persist", type=Path, default=DEFAULT_PERSIST,
                   help=f"vectorstore directory (default: {DEFAULT_PERSIST})")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"embedding model (must match build_index.py) (default: {DEFAULT_MODEL})")
    p.add_argument("--collection", default=DEFAULT_COLLECTION,
                   help=f"Chroma collection name (default: {DEFAULT_COLLECTION})")
    p.add_argument("--k", type=int, default=5, help="number of results (default: 5)")
    p.add_argument("--full", action="store_true",
                   help="print the full chunk text instead of a 400-char preview")
    args = p.parse_args()

    if not args.persist.exists():
        raise SystemExit(
            f"ERROR: vectorstore not found at {args.persist}. "
            f"Run scripts/build_index.py first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=args.model)
    store = Chroma(
        persist_directory=str(args.persist),
        collection_name=args.collection,
        embedding_function=embeddings,
    )

    results = store.similarity_search_with_score(args.query, k=args.k)
    if not results:
        print("(no results)")
        return 0

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n=== result {i}  (distance={score:.3f}) ===")
        for k, v in doc.metadata.items():
            print(f"  {k}: {v}")
        text = doc.page_content.strip()
        if not args.full and len(text) > 400:
            text = text[:400] + " ..."
        print()
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
