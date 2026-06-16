#!/usr/bin/env python3
"""Build the Combine-bot vector index.

Sources currently ingested:
  - paper:  knowledge_base/paper/paper_clean.txt
  - code:   selected files from knowledge_base/combine/
            (python/, scripts/, interface/*.h, bin/*.cpp)

Each source has its own loader; their chunks are merged and embedded into
a single Chroma collection under vectorstore/. Re-running wipes and
rebuilds the persist directory so the index always matches the current
source files.

Usage:
    uv run scripts/build_index.py
    uv run scripts/build_index.py --chunk-size 800 --chunk-overlap 100
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

PAPER_PATH = Path("knowledge_base/paper/paper_clean.txt")
COMBINE_ROOT = Path("knowledge_base/combine")
COMBINE_VERSION = "v10.6.0"  # the submodule is pinned to this tag
GITHUB_BASE = (
    f"https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/{COMBINE_VERSION}"
)

DEFAULT_PERSIST = Path("vectorstore")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "combine"

# (glob relative to COMBINE_ROOT, language tag for metadata, splitter language or None)
CODE_INCLUDES: list[tuple[str, str, Language | None]] = [
    ("python/**/*.py",   "python", Language.PYTHON),
    ("scripts/*.py",     "python", Language.PYTHON),
    ("scripts/*.sh",     "shell",  None),
    ("interface/*.h",    "header", Language.CPP),
    ("bin/*.cpp",        "cpp",    Language.CPP),
]


def _splitter(language: Language | None, chunk_size: int, chunk_overlap: int):
    if language is None:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    return RecursiveCharacterTextSplitter.from_language(
        language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


def load_paper_chunks(path: Path, chunk_size: int, chunk_overlap: int):
    text = path.read_text()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.create_documents(
        [text],
        metadatas=[
            {
                "source_type": "paper",
                "source_path": str(path),
                "title": "Combine paper (arXiv:2404.06614v2)",
            }
        ],
    )


def load_code_chunks(root: Path, chunk_size: int, chunk_overlap: int):
    """Walk CODE_INCLUDES under `root`, chunk per language, attach metadata."""
    all_chunks = []
    for pattern, lang_label, splitter_lang in CODE_INCLUDES:
        splitter = _splitter(splitter_lang, chunk_size, chunk_overlap)
        files = sorted(root.glob(pattern))
        per_pattern = 0
        for path in files:
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                print(f"  skipping non-utf8 file: {path}", file=sys.stderr)
                continue
            if not text.strip():
                continue
            rel = path.relative_to(root)
            metadata = {
                "source_type": "code",
                "language": lang_label,
                "source_path": str(path),
                "github_url": f"{GITHUB_BASE}/{rel}",
            }
            file_chunks = splitter.create_documents([text], metadatas=[metadata])
            all_chunks.extend(file_chunks)
            per_pattern += len(file_chunks)
        print(f"  {pattern}: {len(files)} files -> {per_pattern} chunks")
    return all_chunks


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--paper", type=Path, default=PAPER_PATH,
                   help=f"path to cleaned paper text (default: {PAPER_PATH})")
    p.add_argument("--combine-root", type=Path, default=COMBINE_ROOT,
                   help=f"Combine submodule root (default: {COMBINE_ROOT})")
    p.add_argument("--persist", type=Path, default=DEFAULT_PERSIST,
                   help=f"vectorstore directory (default: {DEFAULT_PERSIST})")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"sentence-transformers model (default: {DEFAULT_MODEL})")
    p.add_argument("--collection", default=DEFAULT_COLLECTION,
                   help=f"Chroma collection name (default: {DEFAULT_COLLECTION})")
    p.add_argument("--chunk-size", type=int, default=1000,
                   help="max characters per chunk (default: 1000)")
    p.add_argument("--chunk-overlap", type=int, default=150,
                   help="characters of overlap between adjacent chunks (default: 150)")
    args = p.parse_args()

    if not args.paper.exists():
        raise SystemExit(f"ERROR: paper file not found: {args.paper}")
    if not args.combine_root.exists():
        raise SystemExit(
            f"ERROR: combine submodule not found at {args.combine_root}. "
            f"Run: git submodule update --init"
        )

    if args.persist.exists():
        print(f"removing existing vectorstore at {args.persist}")
        shutil.rmtree(args.persist)

    all_chunks = []

    print(f"\nloading paper from {args.paper}")
    paper_chunks = load_paper_chunks(args.paper, args.chunk_size, args.chunk_overlap)
    print(f"  -> {len(paper_chunks)} chunks")
    all_chunks.extend(paper_chunks)

    print(f"\nloading code from {args.combine_root}")
    code_chunks = load_code_chunks(args.combine_root, args.chunk_size, args.chunk_overlap)
    print(f"  -> {len(code_chunks)} code chunks total")
    all_chunks.extend(code_chunks)

    print(f"\ntotal: {len(all_chunks)} chunks (chunk_size={args.chunk_size}, overlap={args.chunk_overlap})")

    print(f"\nloading embedding model: {args.model}")
    embeddings = HuggingFaceEmbeddings(model_name=args.model)

    print(f"writing vectorstore to {args.persist}")
    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=str(args.persist),
        collection_name=args.collection,
    )
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
