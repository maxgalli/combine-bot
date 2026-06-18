#!/usr/bin/env python3
"""Retrieval-only evaluation with RAGAS.

================================================================================
WHAT THIS SCRIPT EVALUATES (and what it does NOT)
================================================================================
This script DOES NOT call the bot's answer LLM. It only measures whether
the chunks returned by similarity_search contain the information needed to
answer each question. A judge LLM is used to make that semantic judgement.

If the bot eventually gives a bad answer to a question, run this script first.
If retrieval scores are bad, the LLM never had a chance and you should tune
chunking / embedding / k. If retrieval scores are good but answers are bad,
fall through to eval_answers.py.

================================================================================
METRICS (all judge-LLM based)
================================================================================
  LLMContextPrecisionWithReference
        Of the chunks we retrieved, what fraction were useful for the gold
        answer? Weighted by rank (top hits count more). 0..1, higher is better.

  LLMContextRecall
        Decompose the gold answer into atomic claims; what fraction of those
        claims can be derived from the retrieved chunks? 0..1, higher is better.

================================================================================
USAGE
================================================================================
    export OPENAI_API_KEY=<CERN LiteLLM gateway key>
    uv run scripts/eval_retrieval.py
    uv run scripts/eval_retrieval.py --k 12
    uv run scripts/eval_retrieval.py --questions evals/questions.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_io import save_eval_results  # noqa: E402

DEFAULT_PERSIST = Path("vectorstore")
DEFAULT_COLLECTION = "combine"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_QUESTIONS = Path("evals/questions.yaml")
DEFAULT_JUDGE_MODEL = "gpt-4.1"
DEFAULT_BASE_URL = "https://llmgw-litellm.web.cern.ch/v1"
DEFAULT_K = 8
DEFAULT_RESULTS_DIR = Path("evals/results")


def load_questions(path: Path) -> list[dict]:
    with path.open() as f:
        return yaml.safe_load(f)


def expected_source_rank(docs, expected_source: str | None) -> int | None:
    """Deterministic sanity check: at which rank (1-indexed) did the expected
    source file first appear in the retrieved chunks? Returns None if the
    question has no `source_file` set, or if the source wasn't retrieved."""
    if not expected_source:
        return None
    for i, doc in enumerate(docs, 1):
        source_path = doc.metadata.get("source_path", "")
        if source_path.endswith(expected_source):
            return i
    return None


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--persist", type=Path, default=DEFAULT_PERSIST)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                   help="LLM used by RAGAS to score (does NOT generate answers)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                   help=f"where to save CSV + meta.json (default: {DEFAULT_RESULTS_DIR})")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("ERROR: set OPENAI_API_KEY (CERN LiteLLM gateway key)")
    if not args.persist.exists():
        raise SystemExit(f"ERROR: vectorstore not found at {args.persist}")
    if not args.questions.exists():
        raise SystemExit(f"ERROR: questions file not found at {args.questions}")

    # --- Retrieval step (no LLM here) ---------------------------------------
    print(f"loading vectorstore from {args.persist}")
    hf_embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
    store = Chroma(
        persist_directory=str(args.persist),
        collection_name=args.collection,
        embedding_function=hf_embeddings,
    )

    print(f"loading questions from {args.questions}")
    questions = load_questions(args.questions)
    print(f"  -> {len(questions)} questions\n")

    print(f"retrieving top-{args.k} chunks per question")
    samples: list[SingleTurnSample] = []
    retrieved_docs: list[list] = []  # parallel to samples; used for the source check
    for q in questions:
        docs = store.similarity_search(q["question"], k=args.k)
        retrieved_docs.append(docs)
        samples.append(
            SingleTurnSample(
                user_input=q["question"],
                retrieved_contexts=[d.page_content for d in docs],
                # NOTE: no `response` field — we don't run the bot LLM here.
                reference=q["gold_answer"],
            )
        )
    dataset = EvaluationDataset(samples=samples)

    # --- Judge setup --------------------------------------------------------
    print(f"\nconfiguring judge LLM: {args.judge_model} via {args.base_url}")
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=args.judge_model,
            base_url=args.base_url,
            api_key=os.environ["OPENAI_API_KEY"],
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    # --- Run RAGAS retrieval metrics ----------------------------------------
    print("\nrunning RAGAS retrieval metrics (judge-LLM-based)...\n")
    result = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        show_progress=True,
    )

    # --- Deterministic source-file check (no LLM) ---------------------------
    # For each question with `source_file` set in the YAML, did retrieval
    # bring back at least one chunk from that file? At what rank?
    expected_sources = [q.get("source_file") for q in questions]
    ranks = [
        expected_source_rank(retrieved_docs[i], expected_sources[i])
        for i in range(len(questions))
    ]
    hits = [
        (r is not None) if expected_sources[i] else None
        for i, r in enumerate(ranks)
    ]

    df = result.to_pandas()
    if "retrieved_contexts" in df.columns:
        df = df.drop(columns=["retrieved_contexts"])
    df.insert(0, "id", [q.get("id", "?") for q in questions])
    df["expected_source"] = expected_sources
    df["source_rank"] = ranks
    df["source_hit"] = hits

    # Slim view for the terminal: drop the long-text columns that make the
    # table unreadable. They're still in the CSV written below.
    print("\n=== Retrieval results ===\n")
    display_df = df.drop(
        columns=[c for c in ("user_input", "reference") if c in df.columns]
    )
    print(display_df.to_string(index=False))

    checked = [r for r, exp in zip(ranks, expected_sources) if exp]
    if checked:
        found = [r for r in checked if r is not None]
        print(
            f"\nDeterministic source-file check: "
            f"{len(found)}/{len(checked)} questions had their expected source "
            f"in top-{args.k}"
        )
        if found:
            print(f"  mean rank when found: {sum(found) / len(found):.1f}")
    else:
        print("\n(no questions had `source_file` set — skipping deterministic check)")

    run_meta = {
        "vectorstore": str(args.persist),
        "collection": args.collection,
        "embedding_model": args.embedding_model,
        "judge_model": args.judge_model,
        "bot_model": None,  # retrieval-only run, no bot LLM
        "base_url": args.base_url,
        "k": args.k,
        "questions_file": str(args.questions),
    }
    csv_path, meta_path = save_eval_results(df, "retrieval", run_meta, args.results_dir)
    print(f"\nsaved:\n  {csv_path}\n  {meta_path}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
