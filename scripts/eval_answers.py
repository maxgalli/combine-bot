#!/usr/bin/env python3
"""End-to-end (retrieval + generation) evaluation with RAGAS.

================================================================================
WHAT THIS SCRIPT EVALUATES (and what it does NOT)
================================================================================
This script DOES call the bot's answer LLM. For each question it runs the
full pipeline (retrieve top-k chunks, format the prompt the way ask.py does,
call the LLM to get an answer), then asks a SECOND LLM (the "judge") to
score that answer.

Two LLM roles, with separate flags so you can mix them:
    --bot-model     same model ask.py uses; generates the answers under test
    --judge-model   the RAGAS judge; scores the answers

If retrieval is bad, these scores will look bad too — but you won't know
which side is to blame. Run scripts/eval_retrieval.py first to isolate.

================================================================================
METRICS
================================================================================
  Faithfulness
        Does the bot's answer make claims not supported by the retrieved
        chunks? 0..1, higher is better. Direct hallucination measure.

  ResponseRelevancy
        Does the answer actually address the question, or wander? 0..1.

  FactualCorrectness
        Substance match against the gold answer (precision + recall on
        claims). 0..1, higher is better. Requires `gold_answer`.

================================================================================
USAGE
================================================================================
    export OPENAI_API_KEY=<CERN LiteLLM gateway key>
    uv run scripts/eval_answers.py
    uv run scripts/eval_answers.py --bot-model gpt-4.1-mini --judge-model gpt-4.1
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
from litellm import completion
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, FactualCorrectness, ResponseRelevancy

# Reuse the bot's system prompt and context formatter from ask.py.
# scripts/ is the script's directory, so plain import works under uv run.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask import SYSTEM_PROMPT, format_context  # noqa: E402

DEFAULT_PERSIST = Path("vectorstore")
DEFAULT_COLLECTION = "combine"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_QUESTIONS = Path("evals/questions.yaml")
DEFAULT_BOT_MODEL = "gpt-4.1"
DEFAULT_JUDGE_MODEL = "gpt-4.1"
DEFAULT_BASE_URL = "https://llmgw-litellm.web.cern.ch/v1"
DEFAULT_K = 8


def load_questions(path: Path) -> list[dict]:
    with path.open() as f:
        return yaml.safe_load(f)


def run_bot(question: str, store, k: int, model: str, base_url: str) -> tuple[str, list]:
    """Reproduce ask.py's RAG call once, non-streaming."""
    docs = store.similarity_search(question, k=k)
    user_prompt = (
        "Context excerpts:\n\n"
        f"{format_context(docs)}\n\n"
        "---\n\n"
        f"Question: {question}"
    )
    response = completion(
        model=model,
        base_url=base_url,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content, docs


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--persist", type=Path, default=DEFAULT_PERSIST)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--bot-model", default=DEFAULT_BOT_MODEL,
                   help="LLM that generates the answer under test (mirrors ask.py)")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                   help="LLM that scores the answer (RAGAS metrics)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("ERROR: set OPENAI_API_KEY (CERN LiteLLM gateway key)")
    if not args.persist.exists():
        raise SystemExit(f"ERROR: vectorstore not found at {args.persist}")
    if not args.questions.exists():
        raise SystemExit(f"ERROR: questions file not found at {args.questions}")

    # --- Retrieval + generation step (LLM #1) -------------------------------
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

    print(f"running bot LLM ({args.bot_model}) on each question...")
    samples: list[SingleTurnSample] = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['id']}: {q['question'][:70]}")
        answer, docs = run_bot(
            q["question"], store, args.k, args.bot_model, args.base_url
        )
        samples.append(
            SingleTurnSample(
                user_input=q["question"],
                retrieved_contexts=[d.page_content for d in docs],
                response=answer,
                reference=q["gold_answer"],
            )
        )
    dataset = EvaluationDataset(samples=samples)

    # --- Judge setup (LLM #2) -----------------------------------------------
    print(f"\nconfiguring judge LLM: {args.judge_model} via {args.base_url}")
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=args.judge_model,
            base_url=args.base_url,
            api_key=os.environ["OPENAI_API_KEY"],
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    # --- Run RAGAS generation metrics ---------------------------------------
    print("\nrunning RAGAS generation metrics (judge-LLM-based)...\n")
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            FactualCorrectness(),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        show_progress=True,
    )

    print("\n=== Generation results ===\n")
    df = result.to_pandas()
    for col in ("retrieved_contexts", "response", "reference"):
        if col in df.columns:
            df = df.drop(columns=[col])
    print(df.to_string(index=False))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
