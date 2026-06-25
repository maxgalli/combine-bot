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
from ask import (  # noqa: E402
    SYSTEM_PROMPT,
    TOOL_PROMPT_SUFFIX,
    TOOL_DEFS,
    VISION_PROMPT_SUFFIX,
    encode_image,
    format_context,
    run_agent_loop,
    stage_files,
)
from eval_io import save_eval_results  # noqa: E402

DEFAULT_PERSIST = Path("vectorstore")
DEFAULT_COLLECTION = "combine"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_QUESTIONS = Path("evals/questions.yaml")
DEFAULT_BOT_MODEL = "gpt-4.1"
DEFAULT_JUDGE_MODEL = "gpt-4.1"
DEFAULT_BASE_URL = "https://llmgw-litellm.web.cern.ch/v1"
DEFAULT_K = 8
DEFAULT_RESULTS_DIR = Path("evals/results")


def load_questions(path: Path) -> list[dict]:
    with path.open() as f:
        return yaml.safe_load(f)


def run_bot(
    question: str,
    store,
    k: int,
    model: str,
    base_url: str,
    image_paths: list[Path] | None = None,
    attach_paths: list[Path] | None = None,
    cmssw_env: dict | None = None,
) -> tuple[str, list]:
    """Reproduce ask.py's RAG call once, non-streaming.

    If image_paths is non-empty, the user message is built as a multimodal
    content list and VISION_PROMPT_SUFFIX is appended to the system prompt —
    matching ask.py's behavior so the eval faithfully tests image-bearing
    questions.

    If attach_paths is non-empty, files are staged into a session sandbox
    and the agent loop (with tool calls) is used instead of a single
    completion call.
    """
    docs = store.similarity_search(question, k=k)
    user_prompt = (
        "Context excerpts:\n\n"
        f"{format_context(docs)}\n\n"
        "---\n\n"
        f"Question: {question}"
    )

    use_tools = bool(attach_paths)
    system_content = SYSTEM_PROMPT
    if use_tools:
        system_content += TOOL_PROMPT_SUFFIX
    if image_paths:
        system_content += VISION_PROMPT_SUFFIX

    if image_paths:
        user_content: list[dict] | str = [{"type": "text", "text": user_prompt}]
        for path in image_paths:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": encode_image(path), "detail": "auto"},
            })
    else:
        user_content = user_prompt

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    if use_tools:
        sandbox = stage_files(attach_paths)
        answer = run_agent_loop(
            messages=messages,
            model=model,
            base_url=base_url,
            tools=TOOL_DEFS,
            sandbox=sandbox,
            cmssw_env=cmssw_env,
            max_rounds=10,
            quiet=False,
            stream=False,
        )
        return answer or "", docs

    response = completion(
        model=model,
        base_url=base_url,
        messages=messages,
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
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                   help=f"where to save CSV + meta.json (default: {DEFAULT_RESULTS_DIR})")
    p.add_argument(
        "--cmssw-release", type=Path,
        default=Path(
            os.environ.get(
                "CMSSW_RELEASE",
                "/afs/cern.ch/work/g/gallim/Postdoc/Combine/combine_bot/CMSSW_14_1_0_pre4",
            )
        ),
        help="Path to CMSSW release (needed for agent-mode questions with attach)",
    )
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

    # Capture CMSSW env once if any question uses attach (agent mode).
    cmssw_env = None
    has_agent_questions = any(q.get("attach") for q in questions)
    if has_agent_questions:
        from combine_runner import capture_cmssw_env
        print(f"capturing CMSSW env from {args.cmssw_release} (agent questions detected)")
        cmssw_env = capture_cmssw_env(args.cmssw_release)
        print(f"  captured {len(cmssw_env)} env vars\n")

    print(f"running bot LLM ({args.bot_model}) on each question...")
    samples: list[SingleTurnSample] = []
    for i, q in enumerate(questions, 1):
        image_paths = [Path(p) for p in q.get("images", [])]
        for img in image_paths:
            if not img.exists():
                raise SystemExit(
                    f"ERROR: image not found for question {q['id']}: {img}"
                )
        attach_paths = [Path(p) for p in q.get("attach", [])]
        for att in attach_paths:
            if not att.exists():
                raise SystemExit(
                    f"ERROR: attached file not found for question {q['id']}: {att}"
                )
        markers = []
        if image_paths:
            markers.append(f"+{len(image_paths)} image(s)")
        if attach_paths:
            markers.append(f"+{len(attach_paths)} attach(s), agent mode")
        marker = f" [{', '.join(markers)}]" if markers else ""
        print(f"  [{i}/{len(questions)}] {q['id']}: {q['question'][:70]}{marker}")
        answer, docs = run_bot(
            q["question"],
            store,
            args.k,
            args.bot_model,
            args.base_url,
            image_paths=image_paths,
            attach_paths=attach_paths if attach_paths else None,
            cmssw_env=cmssw_env,
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

    df = result.to_pandas()
    if "retrieved_contexts" in df.columns:
        df = df.drop(columns=["retrieved_contexts"])
    df.insert(0, "id", [q.get("id", "?") for q in questions])

    # Slim view for the terminal: drop the long-text columns. They're still
    # in the CSV written below.
    print("\n=== Generation results ===\n")
    display_df = df.drop(
        columns=[c for c in ("user_input", "reference", "response") if c in df.columns]
    )
    print(display_df.to_string(index=False))

    run_meta = {
        "vectorstore": str(args.persist),
        "collection": args.collection,
        "embedding_model": args.embedding_model,
        "judge_model": args.judge_model,
        "bot_model": args.bot_model,
        "base_url": args.base_url,
        "k": args.k,
        "questions_file": str(args.questions),
    }
    csv_path, meta_path = save_eval_results(df, "answers", run_meta, args.results_dir)
    print(f"\nsaved:\n  {csv_path}\n  {meta_path}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
