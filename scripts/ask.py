#!/usr/bin/env python3
"""Ask Combine-bot a question.

Retrieves the top-k chunks from the vectorstore, hands them to an LLM
(GPT-4.1 via CERN's LiteLLM gateway by default), and prints the answer
along with the sources Claude was given.

When tools are enabled (``--attach`` or ``--enable-tools``), the bot can
also run combine commands and inspect ROOT files inside a sandboxed
session directory, using a captured CMSSW environment.

Auth:
    Set OPENAI_API_KEY to the key issued by the CERN LiteLLM gateway.

Usage:
    uv run scripts/ask.py "how do I run AsymptoticLimits with toys?"
    uv run scripts/ask.py "..." --k 12 --verbose
    uv run scripts/ask.py "..." --model gpt-4.1-mini

    # With one or more images (PNG, JPG, GIF, WEBP):
    uv run scripts/ask.py "why does my impact plot look strange?" \\
        --image plot.png

    # With attached files + tool use (agent mode):
    uv run scripts/ask.py "my fit fails — what is wrong?" \\
        --attach datacard.txt --attach workspace.root --image error.png
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import shutil
import sys
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from litellm import completion

DEFAULT_PERSIST = Path("vectorstore")
DEFAULT_COLLECTION = "combine"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_BASE_URL = "https://llmgw-litellm.web.cern.ch/v1"
DEFAULT_K = 8
DEFAULT_MAX_ROUNDS = 10
DEFAULT_TOOL_TIMEOUT = 120
SESSIONS_DIR = Path("sessions")
ATTACH_SIZE_CAP = 50 * 1024 * 1024  # 50 MB per file
READ_FILE_CAP = 20 * 1024  # 20 KB for read_attached_file

SYSTEM_PROMPT = """You are an expert on the CMS Combine statistical analysis tool (HiggsAnalysis-CombinedLimit).

Answer the user's question using ONLY the provided context excerpts.

Rules:
- If the context is insufficient to answer, say so explicitly. Do NOT invent option names, command flags, file paths, or behaviors.
- Cite the sources you used inline with their bracket number, e.g. "use the --robustFit option [2]".
- Be precise and concise. Combine users are technical: prefer exact command syntax over hand-waving.
- If sources disagree (e.g. an old forum reply vs. current docs), trust the docs and note the discrepancy.
- When a question has both a documentation answer and a relevant forum thread, mention the forum thread as a real-world example."""

TOOL_PROMPT_SUFFIX = """

You have access to tools that let you run Combine commands and inspect ROOT files.

Tool-use rules:
- You may call list_attached_files to see what files the user provided.
- You may call read_attached_file to read small text files (datacards, logs).
- You may call run_combine to execute a combine command. Prefer reproducing the user's exact command first, then try diagnostic variants (e.g. --robustFit 1, --robustHesse 1, --cminDefaultMinimizerStrategy 1).
- You may call inspect_root_file to examine the structure of a ROOT workspace.
- Do NOT invent file paths — only use files you have confirmed exist via list_attached_files or that were produced by a previous run_combine call.
- If the user's command references files they did not attach, list what is available and ask for the missing files.
- When reporting results, cite the actual tool output (exit codes, stdout, workspace contents). Do not fabricate combine output."""

VISION_PROMPT_SUFFIX = """

The user has attached one or more images.
- Before drawing conclusions from an image, briefly describe what you actually see (axes, curves, legend, error messages, etc.).
- If an image is unclear, cropped, or doesn't show what you'd expect for the question, say so explicitly rather than speculating.
- Treat the image as supplementary evidence: cross-check what it shows against the provided context excerpts before recommending fixes."""


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "list_attached_files",
            "description": (
                "List the files available in the current session sandbox. "
                "Returns a JSON array of objects with name, size_bytes, and "
                "is_text fields."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_attached_file",
            "description": (
                "Read a small attached text file (e.g. a datacard or log). "
                "Maximum 20 KB. Refuses binary files. "
                "The path must be a filename present in the sandbox."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Filename to read (e.g. 'datacard.txt')",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_combine",
            "description": (
                "Execute a Combine command in the session sandbox. "
                "Accepts the full command string verbatim (e.g. "
                "'combine -M FitDiagnostics card.txt -t -1'). "
                "Allowed binaries: combine, combineTool.py, "
                "text2workspace.py, combineCards.py. "
                "Returns exit_code, stdout (≤20 KB), stderr (≤20 KB), "
                "runtime_s, and killed_by_timeout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Full combine command string",
                    },
                    "timeout_s": {
                        "type": "integer",
                        "description": "Wall-clock timeout in seconds (default: 120)",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_root_file",
            "description": (
                "Open a ROOT file and return a structured summary. "
                "Lists top-level objects; for any RooWorkspace found, "
                "enumerates variables (with values/ranges), PDFs, functions, "
                "datasets (with entry counts), named sets (POI, nuisances, "
                "observables), and snapshots. "
                "The path must be a filename in the sandbox or an absolute "
                "path under /afs/cern.ch/ or /eos/."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the .root file",
                    }
                },
                "required": ["path"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers shared with eval scripts (public API — do not rename)
# ---------------------------------------------------------------------------


def encode_image(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None or not mime.startswith("image/"):
        raise SystemExit(
            f"ERROR: cannot determine image MIME type for {path} "
            f"(supported: .png, .jpg, .jpeg, .gif, .webp)"
        )
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


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


# ---------------------------------------------------------------------------
# File staging
# ---------------------------------------------------------------------------


def stage_files(attach_paths: list[Path]) -> Path:
    """Copy attached files into a fresh session sandbox. Returns the sandbox path."""
    session_id = str(uuid.uuid4())[:8]
    sandbox = SESSIONS_DIR / session_id
    sandbox.mkdir(parents=True, exist_ok=True)
    for src in attach_paths:
        if not src.exists():
            raise SystemExit(f"ERROR: attached file not found: {src}")
        size = src.stat().st_size
        if size > ATTACH_SIZE_CAP:
            raise SystemExit(
                f"ERROR: {src} is {size / 1024 / 1024:.1f} MB, "
                f"exceeds {ATTACH_SIZE_CAP / 1024 / 1024:.0f} MB cap"
            )
        shutil.copy2(src, sandbox / src.name)
    return sandbox


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def _is_text_file(path: Path) -> bool:
    """Heuristic: try reading a small sample as UTF-8."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            f.read(1024)
        return True
    except (UnicodeDecodeError, OSError):
        return False


def tool_list_attached_files(sandbox: Path, **_kwargs) -> str:
    files = []
    for p in sorted(sandbox.iterdir()):
        if p.is_file():
            files.append({
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "is_text": _is_text_file(p),
            })
    return json.dumps(files, indent=2)


def tool_read_attached_file(sandbox: Path, path: str, **_kwargs) -> str:
    target = sandbox / Path(path).name  # strip any directory components
    if not target.exists():
        available = [p.name for p in sandbox.iterdir() if p.is_file()]
        return json.dumps({
            "error": f"File not found: {path!r}",
            "available_files": available,
        })
    if not _is_text_file(target):
        return json.dumps({"error": f"Binary file, cannot read: {path}"})
    size = target.stat().st_size
    if size > READ_FILE_CAP:
        text = target.read_text(encoding="utf-8", errors="replace")[:READ_FILE_CAP]
        return text + f"\n... [truncated at {READ_FILE_CAP} bytes, file is {size} bytes]"
    return target.read_text(encoding="utf-8", errors="replace")


def tool_run_combine(sandbox: Path, cmssw_env: dict, command: str,
                     timeout_s: int | None = None, **_kwargs) -> str:
    from scripts.combine_runner import run_combine as _run_combine

    if timeout_s is None:
        timeout_s = DEFAULT_TOOL_TIMEOUT
    try:
        result = _run_combine(
            command=command,
            sandbox=sandbox,
            cmssw_env=cmssw_env,
            timeout_s=timeout_s,
        )
    except ValueError as exc:
        result = {"error": str(exc)}
    return json.dumps(result, indent=2, default=str)


def tool_inspect_root_file(sandbox: Path, cmssw_env: dict, path: str,
                           **_kwargs) -> str:
    from scripts.combine_runner import inspect_root_file as _inspect_root

    # Resolve relative paths against sandbox.
    if not os.path.isabs(path):
        path = str(sandbox / path)
    result = _inspect_root(path=path, cmssw_env=cmssw_env)
    return json.dumps(result, indent=2, default=str)


TOOL_DISPATCH = {
    "list_attached_files": tool_list_attached_files,
    "read_attached_file": tool_read_attached_file,
    "run_combine": tool_run_combine,
    "inspect_root_file": tool_inspect_root_file,
}


def dispatch_tool(name: str, arguments: dict, sandbox: Path,
                  cmssw_env: dict | None) -> str:
    """Execute a tool call and return the result as a string."""
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    return fn(sandbox=sandbox, cmssw_env=cmssw_env, **arguments)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run_agent_loop(
    messages: list[dict],
    model: str,
    base_url: str,
    tools: list[dict],
    sandbox: Path,
    cmssw_env: dict | None,
    max_rounds: int,
    quiet: bool,
    stream: bool,
) -> str | None:
    """Run the LLM in a loop, handling tool calls until a text response."""
    for round_n in range(1, max_rounds + 1):
        response = completion(
            model=model, base_url=base_url, messages=messages, tools=tools
        )
        choice = response.choices[0]
        msg = choice.message

        # If the LLM wants to call tools, execute them and loop.
        if msg.tool_calls:
            # Append the assistant message with tool_calls.
            messages.append(msg.model_dump())

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                if not quiet:
                    print(f"  [tool] {fn_name}({json.dumps(fn_args, separators=(',', ':'))})",
                          file=sys.stderr)

                result_str = dispatch_tool(fn_name, fn_args, sandbox, cmssw_env)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
            continue

        # No tool calls — this is the final text answer.
        answer = msg.content or ""
        return answer

    # Exhausted max rounds — return whatever we have.
    if not quiet:
        print(f"  [warning] reached max tool rounds ({max_rounds})", file=sys.stderr)
    return messages[-1].get("content", "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
    p.add_argument("--image", type=Path, action="append", default=[],
                   help="attach an image to the question (PNG/JPG/GIF/WEBP); "
                        "can be passed multiple times")
    p.add_argument("--image-detail", default="auto", choices=("auto", "low", "high"),
                   help="image resolution detail (default: auto; use 'high' for "
                        "screenshots with text)")

    # Agent / tool-use flags
    p.add_argument("--attach", type=Path, action="append", default=[],
                   help="attach a file (datacard, workspace, log) to the session; "
                        "enables tool use; can be passed multiple times")
    p.add_argument("--enable-tools", action="store_true",
                   help="enable tool use even without --attach (e.g. to inspect "
                        "files on AFS)")
    p.add_argument("--no-tools", action="store_true",
                   help="force one-shot mode (no tool use) even with --attach")
    p.add_argument("--max-tool-rounds", type=int, default=DEFAULT_MAX_ROUNDS,
                   help=f"max LLM↔tool round-trips (default: {DEFAULT_MAX_ROUNDS})")
    p.add_argument(
        "--cmssw-release", type=Path,
        default=Path(
            os.environ.get(
                "CMSSW_RELEASE",
                "/afs/cern.ch/work/g/gallim/Postdoc/Combine/combine_bot/CMSSW_14_1_0_pre4",
            )
        ),
        help="Path to CMSSW release (default: $CMSSW_RELEASE)",
    )

    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("ERROR: set OPENAI_API_KEY (the CERN LiteLLM gateway key)")
    if not args.persist.exists():
        raise SystemExit(
            f"ERROR: vectorstore not found at {args.persist}. "
            f"Run scripts/build_index.py first."
        )

    # --- Decide whether tools are active ---
    use_tools = (args.attach or args.enable_tools) and not args.no_tools

    # --- Stage attached files ---
    sandbox = None
    cmssw_env = None
    if use_tools:
        if args.attach:
            sandbox = stage_files(args.attach)
            if not args.quiet:
                print(f"=== Session sandbox: {sandbox} ===\n", file=sys.stderr)
        else:
            # --enable-tools without --attach: create an empty sandbox
            sandbox = stage_files([])

        # Capture CMSSW env once.
        from scripts.combine_runner import capture_cmssw_env
        if not args.quiet:
            print(f"Capturing CMSSW env from {args.cmssw_release} ...",
                  file=sys.stderr)
        cmssw_env = capture_cmssw_env(args.cmssw_release)
        if not args.quiet:
            print(f"  captured {len(cmssw_env)} env vars\n", file=sys.stderr)

    # --- Retrieval ---
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

    for img in args.image:
        if not img.exists():
            raise SystemExit(f"ERROR: image not found: {img}")

    if not args.quiet:
        print_sources(docs, verbose=args.verbose)
        if args.image:
            print(f"=== Attached images ({len(args.image)}) ===\n")
            for img in args.image:
                kb = img.stat().st_size / 1024
                print(f"  {img} ({kb:.1f} KB)")
            print()
        if args.attach:
            print(f"=== Attached files ({len(args.attach)}) ===\n")
            for att in args.attach:
                kb = att.stat().st_size / 1024
                print(f"  {att.name} ({kb:.1f} KB)")
            print()
        print("=== Answer ===\n")

    # --- Build messages ---
    user_prompt = (
        "Context excerpts:\n\n"
        f"{format_context(docs)}\n\n"
        "---\n\n"
        f"Question: {args.question}"
    )

    system_content = SYSTEM_PROMPT
    if use_tools:
        system_content += TOOL_PROMPT_SUFFIX
    if args.image:
        system_content += VISION_PROMPT_SUFFIX

    if args.image:
        user_content: list[dict] | str = [{"type": "text", "text": user_prompt}]
        for img in args.image:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": encode_image(img), "detail": args.image_detail},
            })
    else:
        user_content = user_prompt

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    # --- Generate answer ---
    if use_tools:
        # Agent mode: non-streaming loop with tool calls.
        answer = run_agent_loop(
            messages=messages,
            model=args.model,
            base_url=args.base_url,
            tools=TOOL_DEFS,
            sandbox=sandbox,
            cmssw_env=cmssw_env,
            max_rounds=args.max_tool_rounds,
            quiet=args.quiet,
            stream=not args.no_stream,
        )
        if answer:
            print(answer)
    elif args.no_stream:
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
