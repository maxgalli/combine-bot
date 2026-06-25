#!/usr/bin/env python3
"""Combine-bot web application.

A FastAPI server that wraps the existing ask.py RAG + agent pipeline
behind an HTTP endpoint. Serves a single-page frontend and exposes a
POST /ask API.

Usage:
    export OPENAI_API_KEY=<CERN LiteLLM gateway key>
    export CMSSW_RELEASE=/path/to/CMSSW_14_1_0_pre4  # optional
    uv run uvicorn app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Make scripts/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from ask import (
    DEFAULT_BASE_URL,
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_K,
    DEFAULT_MODEL,
    DEFAULT_PERSIST,
    SYSTEM_PROMPT,
    TOOL_DEFS,
    TOOL_PROMPT_SUFFIX,
    VISION_PROMPT_SUFFIX,
    encode_image,
    format_context,
    print_sources,
    run_agent_loop,
    source_tag,
    source_url,
    stage_files,
)
from combine_runner import capture_cmssw_env

# ---------------------------------------------------------------------------
# App startup — load heavy resources once
# ---------------------------------------------------------------------------

app = FastAPI(title="Combine Bot")

# Serve the frontend
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# These are populated in the lifespan startup event.
_store = None
_cmssw_env = None


@app.on_event("startup")
def startup():
    global _store, _cmssw_env

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before starting the server")

    # Load vectorstore.
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    persist = DEFAULT_PERSIST
    if not persist.exists():
        raise RuntimeError(f"Vectorstore not found at {persist}")

    print(f"Loading embedding model: {DEFAULT_EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
    _store = Chroma(
        persist_directory=str(persist),
        collection_name=DEFAULT_COLLECTION,
        embedding_function=embeddings,
    )
    print(f"Vectorstore loaded from {persist}")

    # Capture CMSSW env.
    cmssw_release = Path(
        os.environ.get(
            "CMSSW_RELEASE",
            "/afs/cern.ch/work/g/gallim/Postdoc/Combine/combine_bot/CMSSW_14_1_0_pre4",
        )
    )
    print(f"Capturing CMSSW env from {cmssw_release}")
    _cmssw_env = capture_cmssw_env(cmssw_release)
    print(f"  captured {len(_cmssw_env)} env vars")

    print("\nServer ready.\n")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()


@app.post("/ask")
async def ask(
    question: str = Form(...),
    files: list[UploadFile] = File(default=[]),
):
    """Accept a question + optional file uploads, return the bot's answer."""
    from litellm import completion

    # --- Stage uploaded files ---
    attach_paths: list[Path] = []
    image_paths: list[Path] = []
    sandbox = None

    if files:
        # Write uploads to a temp sandbox.
        import shutil
        import uuid

        sessions_dir = Path("sessions")
        session_id = str(uuid.uuid4())[:8]
        sandbox = sessions_dir / session_id
        sandbox.mkdir(parents=True, exist_ok=True)

        for f in files:
            if f.filename and f.size and f.size > 0:
                dest = sandbox / f.filename
                with open(dest, "wb") as out:
                    content = await f.read()
                    out.write(content)

                # Classify as image or attachment.
                lower = f.filename.lower()
                if any(lower.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp")):
                    image_paths.append(dest)
                else:
                    attach_paths.append(dest)

    use_tools = bool(attach_paths)

    # --- Retrieval ---
    docs = _store.similarity_search(question, k=DEFAULT_K)

    sources = []
    for i, doc in enumerate(docs, 1):
        sources.append({
            "index": i,
            "tag": source_tag(doc),
            "url": source_url(doc),
        })

    # --- Build messages ---
    user_prompt = (
        "Context excerpts:\n\n"
        f"{format_context(docs)}\n\n"
        "---\n\n"
        f"Question: {question}"
    )

    system_content = SYSTEM_PROMPT
    if use_tools:
        system_content += TOOL_PROMPT_SUFFIX
    if image_paths:
        system_content += VISION_PROMPT_SUFFIX

    if image_paths:
        user_content: list[dict] | str = [{"type": "text", "text": user_prompt}]
        for img in image_paths:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": encode_image(img), "detail": "auto"},
            })
    else:
        user_content = user_prompt

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    # --- Generate answer ---
    tool_log: list[str] = []

    if use_tools:
        # Patch to capture tool calls for the UI.
        original_dispatch = None
        from ask import dispatch_tool as _dispatch_tool

        def logging_dispatch(name, arguments, sandbox, cmssw_env):
            tool_log.append(f"{name}({json.dumps(arguments, separators=(',', ':'))})")
            return _dispatch_tool(name, arguments, sandbox, cmssw_env)

        # Temporarily monkey-patch — not great, but avoids refactoring
        # run_agent_loop's internals for now.
        import ask as ask_module
        original = ask_module.dispatch_tool
        ask_module.dispatch_tool = logging_dispatch

        try:
            answer = run_agent_loop(
                messages=messages,
                model=DEFAULT_MODEL,
                base_url=DEFAULT_BASE_URL,
                tools=TOOL_DEFS,
                sandbox=sandbox,
                cmssw_env=_cmssw_env,
                max_rounds=10,
                quiet=True,
                stream=False,
            )
        finally:
            ask_module.dispatch_tool = original

        answer = answer or ""
    else:
        response = completion(
            model=DEFAULT_MODEL,
            base_url=DEFAULT_BASE_URL,
            messages=messages,
        )
        answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": sources,
        "tool_calls": tool_log,
        "sandbox": str(sandbox) if sandbox else None,
    }
