#!/usr/bin/env python3
"""Sandboxed Combine command runner.

Executes combine commands inside a captured CMSSW environment without
polluting the bot's own Python process.  Designed to be driven from the
CLI first (Phase A) and later wired into the agent loop in ask.py.

Security model (process-level, not container-level):
  - Binary whitelist: only known Combine executables are allowed.
  - Path classification: sandbox-relative, mountable-absolute (/afs, /eos),
    or rejected.  ``..`` is always rejected.
  - No shell=True anywhere — args are passed as a list.
  - Resource limits via setrlimit (CPU time, virtual memory, file size,
    open file descriptors).
  - Wall-clock timeout via subprocess.run(timeout=...).
  - stdout/stderr truncated to a configurable cap (default 20 KB).

Usage:
    export CMSSW_RELEASE=/path/to/CMSSW_14_1_0_pre4

    uv run scripts/combine_runner.py run \\
        --sandbox sessions/test-1 \\
        --timeout 60 \\
        --command "combine -M FitDiagnostics card.txt -t -1"

    uv run scripts/combine_runner.py inspect-root \\
        --file sessions/test-1/higgsCombineTest.AsymptoticLimits.mH120.root
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shlex
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BINARY_WHITELIST = frozenset(
    {
        "combine",
        "combineTool.py",
        "text2workspace.py",
        "combineCards.py",
    }
)

MOUNTABLE_PREFIXES = (
    "/afs/cern.ch/",
    "/eos/cms/",
    "/eos/user/",
)

# Resource limits applied to the child process.
RLIMIT_CPU_S = 300  # CPU seconds (not wall-clock)
RLIMIT_AS_BYTES = 4 * 1024**3  # 4 GB virtual memory
RLIMIT_FSIZE_BYTES = 512 * 1024**2  # 512 MB max file size
RLIMIT_NOFILE = 1024  # max open file descriptors

OUTPUT_CAP_BYTES = 20 * 1024  # 20 KB — truncate stdout/stderr beyond this

DEFAULT_TIMEOUT_S = 120

# Flags whose next token is an output path (file to be created, not read).
# We still validate traversal and mountable-prefix rules, but skip the
# existence check because the file doesn't exist yet.
OUTPUT_FLAGS = frozenset({"-o", "--out", "--output"})


# ---------------------------------------------------------------------------
# CMSSW environment capture
# ---------------------------------------------------------------------------


def capture_cmssw_env(cmssw_release: Path) -> dict[str, str]:
    """Source ``scram runtime -sh`` in a subshell and return the resulting env.

    The bot's own process never sees CMSSW's PYTHONPATH / LD_LIBRARY_PATH;
    the captured dict is applied only to combine subprocesses via ``env=``.
    """
    src_dir = cmssw_release / "src"
    if not src_dir.is_dir():
        raise SystemExit(
            f"ERROR: CMSSW src directory not found: {src_dir}\n"
            f"Is CMSSW_RELEASE set correctly?"
        )
    out = subprocess.check_output(
        [
            "bash",
            "-c",
            f"cd {shlex.quote(str(src_dir))} && eval $(scram runtime -sh 2>/dev/null) && env -0",
        ],
        text=True,
        timeout=30,
    )
    env = dict(
        item.split("=", 1) for item in out.split("\0") if "=" in item
    )
    if not env:
        raise SystemExit("ERROR: captured CMSSW env is empty — scram failed?")
    return env


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_binary(token: str) -> None:
    """Raise if the first token of the command is not whitelisted."""
    binary = Path(token).name
    if binary not in BINARY_WHITELIST:
        raise ValueError(
            f"Binary not whitelisted: {token!r}\n"
            f"Allowed: {', '.join(sorted(BINARY_WHITELIST))}"
        )


def _looks_like_path(token: str) -> bool:
    """Heuristic: does this token look like a filesystem path?

    Combine commands mix file paths with method names (``AsymptoticLimits``),
    parameter expressions (``r=1``), numeric values (``-1``), and keyword
    arguments.  We only want to validate tokens that plausibly refer to files.

    A token is considered path-like if it:
      - contains a ``/``  (e.g. ``datacards/card.txt``, ``/afs/...``), OR
      - has a file-like extension (e.g. ``card.txt``, ``workspace.root``).

    Tokens containing ``=`` (e.g. ``r=0.2,0.4``, ``r=10``) are combine
    parameter assignments, never file paths.
    """
    if "=" in token:
        return False
    if "/" in token:
        return True
    # Check for a file extension: last component has a dot followed by
    # 1-6 alphanumeric chars (covers .txt, .root, .json, .yaml, .py, ...).
    name = Path(token).name
    if "." in name:
        ext = name.rsplit(".", 1)[-1]
        if ext.isalnum() and 1 <= len(ext) <= 6:
            return True
    return False


def classify_path(token: str, sandbox: Path) -> str | None:
    """Classify a path token. Returns None for non-path tokens (flags, etc.).

    Returns one of:
      "sandbox"   — relative path resolved inside sandbox
      "mountable" — absolute path under an allowed prefix
    Raises ValueError for disallowed paths.
    """
    if token.startswith("-"):
        return None

    # Always reject path traversal, even in non-path-like tokens.
    if ".." in Path(token).parts:
        raise ValueError(f"Path traversal rejected: {token!r}")

    # Always reject disallowed absolute paths.
    if os.path.isabs(token):
        if any(token.startswith(p) for p in MOUNTABLE_PREFIXES):
            return "mountable"
        raise ValueError(
            f"Absolute path not under a mountable prefix: {token!r}\n"
            f"Allowed prefixes: {', '.join(MOUNTABLE_PREFIXES)}"
        )

    if not _looks_like_path(token):
        return None

    # Relative path that looks like a file — will be resolved against sandbox.
    return "sandbox"


def validate_paths(args: list[str], sandbox: Path) -> None:
    """Walk every non-flag token after the binary and validate path semantics.

    Security checks (traversal, mountable prefix) apply to ALL tokens.
    Existence checks are skipped for tokens that follow an output flag
    (``-o``, ``--out``, ``--output``) since those files are about to be
    created.
    """
    is_output = False
    for token in args[1:]:
        # Track whether the previous token was an output flag.
        if token in OUTPUT_FLAGS:
            is_output = True
            continue

        # Tokens that look like --flag=value: validate the value part.
        if "=" in token and token.startswith("-"):
            _, _, value = token.partition("=")
            if value:
                classify_path(value, sandbox)
            is_output = False
            continue

        # Any flag resets is_output (e.g. "-o -M ..." — the -M is not an output path).
        if token.startswith("-"):
            is_output = False
            continue

        kind = classify_path(token, sandbox)
        if not is_output:
            if kind == "sandbox":
                resolved = (sandbox / token).resolve()
                if not resolved.exists():
                    raise ValueError(
                        f"Sandbox file not found: {token!r}\n"
                        f"Looked in: {sandbox}"
                    )
            elif kind == "mountable":
                p = Path(token)
                if not p.exists():
                    parent = p.parent
                    if parent.exists():
                        raise ValueError(
                            f"File not found (but parent dir exists): {token!r}\n"
                            f"Check the filename."
                        )
                    raise ValueError(
                        f"Path not reachable: {token!r}\n"
                        f"If this is an AFS path, check your Kerberos ticket (kinit)."
                    )
        is_output = False


# ---------------------------------------------------------------------------
# Resource-limit helper (applied via preexec_fn)
# ---------------------------------------------------------------------------


def _set_limits() -> None:
    """Called in the child process before exec."""
    resource.setrlimit(resource.RLIMIT_CPU, (RLIMIT_CPU_S, RLIMIT_CPU_S))
    resource.setrlimit(resource.RLIMIT_AS, (RLIMIT_AS_BYTES, RLIMIT_AS_BYTES))
    resource.setrlimit(
        resource.RLIMIT_FSIZE, (RLIMIT_FSIZE_BYTES, RLIMIT_FSIZE_BYTES)
    )
    resource.setrlimit(resource.RLIMIT_NOFILE, (RLIMIT_NOFILE, RLIMIT_NOFILE))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _truncate(text: str, cap: int = OUTPUT_CAP_BYTES) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n... [truncated at {cap} bytes]"


INSPECT_SCRIPT = Path(__file__).parent / "_inspect_root.py"


def run_combine(
    command: str,
    sandbox: Path,
    cmssw_env: dict[str, str],
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> dict:
    """Parse, validate, and execute a combine command.

    Returns a dict with: exit_code, stdout, stderr, runtime_s,
    killed_by_timeout, command (the parsed arg list).
    """
    tokens = shlex.split(command)
    if not tokens:
        return {
            "error": "Empty command",
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "runtime_s": 0.0,
            "killed_by_timeout": False,
            "command": [],
        }

    # --- validation ---
    validate_binary(tokens[0])
    validate_paths(tokens, sandbox)

    # --- execution ---
    killed_by_timeout = False
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            tokens,
            cwd=str(sandbox),
            env=cmssw_env,
            preexec_fn=_set_limits,
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
        exit_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        killed_by_timeout = True
        exit_code = None
        stdout = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    elapsed = time.monotonic() - t0

    return {
        "exit_code": exit_code,
        "stdout": _truncate(stdout),
        "stderr": _truncate(stderr),
        "runtime_s": round(elapsed, 2),
        "killed_by_timeout": killed_by_timeout,
        "command": tokens,
    }


def inspect_root_file(
    path: str,
    cmssw_env: dict[str, str],
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> dict:
    """Run _inspect_root.py as a subprocess inside the CMSSW env.

    Returns the parsed JSON summary, or a dict with an "error" key.
    """
    resolved = Path(path).resolve()
    if not resolved.exists():
        return {"error": f"File not found: {path}"}
    if not resolved.suffix == ".root":
        return {"error": f"Not a .root file: {path}"}

    killed_by_timeout = False
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            ["python3", str(INSPECT_SCRIPT), str(resolved)],
            env=cmssw_env,
            preexec_fn=_set_limits,
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
        elapsed = time.monotonic() - t0
        if proc.returncode != 0:
            return {
                "error": f"Inspector exited with code {proc.returncode}",
                "stderr": _truncate(proc.stderr),
                "runtime_s": round(elapsed, 2),
            }
        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        return {
            "error": "Inspector timed out",
            "runtime_s": round(elapsed, 2),
            "killed_by_timeout": True,
        }
    except json.JSONDecodeError as exc:
        return {"error": f"Failed to parse inspector output: {exc}"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_inspect_root(args: argparse.Namespace, cmssw_env: dict[str, str]) -> int:
    file_path = args.file

    # Apply the same path validation as run_combine.
    if os.path.isabs(file_path):
        if ".." in Path(file_path).parts:
            print(json.dumps({"error": f"Path traversal rejected: {file_path!r}"}), indent=2)
            return 1
        if not any(file_path.startswith(p) for p in MOUNTABLE_PREFIXES):
            print(
                json.dumps(
                    {
                        "error": (
                            f"Absolute path not under a mountable prefix: {file_path!r}\n"
                            f"Allowed prefixes: {', '.join(MOUNTABLE_PREFIXES)}"
                        )
                    },
                    indent=2,
                )
            )
            return 1
    else:
        if ".." in Path(file_path).parts:
            print(json.dumps({"error": f"Path traversal rejected: {file_path!r}"}, indent=2))
            return 1

    result = inspect_root_file(file_path, cmssw_env, timeout_s=args.timeout)
    print(json.dumps(result, indent=2, default=str))
    return 1 if "error" in result else 0


def cmd_run(args: argparse.Namespace, cmssw_env: dict[str, str]) -> int:
    sandbox = Path(args.sandbox).resolve()
    if not sandbox.is_dir():
        print(f"ERROR: sandbox directory does not exist: {sandbox}", file=sys.stderr)
        return 1

    try:
        result = run_combine(
            command=args.command,
            sandbox=sandbox,
            cmssw_env=cmssw_env,
            timeout_s=args.timeout,
        )
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1

    print(json.dumps(result, indent=2))
    return 0 if result.get("exit_code") == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--cmssw-release",
        type=Path,
        default=Path(
            os.environ.get(
                "CMSSW_RELEASE",
                "/afs/cern.ch/work/g/gallim/Postdoc/Combine/combine_bot/CMSSW_14_1_0_pre4",
            )
        ),
        help="Path to CMSSW release (default: $CMSSW_RELEASE or hardcoded fallback)",
    )

    sub = p.add_subparsers(dest="subcmd")

    run_p = sub.add_parser("run", help="Execute a combine command in a sandbox")
    run_p.add_argument(
        "--sandbox",
        type=str,
        required=True,
        help="Working directory (session sandbox)",
    )
    run_p.add_argument(
        "--command",
        type=str,
        required=True,
        help="Full combine command string (parsed via shlex)",
    )
    run_p.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help=f"Wall-clock timeout in seconds (default: {DEFAULT_TIMEOUT_S})",
    )

    inspect_p = sub.add_parser(
        "inspect-root", help="Inspect a ROOT file (list objects, walk workspaces)"
    )
    inspect_p.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the .root file to inspect",
    )
    inspect_p.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help=f"Wall-clock timeout in seconds (default: {DEFAULT_TIMEOUT_S})",
    )

    args = p.parse_args()
    if not args.subcmd:
        p.print_help()
        return 1

    print(f"Capturing CMSSW env from {args.cmssw_release} ...", file=sys.stderr)
    cmssw_env = capture_cmssw_env(args.cmssw_release)
    print(
        f"  captured {len(cmssw_env)} env vars; "
        f"combine = {cmssw_env.get('PATH', '').split(':')[0]}/combine",
        file=sys.stderr,
    )

    if args.subcmd == "run":
        return cmd_run(args, cmssw_env)
    elif args.subcmd == "inspect-root":
        return cmd_inspect_root(args, cmssw_env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
