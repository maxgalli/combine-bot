# Combine-bot

A retrieval-augmented Q&A bot for the CMS Combine statistical analysis tool.
Built on top of:

- a vector knowledge base over the Combine paper, source code, official
  docs, and the cms-talk Statistics-category forum,
- GPT-4.1 via CERN's LiteLLM gateway,
- a RAGAS-based evaluation suite.

For the design rationale, the tuning history, and the planned
agent / tool-use extension (combine runner, ROOT inspector), see
[`conversation_log.md`](./conversation_log.md). This README is the
operational runbook.

---

## 0. Prerequisites

Install once per machine.

```bash
# uv — Python package + interpreter manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# git (probably already there)
git --version

# Optional but recommended for extracting forum-attached PDF pages
brew install poppler          # macOS
# or: sudo apt install poppler-utils
```

The bot itself doesn't need Python pre-installed — `uv` pins and fetches
3.11 automatically (per `pyproject.toml`'s `requires-python = ">=3.11,<3.12"`).

---

## 1. Clone the repo (with the Combine submodule)

```bash
git clone --recurse-submodules <repo-url> combine-bot
cd combine-bot
```

If you cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

Verify the submodule is checked out at the pinned tag `v10.6.0`:

```bash
git -C knowledge_base/combine describe --tags
# expected: v10.6.0
```

---

## 2. Create the Python environment

```bash
uv sync
```

This:

- Picks Python 3.11 (per `pyproject.toml`).
- Creates `.venv/` in the project root.
- Installs all locked deps from `uv.lock` (langchain, chromadb,
  sentence-transformers, ragas, litellm, etc.).
- First run downloads a few hundred MB of wheels (torch, transformers, ...).

---

## 3. Stage the knowledge sources

The corpus has four parts. Two are already in the repo; two you need to
populate.

### 3a. Paper — already in the repo

`knowledge_base/paper/paper_clean.txt` is committed.

### 3b. Combine source — via the submodule

Confirmed already by step 1. The code lives under
`knowledge_base/combine/{python,scripts,interface,bin,src}` and the docs
under `knowledge_base/combine/docs/`. The build script reads from these
directly.

### 3c. Forum (`knowledge_base/forum/topic_*.json`)

The forum scrape isn't committed — it's user-data and large. You either:

**Option A — Copy the existing scrape from another machine** (fastest):

```bash
rsync -avh dev-machine:~/combine-bot/knowledge_base/forum/ \
       knowledge_base/forum/
```

**Option B — Run the scraper from scratch.**

You need a cms-talk session cookie:

1. Log into <https://cms-talk.web.cern.ch> in your browser via CERN SSO.
2. Devtools → Application → Cookies → `https://cms-talk.web.cern.ch`.
3. Copy the values of **both** `_forum_session` and `_t`.
4. Export them semicolon-joined:

   ```bash
   export DISCOURSE_COOKIE='_forum_session=<v>; _t=<v>'
   ```

   `_t` alone is NOT enough — Discourse rejects `.json` API calls without
   `_forum_session`.

Then:

```bash
uv run scripts/scrape_forum.py            # incremental (or first-ever run)
uv run scripts/scrape_forum.py --full     # re-scrape everything (catches edits)
uv run scripts/scrape_forum.py --limit 3  # smoke test (3 topics only)
```

Initial scrape over the full Statistics category (~1100 topics):
~20–30 min wall-clock at the default 0.5 s/request politeness sleep.
Subsequent runs are incremental and finish in under a minute.

Session cookies expire with CERN SSO (days to weeks). If you start seeing
403s mid-run, re-grab the cookie and re-run — the script keeps a manifest
in `knowledge_base/forum/.manifest.json` so it picks up where it left off.

### 3d. (Optional) Forum images for image-bearing eval questions

If an eval question references an image (`images:` field in
`evals/questions.yaml`), the file needs to exist locally. For PDFs
attached to forum threads, extract the relevant page with `pdftoppm`:

```bash
mkdir -p knowledge_base/forum/images
pdftoppm -f 2 -l 2 -r 150 -png \
    Regularization_spike.pdf \
    knowledge_base/forum/images/topic_142937_page
# Produces knowledge_base/forum/images/topic_142937_page-2.png; rename if needed.
```

---

## 4. Build the vector index

```bash
uv run scripts/build_index.py
```

What happens:

- Walks all four corpora and chunks per source-specific knobs.
- Embeds with the default model `BAAI/bge-base-en-v1.5` (first call
  downloads the model, ~430 MB; cached under `~/.cache/huggingface/`).
- Writes the Chroma store to `vectorstore/`.

Time: ~10 min on CPU for ~10 k chunks. Watch:

```bash
top -pid $(pgrep -f build_index.py)
```

Expected console output (numbers will vary slightly with the forum scrape size):

```
loading paper from knowledge_base/paper/paper_clean.txt
  -> 253 chunks
loading code from knowledge_base/combine
  python/**/*.py: 61 files -> 1514 chunks
  ...
loading docs from knowledge_base/combine/docs
  pages from nav: 28 used, 0 missing, 3 non-md skipped
  -> 1520 docs chunks
loading forum from knowledge_base/forum
  topics processed: 549  (solved: 36, short replies dropped: 24)
  -> 6654 forum chunks
total: 10766 chunks ...
writing vectorstore to vectorstore
done.
```

Disk footprint: ~150–200 MB under `vectorstore/`. Rebuild any time with
the same command (it wipes and recreates).

### Per-source chunking knobs (override only when tuning)

```bash
uv run scripts/build_index.py \
    --chunk-size-docs 1500 --chunk-overlap-docs 300 \
    --chunk-size-code 1200
```

Current defaults (in `scripts/build_index.py`):

```
paper:  size=1000  overlap=150
code:   size=1000  overlap=150
docs:   size=1000  overlap=300
forum:  size=1000  overlap=150
```

---

## 5. Configure the LLM gateway key

The bot calls GPT-4.1 via CERN's LiteLLM gateway. Get an API key from
the gateway team and export it:

```bash
export OPENAI_API_KEY='<CERN LiteLLM gateway key>'
```

Stash it in your shell rc (`~/.zshrc` / `~/.bashrc`) or use
`direnv` / a `.envrc`. **Don't commit the key.** The default gateway URL
is hard-coded in `ask.py` (`https://llmgw-litellm.web.cern.ch/v1`) and
can be overridden per call with `--base-url`.

---

## 6. Ask the bot a question

```bash
uv run scripts/ask.py "what does --robustFit do?"
```

Streaming reply, with the retrieved sources block first. Useful flags:

```bash
--k 12                          # retrieve more chunks
--verbose                       # show chunk previews in the sources block
--quiet                         # skip the sources block, answer only
--no-stream                     # wait for the full response
--image PATH                    # attach an image (multi-allowed)
--image-detail high             # for OCR-heavy screenshots
--model gpt-4.1-mini            # cheaper alternative model
```

---

## 7. Run the evaluation suite

```bash
uv run scripts/eval_retrieval.py    # retrieval-only metrics, ~30 s
uv run scripts/eval_answers.py      # full pipeline, ~3 min
```

Both:

- Read questions from `evals/questions.yaml`.
- Run RAGAS metrics + (for retrieval) a deterministic source-file check.
- Save results to
  `evals/results/<timestamp>-{retrieval|answers}.{csv,meta.json}`.

Inspect results:

```bash
ls -t evals/results/ | head
column -ts, < evals/results/<latest>-retrieval.csv | less -S

# Track a metric across runs:
ls evals/results/*-retrieval.meta.json | while read f; do
  echo "$f: $(jq -r '.timestamp + " " + .embedding_model + " ctx_recall=" + (.aggregates.context_recall|tostring)' "$f")"
done
```

### Adding an eval question

Edit `evals/questions.yaml`. Each entry needs `id`, `question`,
`gold_answer`. Optional: `category`, `source_file` (str or list),
`images`, `topic_url`. Schema is documented at the top of the YAML.

---

## 8. VM-only: install CMSSW + Combine for the runner

> Required only on the deploy VM where the bot will actually execute
> combine commands. Skip on dev machines.

### 8a. Install CMSSW with Combine

Follow the official Combine install instructions for the chosen CMSSW
release (typically `CMSSW_14_X_X` at the time of writing — match what's
documented at
<https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/>).

Pin the install location and remember the path. The runner will source
its env from `<CMSSW>/src/`.

```bash
ls /opt/cmssw/CMSSW_14_X_X/src/HiggsAnalysis/CombinedLimit
```

### 8b. Set up AFS / EOS access

The bot will be passed absolute paths like `/afs/cern.ch/user/.../card.txt`
in pasted commands. To read those, the bot user needs Kerberos tickets
and AFS tokens.

```bash
# One-time: create a keytab for the bot's service account.
ktutil
addent -password -p combine-bot@CERN.CH -k 1 -e aes256-cts-hmac-sha1-96
wkt /var/lib/combine-bot/keytab/combine-bot.keytab

# Periodic renewal (cron or systemd timer):
kinit -k -t /var/lib/combine-bot/keytab/combine-bot.keytab combine-bot@CERN.CH
aklog
```

Plan to renew every 8 hours via `k5start` / `krenew` or a cron. The
runner will report `"file exists but is not readable"` on expired tokens;
that's the signal renewal is needed.

### 8c. Pointer for the runner

The runner reads CMSSW location from one env var (or its default
constant):

```bash
export CMSSW_RELEASE=/opt/cmssw/CMSSW_14_X_X
```

The runner does **not** live inside the CMSSW area; it lives in the bot
repo, captures CMSSW's env once at startup, and applies that env only to
combine subprocess calls. The bot's own Python (uv-managed) stays
isolated. See `conversation_log.md` §14 for the design rationale.

> **Note**: the agent/tool-use code (`scripts/combine_runner.py`, the
> agent loop in `ask.py`) is **not yet implemented** at the time of
> writing. See `conversation_log.md` §16 for the planned next step.

### 8d. Systemd service (when ready to deploy as a daemon)

A draft unit (`/etc/systemd/system/combine-bot.service`):

```ini
[Unit]
Description=Combine-bot agent
After=network.target

[Service]
User=combine-bot
Group=combine-bot
WorkingDirectory=/home/combine-bot/combine-bot
Environment=CMSSW_RELEASE=/opt/cmssw/CMSSW_14_X_X
EnvironmentFile=/etc/combine-bot.env       # OPENAI_API_KEY, DISCOURSE_COOKIE, ...
ExecStart=/home/combine-bot/.local/bin/uv run scripts/<entrypoint>.py
Restart=on-failure
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/home/combine-bot/combine-bot/sessions /tmp/combine-bot

[Install]
WantedBy=multi-user.target
```

The `<entrypoint>` is TBD — depends on whether the bot ends up as a
forum-polling daemon or stays CLI-only.

---

## 9. Common operations cheat sheet

```bash
# Refresh forum + rebuild index (catches new threads):
uv run scripts/scrape_forum.py
uv run scripts/build_index.py

# Smoke test the bot on one question:
uv run scripts/ask.py "how do I run AsymptoticLimits?"

# Check the gateway supports image input on this VM:
uv run scripts/test_vision.py knowledge_base/combine/docs/logo.png

# Retrieval-only diagnostic for a specific query (no LLM):
uv run scripts/query_index.py "..." --k 8 --full

# Full eval:
uv run scripts/eval_retrieval.py && uv run scripts/eval_answers.py

# Tabulate context_recall across all retrieval runs:
ls evals/results/*-retrieval.meta.json | while read f; do
  jq -r '[.timestamp, .embedding_model, .aggregates.context_recall|tostring] | @tsv' "$f"
done | sort
```

---

## 10. Troubleshooting

**`uv sync` fails on torch / chromadb wheels** — usually a transient
PyPI issue; retry. If persistent, check Python version (must be 3.11.x).

**`build_index.py` seems frozen** — embedding step on CPU with bge-base
is ~10 min for the full corpus, no progress bar. Confirm via `top` that
the process is CPU-bound. If `chroma.sqlite3` is created but no UUID
subdirectory appears, the embedding loop didn't finish (OOM or killed) —
check `dmesg | tail` for OOM messages.

**Bot cites sources but they're empty / generic** — vectorstore hasn't
been built or doesn't match the embedding model the bot is configured
with. Rebuild:

```bash
rm -rf vectorstore/
uv run scripts/build_index.py
```

**Forum scrape gets 403 immediately** — session cookie missing or
expired. Re-grab `_forum_session + _t` and re-export `DISCOURSE_COOKIE`.
Single quotes are mandatory to keep the `;` from being interpreted by
the shell.

**Forum scrape gets 403 partway through** — SSO session expired
mid-run. Re-grab cookie, re-run. The manifest preserves progress.

**`OPENAI_API_KEY` not set** in `ask.py` / eval scripts — the gateway
key isn't in this shell's env. `export OPENAI_API_KEY='...'` (or set it
in `~/.zshrc`).

**RAGAS 0.4 fails to import** — known incompatibility with current
`langchain-community`. The project pins `ragas>=0.2,<0.3` and
`langchain-community<0.4` deliberately. Don't loosen these without
testing.

**On the VM: combine subprocess errors with "command not found"** — the
runner hasn't captured the CMSSW env. Verify `CMSSW_RELEASE` points at
the right path, and that `eval $(scram runtime -sh)` works manually
from `$CMSSW_RELEASE/src/`.

**On the VM: combine fails with "Permission denied" on `/afs/...`** —
Kerberos tickets expired. `kinit` from the keytab, `aklog`, retry. Set
up automatic renewal.
