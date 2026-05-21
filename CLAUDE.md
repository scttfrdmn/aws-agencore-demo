# CLAUDE.md

Guidance for Claude Code working in this repo. Read this fully before writing code.

## What this is

**Inside the Lines** is a five-minute *live demo* for a 30-minute conference talk
on AWS Bedrock, given to a university research-computing audience. The thesis of
the talk: you can give researchers frontier-model AI agents **without the data
leaving a secure environment**, and pay pennies for it.

The demo drives one biomedical-research agent through three escalating questions
about the gene *PCSK9*, against a Bedrock Knowledge Base of ~1,000 open-access
papers, and shows — in a **local web page, live** — what the agent is doing and
what it costs.

This is a demo, not a product. Optimise for: legibility on a projector,
reliability of a live run, and honest cost numbers.

## Current state — what is seeded, what you build

**Seeded and working (do not rewrite without reason):**
- `corpus_fetch.py` — pulls the PMC paper corpus, licence-filtered.
- `build_kb.py` — provisions the Bedrock Knowledge Base (S3 Vectors store).
- `teardown.py` — deletes billable resources.
- `src/agentcore_demo/questions.py` — the three locked questions + system prompts.
- `src/agentcore_demo/cost.py` — the cost meter (pure logic).
- `src/agentcore_demo/aws.py` — AWS backend: `retrieve`, `converse`, `code_interpreter_run`.
- `src/agentcore_demo/agent.py` — orchestration; runs the questions, emits events.

**Your job — build these:**
- `src/agentcore_demo/app.py` — FastAPI app: a `/ws` WebSocket, serves the static page,
  auto-opens the browser. A stub with a spec docstring is in place.
- `src/agentcore_demo/static/index.html` — the live page (Alpine.js, no build step).
- `tests/` — fill out the suite. `tests/test_cost.py` and `tests/conftest.py`
  are seeded as the pattern; `tests/test_agent.py` is a stub.

See `INITIAL_PROMPT.md` for the first milestone.

## Architecture

```
corpus_fetch.py ─► S3 bucket ─► build_kb.py ─► Bedrock Knowledge Base (S3 Vectors)
                                                        │
  browser ◄── WebSocket ── app.py (FastAPI) ── agent.py ─┤── Bedrock (Claude + Nova)
  (Alpine.js, live)         emits events                 └── AgentCore Code Interpreter
```

The **event protocol** is the contract between `agent.py` and `app.py`. The agent
calls an `emit(event: dict)` callback; the web layer pushes each event over the
WebSocket; the page renders it. Event shapes (see `agent.py` for the source of
truth):

| `type` | fields | meaning |
|---|---|---|
| `question` | `n`, `text` | a new question started |
| `phase` | `label` | status line ("retrieving...") |
| `retrieval` | `count` | N passages retrieved |
| `model` | `tier`, `label`, `state` (`start`/`done`), `usage?`, `cost?` | a model call |
| `answer` | `title`, `text` | a synthesis / adjudication result |
| `code` | `text` | generated analysis code |
| `chart` | `data` (base64 PNG) | the chart, for inline `<img>` |
| `cost` | `total` | running cost-meter total |
| `route` | `path` (`SYNTHESIS`\|`ANALYSIS`\|`DEBATE`), `label` | free-form routing result; emitted before the question event |
| `setup_cost` | `ingestion_usd`, `storage_usd_per_month` | KB panel costs: ingestion computed from corpus size, storage from vector count (NOT metered, NOT in run total) |
| `receipt` | `rows`, `total` | the final itemised receipt |
| `done` | — | run complete |

Keep this protocol stable. If the page needs more, add a field; don't repurpose.

## Verified AWS facts (do not re-derive — these were checked)

- **Retrieval**: `boto3.client("bedrock-agent-runtime").retrieve(knowledgeBaseId,
  retrievalQuery={"text": q}, retrievalConfiguration={"vectorSearchConfiguration":
  {"numberOfResults": n}})` → `retrievalResults[].content.text` / `.location` / `.score`.
- **Reasoning**: `boto3.client("bedrock-runtime").converse(modelId, system=[{"text":...}],
  messages=[...], inferenceConfig={"maxTokens":..., "temperature":...})`
  → `output.message.content[0].text` and a `usage` block with `inputTokens` /
  `outputTokens`. `converse` works uniformly for Claude **and** Amazon Nova.
- **Code Interpreter**: the `bedrock_agentcore` SDK —
  `from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter`;
  `ci = CodeInterpreter(region); ci.start(); ci.invoke("executeCode",
  {"language":"python","code":...}); ci.stop()`. Result events stream in
  `resp["stream"]`, each with `result.content[]` blocks of `{"type":"text",...}`.
- **Vector store**: the KB uses **Amazon S3 Vectors** (GA, serverless, no hourly
  cost) — not OpenSearch. `build_kb.py` creates an S3 vector bucket + index via
  the `s3vectors` boto3 client and connects it.
- **VERIFY before running `build_kb.py`**: the exact `s3vectors` method names and
  the `create_knowledge_base` `storageConfiguration` shape for S3 Vectors are the
  newest APIs here. Check current boto3 docs and adjust; the file flags the spots.

## Conventions

- Python 3.11, `src/` layout, package is `agentcore_demo`. `pip install -e ".[dev]"`.
- **Lint/format: `ruff`** must be clean (`ruff check .` and `ruff format --check .`).
- **Tests: `pytest`** must pass. CI runs both on every push.
- **No AWS calls in tests.** `agent.py` takes an AWS backend by dependency
  injection; tests pass a fake (see `tests/conftest.py`). Never hit the cloud in CI.
- WebSocket/async code uses `pytest-asyncio` (`asyncio_mode = "auto"`).
- Keep functions small; type-hint public functions; docstrings on modules and
  non-obvious functions.

## Guardrails — do not violate

- **No console.** Everything is scripted. Don't add "go click in the AWS console"
  steps anywhere.
- **No real identifiers committed.** `config.py` is git-ignored; only
  `config.example.py` is tracked. Never hard-code account IDs, bucket names, or
  KB IDs.
- **Always provide teardown.** Any AWS resource a script creates, `teardown.py`
  must be able to delete.
- **The three questions are locked** (`questions.py`). They are rehearsed for the
  live talk; do not reword them.
- **Cost numbers must be real.** The cost meter computes from actual `usage`
  tokens × the rates in `config.py`. Never fabricate or hard-code a total.
- **Frontend: lightweight.** Alpine.js via CDN, plain HTML/CSS, no build step,
  no npm, no bundler. The page must open by just loading a file the FastAPI app
  serves.

## The demo's three beats (for context, so the UI tells the right story)

1. *Friction gone* — one plain question; Claude Haiku reads and cites.
2. *Real work, faster* — Claude Sonnet writes analysis code; AgentCore Code
   Interpreter runs it in an isolated microVM and returns a chart.
3. *A second opinion* — Claude Opus **and** Amazon Nova Pro read the evidence
   independently; Sonnet adjudicates where the two model families disagree.

Then the receipt: a total well under a dollar, billed only for what ran.
