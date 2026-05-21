# Inside the Lines — PCSK9

A five-minute live demo for a research-computing audience showing frontier AI
agents running entirely within a secure AWS boundary — billed only for what runs.

The agent answers four escalating questions about the gene *PCSK9* against a
Bedrock Knowledge Base of ~650 open-access PMC papers, with a live cost meter
ending in an itemised receipt. Two Bedrock security services are demonstrated
in action:

| Beat | Question | Models | Security demo |
|------|----------|--------|---------------|
| Q1 | Role of PCSK9 in LDL regulation | Claude Haiku | **Bedrock Guardrail** intercepts external NCBI URLs → redirects to local corpus |
| Q2 | Compare trial LDL-lowering + chart | Claude Sonnet + Code Interpreter | — |
| Q3 | Where does literature disagree? | Claude Opus AND Nova Pro (parallel) + Sonnet adjudicates | **Bedrock Guardrail** (same) |
| Q4 | Search ClinicalTrials.gov for ongoing trials | Claude Haiku | **AgentCore Gateway Cedar policy** denies `web_fetch` tool call |

---

## Prerequisites

- AWS account with Bedrock model access enabled in **us-west-2**
  (`aws bedrock list-inference-profiles --region us-west-2` — all four models
  must be ACTIVE: Haiku 4.5, Sonnet 4.6, Opus 4.7, Nova Pro)
- An S3 bucket you own (e.g. `my-inside-the-lines-corpus`)
- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv)
- AWS CLI configured with a profile that has Bedrock, S3, IAM, Lambda,
  and bedrock-agentcore-control permissions

---

## One-time setup (before the talk)

### 1  Clone and install

```bash
git clone https://github.com/scttfrdmn/aws-agencore-demo ~/src/aws-agencore-demo
cd ~/src/aws-agencore-demo
uv venv && uv pip install -e ".[dev]"
```

### 2  Configure

```bash
cp config.example.py config.py
```

Edit `config.py` — fill in:

```python
REGION     = "us-west-2"
ACCOUNT_ID = "123456789012"          # your 12-digit account ID
BUCKET     = "my-inside-the-lines-corpus"
```

Verify model inference-profile IDs are correct for your account:

```bash
aws bedrock list-inference-profiles --region us-west-2 \
  --query 'inferenceProfileSummaries[?status==`ACTIVE`].inferenceProfileId'
```

If any ID in `config.py` MODELS differs from the output, update it.

### 3  Fetch the corpus

```bash
make corpus
```

Downloads ~650 CC0 / CC BY PCSK9 papers from PubMed Central. Takes ~5 minutes.
Re-run anytime to refresh; it skips papers already downloaded.

### 4  Sync corpus to S3

```bash
AWS_PROFILE=your-profile aws s3 sync corpus/ s3://YOUR-BUCKET/corpus/ --region us-west-2
```

### 5  Provision all AWS resources

```bash
AWS_PROFILE=your-profile python build_kb.py
```

This single script creates (in order):

1. **IAM role** for the Knowledge Base
2. **S3 Vectors** bucket + index (with `nonFilterableMetadataKeys` — required
   to stay under the 2 048-byte metadata limit)
3. **Bedrock Knowledge Base** + S3 data source
4. **Ingestion job** — embeds all papers with Titan Embed v2 (~5 min for 650 papers)
5. **Bedrock Guardrail** — regex policy that intercepts any `https://` URL in
   model output and returns it as `{EXTERNAL_URL}` for local redirection
6. **AgentCore Gateway** — MCP protocol, Cedar policy engine in ENFORCE mode
   - `PermitAll` policy (baseline allow)
   - `ForbidWebFetch` policy — `forbid(... action == "web-tools___web_fetch" ...)`
   - Lambda function that declares the `web_fetch` tool schema
   - IAM roles for Gateway and Lambda

When it finishes, paste the printed IDs into `config.py`:

```python
KB_ID              = "XXXXXXXXXX"
DATA_SOURCE_ID     = "XXXXXXXXXX"
GUARDRAIL_ID       = "xxxxxxxxxxxx"
GATEWAY_ID         = "inside-the-lines-gateway-xxxxxxxxxx"
GATEWAY_URL        = "https://inside-the-lines-gateway-xxxxxxxxxx.gateway.bedrock-agentcore.us-west-2.amazonaws.com"
GATEWAY_ENGINE_ID  = "InsideTheLinesEngine-xxxxxxxxxx"
```

### 6  Verify with a headless run

```bash
AWS_PROFILE=your-profile make demo-headless
```

Runs all four questions, prints the full receipt. Expected total: **$0.20–0.40**.
Q4 should show `🚫 CEDAR POLICY DENIED: tool=web_fetch`.

---

## Running the demo

```bash
AWS_PROFILE=your-profile make demo
```

Opens `http://localhost:8000` automatically. The page shows:

- **Chips + input** at top — click Q1, Q2, Q3, Q4 in order (each unlocks after
  the previous is answered); or type any free-form PCSK9 question
- **Canned chips** teletype the full question into the input, then fire
- **Live cost meter** in the sidebar, ticking up as models run
- **KB panel** — ingestion cost (computed from corpus size), S3 Vectors storage,
  S3 corpus storage — all measured, not estimated
- **Guardrail badge** on Q1/Q3 — "Bedrock Guardrail: N links intercepted" with
  per-link rows showing local redirects (🔵) and blocked articles (🔴)
- **Cedar policy badge** on Q4 — "Bedrock Cedar Policy — tool call denied"
- **Total receipt** after Q4 with all rows and grand total

### Fake backend (no AWS needed)

```bash
make demo-fake
```

Runs entirely locally with canned responses. Good for rehearsing the UI and
timing without spending money.

---

## Teardown (after the talk)

```bash
AWS_PROFILE=your-profile python teardown.py
```

Deletes: Knowledge Base, S3 Vectors index + bucket, Bedrock Guardrail,
AgentCore Gateway, Cedar PolicyEngine, Lambda function, all IAM roles,
and the S3 corpus bucket. Nothing left running.

> **Note:** the corpus download (`corpus/`) is local only — it is gitignored
> and not deleted by teardown. Delete it manually if needed.

---

## Development

```bash
make lint    # ruff check + format check
make test    # pytest (51 tests, no AWS calls)
make fix     # auto-fix lint and format
```

CI runs lint + test on every push (`.github/workflows/ci.yml`).

### Project structure

```
config.example.py     fill in → config.py (gitignored)
corpus_fetch.py       download CC0/CC BY PMC papers
build_kb.py           provision all AWS resources (one-time)
teardown.py           delete all billable resources

src/agentcore_demo/
  agent.py            orchestration — runs Q1–Q4, emits events
  aws.py              AwsBackend: retrieve, converse, code_interpreter_run,
                      query_gateway, kb_setup_costs, kb_is_ready, kb_ingest
  app.py              FastAPI: /ws, /ingest, /api/kb-status, /corpus/{pmcid}
  questions.py        locked question texts + system prompts
  cost.py             CostMeter — pure logic, thread-safe
  pricing.py          AWS Price List lookup (embed rate, S3 rate)
  fakes.py            FakeBackend for tests and make demo-fake
  run.py              headless terminal runner (make demo-headless)
  static/index.html   Alpine.js chat UI — no build step

tests/                51 tests, all AWS-free via FakeBackend
```

### Verified AWS quirks

These were discovered during development and are worth knowing:

- **Opus 4.7** — `temperature` and `topP` are deprecated; omit them from
  `inferenceConfig` or the call returns 400.
- **S3 Vectors metadata limit** — Bedrock KB chunks produce filterable metadata
  > 2 048 bytes, causing ingestion to fail. Fix: create the index with
  `metadataConfiguration.nonFilterableMetadataKeys` listing all Bedrock KB
  metadata keys (`AMAZON_BEDROCK_TEXT`, `x-amz-bedrock-kb-*`, etc.).
- **Ingestion job statistics** — only document counts, no token count.
  Ingestion cost is computed from local corpus `total_chars / 4 × embed_rate`.
- **S3 Vectors storage size** — `GetVectorBucket` has no `sizeBytes` field.
  Storage cost is derived from `ListVectors` count × per-vector bytes.
- **Cedar action format** — AgentCore Gateway tool actions use the format
  `"{target-name}___{tool-name}"` (e.g. `"web-tools___web_fetch"`), not
  `"InvokeTool"`.
- **Cedar policy denial** — returns HTTP 200 with a JSON-RPC error body
  (`"Tool Execution Denied: ..."`), not HTTP 403.
- **Claude 4.x Bedrock pricing** — in the `AmazonBedrockFoundationModels`
  Price List service code (not `AmazonBedrock`); Global (cross-region inference
  profile) tier applies when using `us.anthropic.*` model IDs.

---

## License

MIT — see [LICENSE](LICENSE).
