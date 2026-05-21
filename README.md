# Inside the Lines

A five-minute live demo: a biomedical-research agent doing a day of work for the
price of a cup of coffee — inside a secure boundary, billed only for what runs.

Built for a 30-minute talk on AWS Bedrock to a research-computing audience. The
point: researchers can have frontier-model agents *without* the data leaving a
secure environment.

## What it does

A local web page shows, live, one agent answering three escalating questions
about the gene *PCSK9* against a Bedrock Knowledge Base of ~1,000 open-access
papers — with a running cost meter that ends in an itemised receipt.

```
  Q1  retrieval + synthesis  ->  Claude Haiku
  Q2  analysis + a chart     ->  Claude Sonnet  +  AgentCore Code Interpreter
  Q3  the hard call          ->  Claude Opus  AND  Amazon Nova Pro, cross-checked
```

All GA AWS services: Bedrock Knowledge Bases (S3 Vectors store), Bedrock models
(Claude + Nova), AgentCore Code Interpreter. No OpenSearch, no console.

## Quickstart

```bash
git clone https://github.com/scttfrdmn/aws-agentcore-demo.git ~/src/aws-agentcore-demo
cd ~/src/aws-agentcore-demo
make setup                          # install package + dev tools
cp config.example.py config.py      # then edit config.py

make corpus                         # build the PMC paper corpus (one-time)
aws s3 sync ./corpus s3://YOUR-BUCKET/corpus/
make build-kb                       # provision the Knowledge Base; paste IDs into config.py

make demo                           # runs the local web app, opens the browser
make teardown                       # AFTER the talk -- removes billable resources
```

`make lint` / `make test` run ruff and pytest; CI runs both on every push.

## Status

The AWS core (`aws.py`, `agent.py`, `cost.py`, `questions.py`) and the
provisioning scripts are seeded. The local web app (`app.py`, `static/`) and the
test suite are built by Claude Code — see `CLAUDE.md` and `INITIAL_PROMPT.md`.

## Honest caveats

- **The corpus is not in this repo** — only code. `corpus_fetch.py` builds it and
  keeps only CC0 / CC BY articles. Verify the current PMC OA licence split before
  relying on it for a public talk.
- **Model IDs and pricing in `config.py` are placeholders** — confirm the
  inference-profile IDs (`aws bedrock list-inference-profiles`) and the per-token
  rates against the live pricing page. The receipt is only as honest as those.
- **Verify the S3 Vectors APIs** in `build_kb.py` against current boto3 before
  running — they are the newest surface here; the file flags the spots.
- Cost figures in any talk slides are illustrative; the app prints the real total
  from actual token usage.

## License

MIT — see [LICENSE](LICENSE).
