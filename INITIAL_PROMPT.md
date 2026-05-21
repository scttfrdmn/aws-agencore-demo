# Initial prompt for Claude Code

Paste the block below as your first message to Claude Code in `~/src/aws-agentcore-demo`.
It scopes the first milestone narrowly: build the web app against the *seeded*
agent layer, with tests and lint green — before anything touches real AWS.

---

```
Read CLAUDE.md fully before doing anything.

This repo is a live-demo web app. The AWS-facing core is already seeded and
working: src/agentcore_demo/aws.py, agent.py, cost.py, questions.py. Do NOT rewrite them.
build_kb.py / corpus_fetch.py / teardown.py are also seeded.

Your milestone: build the local web UI and its tests. In order:

1. src/agentcore_demo/app.py — a FastAPI app:
   - GET /            serves src/agentcore_demo/static/index.html
   - GET /static/*    serves static assets
   - WS  /ws          on connect, runs the agent for all three questions;
                      forwards every agent event (see the event protocol in
                      CLAUDE.md) to the client as JSON.
   - On startup, open http://localhost:8000 in the browser automatically.
   - Wire the agent's emit(event) callback to push over the WebSocket.

2. src/agentcore_demo/static/index.html — one page, Alpine.js from a CDN, no build step:
   - A header: "Inside the Lines — PCSK9".
   - A live transcript: each question, then its phases/models lighting up as
     events arrive (retrieval count, each model start/done with its cost).
   - The generated code shown in a monospace block; the chart rendered inline
     from the base64 in the `chart` event.
   - A cost meter, always visible, updating on every `cost` event.
   - A final receipt table on the `receipt` event.
   - Legible on a projector: large text, clear state, calm colours.

3. Tests:
   - Keep tests AWS-free. Use the FakeBackend in tests/conftest.py.
   - tests/test_agent.py — drive agent.py with the fake backend; assert the
     event sequence is well-formed and the receipt total equals the sum of
     line items.
   - Add a test that the /ws endpoint streams a complete run ending in `done`
     (FastAPI TestClient / httpx).

4. Make `make lint` and `make test` both pass. Then stop and show me the app
   running against the fake backend before we point it at real AWS.

Do not touch the three questions in questions.py. Do not add a build step or
npm. Do not commit a config.py. Ask before changing the event protocol.
```

---

## After milestone 1

Once the app runs green against the fake backend, the remaining work is:

1. `cp config.example.py config.py` and fill it in.
2. **Verify** the S3 Vectors APIs in `build_kb.py` against current boto3 docs
   (the file marks the spots), then `make corpus`, sync to S3, `make build-kb`.
3. Point the app at real AWS; rehearse the three questions to determinism.
4. Record a clean run as the talk-day fallback.
5. `make teardown` when done.
