VENV   := .venv
PYTHON := $(VENV)/bin/python
RUFF   := $(VENV)/bin/ruff
PYTEST := $(VENV)/bin/pytest

.PHONY: setup lint fix test corpus build-kb demo demo-fake demo-fake-ingest demo-headless teardown


setup:    ## create venv and install the package + dev tools
	uv venv
	uv pip install -e ".[dev]"

lint:     ## ruff check + format check
	$(RUFF) check .
	$(RUFF) format --check .

fix:      ## auto-fix lint + format
	$(RUFF) check --fix .
	$(RUFF) format .

test:     ## run the test suite
	$(PYTEST)

corpus:   ## fetch the PMC paper corpus (one-time)
	$(PYTHON) corpus_fetch.py

build-kb: ## provision the Bedrock Knowledge Base (one-time)
	$(PYTHON) build_kb.py

demo:      ## run the live web app (requires config.py)
	$(PYTHON) -m agentcore_demo.app

demo-fake: ## run the web app with FakeBackend (no AWS, no config.py)
	DEMO_FAKE=1 $(PYTHON) -m agentcore_demo.app

demo-fake-ingest: ## run demo with fake backend starting in not-ready (ingestion) state
	DEMO_FAKE=1 DEMO_FAKE_READY=0 DEMO_FAKE_INGEST_DELAY=0.3 $(PYTHON) -m agentcore_demo.app

demo-headless: ## run all three questions headless against real AWS (requires config.py)
	$(PYTHON) -m agentcore_demo.run

teardown: ## delete all billable AWS resources
	$(PYTHON) teardown.py
