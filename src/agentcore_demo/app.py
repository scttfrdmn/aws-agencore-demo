"""
app.py  --  FastAPI web app for the Inside the Lines demo.

Endpoints:
  GET /               serves static/index.html
  GET /static/*       serves static assets
  GET /api/kb-status  {"ready": bool}
  GET /api/questions  returns the three locked question texts
  WS  /ingest         runs ingestion if KB not ready; emits progress + kb_ready
  WS  /ws             runs Agent question(s); forwards events as JSON

On startup the browser page checks kb-status, connects to /ingest if needed
(shows a progress view until done), then opens the question prompt.  If the KB
is already ready it goes straight to the prompt.

Backend selection (single seam, singleton per process):
  DEMO_FAKE=1         FakeBackend from agentcore_demo.fakes (no AWS, no config.py)
  DEMO_FAKE_READY=0   start with KB not ready (only with DEMO_FAKE=1)
  (unset)             AwsBackend built from config.py, rates from Price List API

Run with:  python -m agentcore_demo.app
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import threading
import webbrowser
from pathlib import Path

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agentcore_demo.agent import Agent
from agentcore_demo.backend import Backend
from agentcore_demo.cost import CostMeter

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── open browser once on startup (ASGI lifespan fires once per process) ──────
_launch_config: dict = {}  # populated by __main__ before uvicorn starts


@app.on_event("startup")
async def _on_startup() -> None:
    url = _launch_config.get("browser_url")
    if url:
        await asyncio.sleep(0.5)
        await asyncio.to_thread(webbrowser.open, url)


# ── singleton backend (one instance per process lifetime) ─────────────────────
_backend: Backend | None = None
_backend_lock = threading.Lock()


def _get_backend() -> Backend:
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                _backend = _build_backend()
    return _backend


def _build_backend() -> Backend:
    """Construct the backend. Called once at first use."""
    if os.environ.get("DEMO_FAKE") == "1":
        from agentcore_demo.fakes import FakeBackend  # noqa: PLC0415

        return FakeBackend()

    if importlib.util.find_spec("config") is None:
        raise RuntimeError(
            "config.py not found — copy config.example.py to config.py and fill it in, "
            "or set DEMO_FAKE=1 to run against the fake backend."
        )
    import config  # type: ignore[import]  # noqa: PLC0415
    from agentcore_demo.aws import AwsBackend  # noqa: PLC0415
    from agentcore_demo.pricing import fetch_rates  # noqa: PLC0415

    rates = fetch_rates(config.REGION)
    return AwsBackend(
        config.REGION,
        config.KB_ID,
        config.DATA_SOURCE_ID,
        config.MODELS,
        rates=rates,
        vector_bucket_name=getattr(config, "VECTOR_BUCKET_NAME", ""),
        guardrail_id=getattr(config, "GUARDRAIL_ID", ""),
        guardrail_version=getattr(config, "GUARDRAIL_VERSION", "DRAFT"),
        gateway_url=getattr(config, "GATEWAY_URL", ""),
    )


def _build_meter() -> CostMeter:
    """Build a fresh CostMeter for one query run."""
    if os.environ.get("DEMO_FAKE") == "1":
        from agentcore_demo.fakes import TEST_KB_RATES, TEST_PRICING  # noqa: PLC0415

        return CostMeter(pricing=TEST_PRICING, ci_per_second=0.0, **TEST_KB_RATES)

    import config  # type: ignore[import]  # noqa: PLC0415
    from agentcore_demo.pricing import fetch_rates  # noqa: PLC0415

    rates = fetch_rates(config.REGION)
    return CostMeter(
        pricing=config.PRICING,
        ci_per_second=config.CODE_INTERPRETER_PER_SECOND,
        kb_query_usd_per_1k=rates.get("s3v_query_usd_per_1k", 0.0),
    )


# ── HTTP routes ────────────────────────────────────────────────────────────────


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/kb-status")
async def kb_status() -> JSONResponse:
    """Return whether the KB has indexed documents and is ready to query."""
    backend = _get_backend()
    ready = await asyncio.to_thread(backend.kb_is_ready)
    return JSONResponse({"ready": ready})


@app.get("/api/kb-costs")
async def kb_costs() -> JSONResponse:
    """Return measured KB setup costs for the sidebar panel."""
    backend = _get_backend()
    costs = await asyncio.to_thread(backend.kb_setup_costs)
    return JSONResponse(costs)


@app.get("/corpus/{pmcid}")
async def serve_corpus_doc(pmcid: str):
    """Serve a corpus paper as a readable HTML page."""
    from fastapi import HTTPException  # noqa: PLC0415
    from fastapi.responses import HTMLResponse  # noqa: PLC0415

    path = Path("corpus") / f"{pmcid}.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{pmcid} not in local corpus")

    text = path.read_text(encoding="utf-8")
    # Strip the header comment line
    lines = text.splitlines()
    header = lines[0] if lines and lines[0].startswith("#") else ""
    body = "\n".join(lines[2:] if len(lines) > 2 else lines).strip()

    # Wrap in minimal readable HTML — no external deps
    ncbi_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{pmcid} — Inside the Lines corpus</title>
  <style>
    body {{ max-width: 780px; margin: 2rem auto; padding: 0 1.5rem 4rem;
            font-family: "Atkinson Hyperlegible", Georgia, serif; line-height: 1.75;
            color: #1a1a1a; background: #fafaf8; font-size: 1.05rem; }}
    header {{ border-bottom: 1px solid #ddd; margin-bottom: 1.5rem; padding-bottom: 0.75rem; }}
    h1 {{ font-size: 1.2rem; font-weight: 700; margin-bottom: 0.3rem; }}
    .meta {{ font-size: 0.85rem; color: #666; }}
    .meta a {{ color: #0066cc; }}
    p {{ margin-bottom: 1em; }}
  </style>
</head>
<body>
  <header>
    <h1>{pmcid}</h1>
    <p class="meta">
      {header.lstrip("# ")} &nbsp;·&nbsp;
      <a href="{ncbi_url}" target="_blank" rel="noopener">View on PubMed Central ↗</a>
    </p>
  </header>
  <article>
    {"".join(f"<p>{para.strip()}</p>" for para in body.split("  ") if para.strip())}
  </article>
</body>
</html>"""
    return HTMLResponse(html)


@app.get("/api/questions")
async def get_questions() -> JSONResponse:
    """Return the three locked question texts (used by the teletype UI)."""
    from agentcore_demo import questions as Q  # noqa: PLC0415

    return JSONResponse({"questions": Q.QUESTIONS})


# ── WebSocket: ingestion ───────────────────────────────────────────────────────


@app.websocket("/ingest")
async def websocket_ingest(ws: WebSocket) -> None:
    """Run ingestion, streaming progress; send kb_ready with costs when done."""
    await ws.accept()
    backend = _get_backend()
    loop = asyncio.get_running_loop()

    def progress_cb(indexed: int, total: int) -> None:
        fut = asyncio.run_coroutine_threadsafe(
            ws.send_text(json.dumps({"type": "ingesting", "indexed": indexed, "total": total})),
            loop,
        )
        with contextlib.suppress(Exception):
            fut.result(timeout=5)

    try:
        costs = await asyncio.to_thread(backend.kb_ingest, progress_cb)
        with contextlib.suppress(Exception):
            await ws.send_text(json.dumps({"type": "kb_ready", **costs}))
    except WebSocketDisconnect:
        pass
    finally:
        with contextlib.suppress(Exception):
            await ws.close()


# ── WebSocket: query run ───────────────────────────────────────────────────────


@app.websocket("/ws")
async def websocket_run(
    ws: WebSocket,
    q: int | None = Query(default=None, ge=1, le=4),
    text: str | None = Query(default=None),
) -> None:
    """Run the agent and forward every event as JSON.

    q=1|2|3   run one of the three canned questions (by number).
    text=...  run a free-form question (routed via Haiku).
    Neither   run all three canned questions in sequence.
    """
    await ws.accept()

    backend = _get_backend()
    meter = _build_meter()
    loop = asyncio.get_running_loop()
    stop_event = threading.Event()

    def emit(event: dict) -> None:
        if stop_event.is_set():
            return
        fut = asyncio.run_coroutine_threadsafe(ws.send_text(json.dumps(event)), loop)
        with contextlib.suppress(Exception):
            fut.result(timeout=5)

    try:
        if text:
            await asyncio.to_thread(_run_freeform, backend, meter, emit, stop_event, text)
        else:
            which = (q,) if q is not None else (1, 2, 3)
            await asyncio.to_thread(_run_agent, backend, meter, emit, stop_event, which)
    except WebSocketDisconnect:
        stop_event.set()
    finally:
        stop_event.set()
        with contextlib.suppress(Exception):
            await ws.close()


def _run_agent(
    backend: Backend,
    meter: CostMeter,
    emit,
    stop_event: threading.Event,
    which: tuple[int, ...] = (1, 2, 3),
) -> None:
    def guarded_emit(event: dict) -> None:
        if stop_event.is_set():
            return
        emit(event)

    Agent(backend, meter, guarded_emit).run(which=which)


def _run_freeform(
    backend: Backend,
    meter: CostMeter,
    emit,
    stop_event: threading.Event,
    text: str,
) -> None:
    def guarded_emit(event: dict) -> None:
        if stop_event.is_set():
            return
        emit(event)

    Agent(backend, meter, guarded_emit).run_freeform(text)


if __name__ == "__main__":
    import uvicorn

    host = "127.0.0.1"
    port = 8000
    if importlib.util.find_spec("config") is not None:
        import config  # type: ignore[import]

        host = getattr(config, "HOST", host)
        port = getattr(config, "PORT", port)

    _launch_config["browser_url"] = f"http://{host}:{port}"
    uvicorn.run("agentcore_demo.app:app", host=host, port=port)
