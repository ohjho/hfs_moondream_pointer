# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A HuggingFace Space that exposes Moondream2's open-vocabulary detection (`point` and `object_detection`) via a Gradio interface. The entire app is `app.py` — a single `detect()` function wrapped by `gr.Interface`. README frontmatter (`title`, `sdk`, `sdk_version`, `app_file`) is the HF Space config and must stay valid YAML; the Space rebuilds from it.

## Running

```bash
pip install -r requirements.txt
python app.py
```

Gradio launches with `mcp_server=True` (exposes an MCP endpoint) and `app_kwargs={"docs_url": "/docs"}` (attempt to expose FastAPI Swagger). Per the README, the `/docs` route is known not to work in practice — see https://github.com/gradio-app/gradio/issues/4054.

## Runtime model

- Loads `vikhyatk/moondream2` at revision `2025-04-14` with `trust_remote_code=True`. Pinning the revision is intentional — the model repo changes its API surface across revisions, so bumping it can break `model.point(...)` / `model.detect(...)` return shapes.
- `@spaces.GPU(duration=30)` decorators are HF Spaces ZeroGPU directives — they only activate on the Space; locally they are no-ops requiring a CUDA device (the `device_map={"": "cuda"}` hard-codes GPU). Local CPU runs need that removed.
- `detect()` returns the inner list (`["points"]` or `["objects"]`) rather than the full dict — callers (including the MCP tool surface) depend on that shape.
- Concurrency (Gradio queue): `detect` runs at most 2 calls at once (`concurrency_limit=2` on the Interface; ZeroGPU forks a worker per call), the pilbox `annotate` `.change()` event at most 4, and `demo.queue(max_size=12)` caps waiters (ZeroGPU otherwise forces `max_size=1`, so a second waiter gets a 503 "Queue is full"). Limits are per event; work inside a single request is still sequential.
