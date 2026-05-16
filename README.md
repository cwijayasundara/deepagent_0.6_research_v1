# Deep Agents 0.6 CLI Lab

A compact Python CLI sample for trying Deep Agents 0.6 with local Ollama model `qwen3.6:latest`.

The app demonstrates:

- QuickJS code interpreter middleware.
- Model-agnostic programmatic tool calling from the interpreter.
- Recursive follow-up workflows.
- Terminal streaming with v3-style event rendering where available.
- Local checkpoint/backend fallback with optional LangSmith ContextHub memory.

## Setup

```bash
uv sync
ollama pull qwen3.6:latest
```

## Run

```bash
uv run deepagents-06-lab --example --thread-id demo-001
```

Generated reports are written to `reports/`.

## Optional ContextHub Memory

Set `LANGSMITH_API_KEY` and pass:

```bash
uv run deepagents-06-lab --example --memory context-hub
```

If ContextHub or DeltaChannel APIs are unavailable in the installed packages, the app falls back to local state/checkpoint behavior and prints the limitation.
