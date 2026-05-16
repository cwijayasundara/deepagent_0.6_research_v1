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

Local 36B Qwen runs can take several minutes, especially on the first prompt. Use
`--no-stream` only when you want a single final result; streaming is more useful for
watching long runs progress.

Use a custom task:

```bash
uv run deepagents-06-lab --task "Analyze the product risk in the local notes"
```

Use synchronous invoke fallback instead of streaming:

```bash
uv run deepagents-06-lab --example --no-stream
```

## Optional ContextHub Memory

Set `LANGSMITH_API_KEY` and pass:

```bash
uv run deepagents-06-lab --example --memory context-hub
```

If ContextHub or DeltaChannel APIs are unavailable in the installed packages, the app falls back to local state/checkpoint behavior and prints the limitation.

## Feature Map

- `src/deepagents_06_lab/agent.py` configures `create_deep_agent`, Ollama Qwen, QuickJS interpreter middleware, backend selection, and checkpointing.
- `src/deepagents_06_lab/prompts.py` contains the Qwen harness profile and JavaScript PTC instructions.
- `src/deepagents_06_lab/tools.py` exposes local tools that the interpreter can call programmatically.
- `src/deepagents_06_lab/streaming.py` renders v3-style event dictionaries and falls back to `invoke`.
