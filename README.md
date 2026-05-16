# Deep Agents 0.6 CLI Lab

A compact Python CLI sample for trying Deep Agents 0.6 with an `.env`-configured LLM.

The app demonstrates:

- QuickJS code interpreter middleware.
- Model-agnostic programmatic tool calling from the interpreter.
- Recursive follow-up workflows.
- Terminal streaming with v3-style event rendering where available.
- Local checkpoint/backend fallback with optional LangSmith ContextHub memory.

## Setup

```bash
uv sync
printf 'MOONSHOT_API_KEY=your-key-here\n' > .env
```

Use one switch to choose the LLM:

```env
LLM_CHOICE=moonshot_kimi
MOONSHOT_API_KEY=your-key-here
```

or:

```env
LLM_CHOICE=ollama_nemotron
```

Supported `LLM_CHOICE` values:

- `moonshot_kimi`: uses Moonshot `kimi-k2.6`.
- `ollama_nemotron`: uses local Ollama `nemotron3:33b`.

Advanced overrides are still available:

```env
LLM_PROVIDER=ollama
LLM_MODEL=nemotron3:33b
```

## Run

```bash
uv run deepagents-06-lab --example --thread-id demo-001
```

Generated reports are written to `reports/`.

The app reads `LLM_CHOICE`, optional `LLM_PROVIDER`/`LLM_MODEL` overrides, and
provider credentials from `.env` before creating the model with LangChain's
`init_chat_model`. Use `--no-stream` only when you want a single final result;
streaming is more useful for watching long runs progress.

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

- `src/deepagents_06_lab/agent.py` configures `create_deep_agent`, `.env`-driven `init_chat_model`, QuickJS interpreter middleware, backend selection, and checkpointing.
- `src/deepagents_06_lab/prompts.py` contains model-neutral JavaScript PTC instructions.
- `src/deepagents_06_lab/tools.py` exposes local tools that the interpreter can call programmatically.
- `src/deepagents_06_lab/streaming.py` renders v3-style event dictionaries and falls back to `invoke`.
