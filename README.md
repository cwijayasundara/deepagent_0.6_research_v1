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
LLM_CHOICE=ollama_qwen
```

Supported `LLM_CHOICE` values:

- `moonshot_kimi`: uses Moonshot `kimi-k2.6`.
- `ollama_qwen`: uses local Ollama `qwen3.5:9b`.

Advanced overrides are still available:

```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen3.5:9b
```

## Run

```bash
uv run deepagents-06-lab --example --thread-id demo-001
```

Generated reports are written to `reports/`.

The app reads `LLM_CHOICE`, optional `LLM_PROVIDER`/`LLM_MODEL` overrides, and
provider credentials from `.env` before creating the model with LangChain's
`init_chat_model`. The CLI defaults to synchronous invoke for reliable final
responses with local Ollama models. Use `--stream` when you specifically want
to exercise v3 event streaming.

Use a custom task:

```bash
uv run deepagents-06-lab --task "Analyze the product risk in the local notes"
```

Use v3 event streaming:

```bash
uv run deepagents-06-lab --example --stream
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

## Deep Agents 0.6 Coverage

This sample intentionally covers the major Deep Agents 0.6 concepts:

| Concept | Where it is demonstrated |
| --- | --- |
| Code interpreter | `build_interpreter_middleware()` uses QuickJS `CodeInterpreterMiddleware` with PTC-enabled tools. |
| Model-agnostic PTC | `prompts.py` instructs the model to use JavaScript `Promise.all`, recursive queues, and compact context returns. |
| Recursive workflows | `prompts.py` includes a frontier queue pattern that calls subagents and feeds follow-up work back into the interpreter. |
| Harness profiles | `register_sample_harness_profiles()` registers canonical Deep Agents profile keys for `ollama`, local Qwen, and Moonshot Kimi. |
| Streaming v3 | `run_with_streaming()` calls `stream_events(..., version="v3")` and extracts the final response. |
| DeltaChannel | `build_checkpointer()` detects `langgraph.channels.delta.DeltaChannel` and `DeltaChannelHistory` while using checkpointing. |
| ContextHubBackend | `build_backend(memory="context-hub")` selects `ContextHubBackend` when `LANGSMITH_API_KEY` is set. |

Use ContextHub-backed memory:

```bash
LANGSMITH_API_KEY=... uv run deepagents-06-lab --example --memory context-hub
```

## Deep Research Agent

The repo also includes a SerpApi-backed deep research agent under
`src/deep_research_agent`. It uses the same Deep Agents 0.6 building blocks in
a more realistic research workflow:

- The supervisor creates a research plan and todo list.
- Async subagent specs model long-running `researcher`, `critic`, and
  `synthesizer` tracks.
- The interpreter enables programmatic tool calling for SerpApi fan-out,
  filtering, deduplication, and recursive frontier queues.
- Skills encode reusable research planning, source quality, citation dedup, and
  report synthesis rules.

Configure SerpApi:

```env
SERPAPI_API_KEY=your-serpapi-key
LLM_PROVIDER=ollama
LLM_MODEL=qwen3.5:9b
```

Run a local synchronous-subagent research task:

```bash
uv run deep-research-agent --topic "AI infrastructure market in 2026"
```

Exercise async subagent specs:

```bash
uv run deep-research-agent --topic "AI infrastructure market in 2026" --async-subagents
```

Use `--stream` only when you specifically want to inspect v3 streaming behavior.
