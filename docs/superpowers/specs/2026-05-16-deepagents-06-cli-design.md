# Deep Agents 0.6 CLI Sample Design

## Goal

Build a Python CLI sample app that demonstrates Deep Agents 0.6 capabilities with a local Ollama model, `qwen3.6:latest`. The app should be runnable from a clean checkout, stream agent progress in the terminal, and show how code interpreter, model-agnostic programmatic tool calling, recursive workflows, checkpointing, and optional ContextHub memory fit together in one compact application.

## App Shape

The app is named `deepagents_06_lab`. It exposes a CLI command that accepts a research or analysis task, creates a Deep Agent configured for Ollama/Qwen, streams execution events, and writes a final report artifact. A seeded example task and local source documents let users run the sample without external services beyond Ollama.

The default workflow is a due-diligence style research task:

1. Read seed notes from local files.
2. Use interpreter-driven JavaScript to fan out tool and subagent calls.
3. Keep intermediate aggregation in interpreter state.
4. Recursively create follow-up questions until enough evidence is gathered.
5. Stream model text, tool calls, and subagent progress to the terminal.
6. Save the final report and pass a stable `thread_id` into the Deep Agents/LangGraph config so installs with compatible checkpointers can resume the run.

## Architecture

The implementation is intentionally small and split by responsibility:

- `src/deepagents_06_lab/cli.py`: argument parsing, environment checks, run orchestration.
- `src/deepagents_06_lab/agent.py`: Deep Agent construction, Ollama model binding, middleware, backend, and checkpoint configuration.
- `src/deepagents_06_lab/tools.py`: local demo tools for reading seed files, searching notes, simple calculations, and writing report artifacts.
- `src/deepagents_06_lab/prompts.py`: system instructions that teach the agent to use interpreter-side PTC and recursive follow-up queues.
- `src/deepagents_06_lab/streaming.py`: terminal rendering for `stream_events(..., version="v3")`, with graceful fallback if the installed API differs.
- `examples/`: seed task and local source notes.
- `reports/`: generated output directory.

## Deep Agents 0.6 Features

The app will use `langchain_quickjs.REPLMiddleware` to enable the code interpreter. The prompt will tell the model to write JavaScript that calls tools and subagents from the interpreter, filters noisy intermediate results, and returns compact findings to the model context. This demonstrates model-agnostic PTC because the behavior is implemented by the harness/runtime rather than relying on a provider-specific model feature.

Harness profile support will be represented by a Qwen-oriented profile module: concise tool instructions, explicit JSON/tool-call expectations, and conservative context-return rules. If Deep Agents exposes first-class profile APIs in the installed version, the app will use them; otherwise the profile will be applied through prompts and middleware configuration.

Streaming will prefer v3 event projections and render messages, tool calls, subagent updates, and final output. The renderer will degrade to normal synchronous invocation if the local package version does not expose the exact v3 helper surface.

Checkpointing and delta-channel support will be enabled through the installed Deep Agents/LangGraph APIs after runtime feature detection. The app will keep checkpoint state under `.agent_state/` by default. If the installed package version does not expose DeltaChannel configuration, startup will continue with standard checkpointing and the README will call out the limitation.

ContextHub memory will be optional. If `--memory context-hub` is passed and `LANGSMITH_API_KEY` is set, the app will use `ContextHubBackend`; otherwise it will use local state and print a concise explanation.

## Error Handling

Startup validates that Ollama is reachable and that `qwen3.6:latest` is available or clearly reports how to pull it. Missing optional packages produce actionable errors. Network-dependent tools are not required for the default demo.

During streaming, malformed or unknown event types are ignored after a debug-level representation. Tool failures are returned as structured observations so the agent can retry or continue.

## Testing

Tests focus on the local pieces that do not require a live model:

- CLI argument parsing and environment validation.
- Tool behavior for seed-file search and report writing.
- Prompt/profile content includes the required PTC and recursion guidance.
- Streaming renderer handles representative event shapes and unknown events.

Live Ollama execution will be documented as a manual smoke test because model output is non-deterministic and depends on local model availability.

## Acceptance Criteria

- `uv run deepagents-06-lab --example` runs the seeded CLI flow when Ollama has `qwen3.6:latest`.
- The agent is configured with QuickJS interpreter middleware.
- The prompt and tools support interpreter-side PTC and recursive follow-up workflows.
- Terminal output streams useful progress when v3 streaming is available.
- Reports are written to `reports/`.
- README explains setup, Ollama model pull, ContextHub option, and the fallback path when DeltaChannel or v3 streaming helpers are unavailable in the installed packages.
