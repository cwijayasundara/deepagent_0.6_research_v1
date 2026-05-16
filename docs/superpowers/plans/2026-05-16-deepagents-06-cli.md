# Deep Agents 0.6 CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI sample app that demonstrates Deep Agents 0.6 with Ollama `qwen3.6:latest`, QuickJS interpreter/PTC, streaming, checkpointing, and optional ContextHub memory.

**Architecture:** The app is a small `src/` Python package with focused modules for CLI orchestration, agent construction, tools, prompts, and terminal streaming. Runtime integrations use defensive feature detection so the sample remains runnable when exact Deep Agents 0.6 helper APIs differ across installs.

**Tech Stack:** Python 3.11+, `uv`, `deepagents[quickjs]`, `langchain-ollama`, `langchain-core`, `langgraph`, `pytest`.

---

## File Structure

- `pyproject.toml`: package metadata, dependencies, CLI script, pytest config.
- `README.md`: setup, Ollama model pull, usage, feature map, fallback notes.
- `.gitignore`: generated state, reports, caches.
- `src/deepagents_06_lab/__init__.py`: package version.
- `src/deepagents_06_lab/cli.py`: CLI parser, environment validation, run orchestration.
- `src/deepagents_06_lab/agent.py`: Deep Agent construction, Ollama model binding, middleware/backend/checkpoint feature detection.
- `src/deepagents_06_lab/prompts.py`: Qwen harness profile and interpreter/PTC instructions.
- `src/deepagents_06_lab/tools.py`: local seed search, file read, calculations, report writing.
- `src/deepagents_06_lab/streaming.py`: v3 stream event rendering with invoke fallback.
- `examples/task.txt`: default demo task.
- `examples/source_notes/*.md`: local evidence corpus.
- `tests/test_cli.py`: CLI parser and Ollama validation tests.
- `tests/test_tools.py`: local tool tests.
- `tests/test_prompts.py`: harness prompt tests.
- `tests/test_streaming.py`: renderer tests.

## Task 1: Project Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/deepagents_06_lab/__init__.py`
- Create: `README.md`

- [ ] **Step 1: Write package files**

Create a package named `deepagents-06-lab` with script `deepagents-06-lab = "deepagents_06_lab.cli:main"`. Dependencies should include `deepagents[quickjs]>=0.6`, `langchain-ollama`, `langchain-core`, `langgraph`, `requests`, and test dependency `pytest`.

- [ ] **Step 2: Run metadata check**

Run: `uv sync`

Expected: dependencies resolve, or a clear package-resolution error that identifies any changed package names.

- [ ] **Step 3: Commit**

Run:

```bash
git add pyproject.toml .gitignore src/deepagents_06_lab/__init__.py README.md
git commit -m "chore: scaffold Deep Agents CLI package"
```

## Task 2: Local Tools

**Files:**
- Create: `src/deepagents_06_lab/tools.py`
- Create: `examples/source_notes/company.md`
- Create: `examples/source_notes/product.md`
- Create: `examples/task.txt`
- Create: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests**

Test that `search_notes()` finds matching snippets, `read_note()` blocks paths outside `examples/source_notes`, `calculate()` supports safe arithmetic only, and `write_report()` creates a Markdown report under `reports/`.

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_tools.py -q`

Expected: tests fail because `deepagents_06_lab.tools` does not exist.

- [ ] **Step 3: Implement tools**

Implement plain functions decorated with `@tool` when `langchain_core.tools.tool` is available. Expose `build_tools(project_root: Path) -> list[Any]`. Use path normalization to keep file access inside `examples/source_notes` and `reports`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_tools.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/deepagents_06_lab/tools.py examples tests/test_tools.py
git commit -m "feat: add local research tools"
```

## Task 3: Prompts and Harness Profile

**Files:**
- Create: `src/deepagents_06_lab/prompts.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write failing tests**

Test that the system prompt names Qwen/Ollama, requires interpreter-side JavaScript PTC, asks for recursive follow-up queues, limits returned context, and requires report writing.

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_prompts.py -q`

Expected: tests fail because `prompts.py` does not exist.

- [ ] **Step 3: Implement prompts**

Define `QWEN_HARNESS_PROFILE`, `SYSTEM_PROMPT`, and `build_user_prompt(task: str) -> str`. Include a compact JavaScript example that uses `Promise.all`, `tools.search_notes`, `tools.task`, a `frontier` queue, and final `tools.write_report`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_prompts.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/deepagents_06_lab/prompts.py tests/test_prompts.py
git commit -m "feat: add Qwen harness prompt"
```

## Task 4: Agent Factory

**Files:**
- Create: `src/deepagents_06_lab/agent.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write failing tests**

Test that `AgentConfig` defaults to model `qwen3.6:latest`, `build_model()` creates a `ChatOllama` with that model, `build_backend()` falls back to local state when ContextHub settings are missing, and `build_agent()` passes tools, prompt, model, and `REPLMiddleware` into `create_deep_agent`.

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_agent.py -q`

Expected: tests fail because `agent.py` does not exist.

- [ ] **Step 3: Implement agent factory**

Implement `AgentConfig`, `build_model`, `build_backend`, `build_checkpointer`, and `build_agent`. Import optional APIs inside functions so tests can monkeypatch missing packages. Use `create_deep_agent(model=model, tools=tools, instructions=SYSTEM_PROMPT, middleware=[REPLMiddleware()], backend=backend)` where supported, retrying without unsupported keyword arguments only after inspecting the callable signature.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_agent.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/deepagents_06_lab/agent.py tests/test_agent.py
git commit -m "feat: configure Deep Agent factory"
```

## Task 5: Streaming Renderer

**Files:**
- Create: `src/deepagents_06_lab/streaming.py`
- Create: `tests/test_streaming.py`

- [ ] **Step 1: Write failing tests**

Test representative v3-like event rendering for message text, tool calls, subagent status, final output, and unknown events.

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_streaming.py -q`

Expected: tests fail because `streaming.py` does not exist.

- [ ] **Step 3: Implement renderer**

Implement `render_event(event: Any) -> str | None` and `run_with_streaming(agent, payload, config, stream: bool) -> Any`. Prefer `agent.stream_events(payload, config=config, version="v3")`; if unavailable, call `agent.invoke(payload, config=config)`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_streaming.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/deepagents_06_lab/streaming.py tests/test_streaming.py
git commit -m "feat: add terminal stream renderer"
```

## Task 6: CLI Orchestration

**Files:**
- Create: `src/deepagents_06_lab/cli.py`
- Create: `tests/test_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing tests**

Test parser defaults, `--example`, `--task`, `--thread-id`, `--memory`, `--no-stream`, and Ollama validation behavior using monkeypatched HTTP responses.

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_cli.py -q`

Expected: tests fail because `cli.py` does not exist.

- [ ] **Step 3: Implement CLI**

Implement `parse_args`, `load_task`, `check_ollama_model`, `build_run_config`, and `main`. Use `http://localhost:11434/api/tags` to verify Ollama, report `ollama pull qwen3.6:latest` if missing, and pass `{"configurable": {"thread_id": args.thread_id}}` to the agent run.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_cli.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/deepagents_06_lab/cli.py tests/test_cli.py README.md
git commit -m "feat: add Deep Agents CLI entrypoint"
```

## Task 7: Full Verification

**Files:**
- Modify: `README.md` if verification exposes missing setup notes.

- [ ] **Step 1: Run unit tests**

Run: `uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Run import smoke test**

Run: `uv run python -c "from deepagents_06_lab.cli import main; from deepagents_06_lab.agent import AgentConfig; print(AgentConfig().model)"`

Expected: prints `qwen3.6:latest`.

- [ ] **Step 3: Run CLI help**

Run: `uv run deepagents-06-lab --help`

Expected: shows options for `--example`, `--task`, `--thread-id`, `--memory`, and `--no-stream`.

- [ ] **Step 4: Optional live smoke**

Run only if Ollama is running and the model exists:

```bash
uv run deepagents-06-lab --example --thread-id demo-001
```

Expected: terminal progress appears and a Markdown report is written to `reports/`.

- [ ] **Step 5: Commit final docs**

Run:

```bash
git add README.md
git commit -m "docs: document Deep Agents CLI demo"
```
