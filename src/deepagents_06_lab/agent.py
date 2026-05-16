from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from langchain_ollama import ChatOllama

from deepagents_06_lab.prompts import SYSTEM_PROMPT
from deepagents_06_lab.tools import build_tools


@dataclass(frozen=True)
class AgentConfig:
    model: str = "qwen3.6:latest"
    memory: str = "local"
    thread_id: str = "demo"
    project_root: Path = field(default_factory=lambda: Path.cwd())


def build_model(config: AgentConfig) -> Any:
    return ChatOllama(model=config.model, temperature=0)


def build_backend(config: AgentConfig) -> tuple[Any, str]:
    from deepagents.backends import ContextHubBackend, StateBackend

    if config.memory == "context-hub":
        if not os.getenv("LANGSMITH_API_KEY"):
            return StateBackend(), (
                "ContextHub requested but LANGSMITH_API_KEY is not set; using StateBackend."
            )
        return ContextHubBackend("deepagents-06-lab"), "Using ContextHubBackend."

    return StateBackend(), "Using local StateBackend."


def build_checkpointer() -> tuple[Any, str]:
    from langgraph.checkpoint.memory import InMemorySaver

    note = "Using InMemorySaver checkpointing."
    try:
        from langgraph.checkpoint.memory import DeltaChannelHistory  # noqa: F401

        note += " DeltaChannelHistory is available in this LangGraph install."
    except Exception:
        note += " Delta channel helpers were not detected; using standard checkpoints."
    return InMemorySaver(), note


def build_interpreter_middleware(tools: list[Any]) -> Any:
    try:
        from langchain_quickjs import REPLMiddleware

        return REPLMiddleware()
    except (ImportError, AttributeError):
        from langchain_quickjs.middleware import CodeInterpreterMiddleware

        return CodeInterpreterMiddleware(ptc=tools)


def build_agent(config: AgentConfig) -> tuple[Any, list[str]]:
    model = build_model(config)
    tools = build_tools(config.project_root)
    middleware = [build_interpreter_middleware(tools)]
    backend, backend_note = build_backend(config)
    checkpointer, checkpoint_note = build_checkpointer()

    kwargs: dict[str, Any] = {
        "model": model,
        "tools": tools,
        "system_prompt": SYSTEM_PROMPT,
        "middleware": middleware,
        "backend": backend,
        "checkpointer": checkpointer,
    }
    signature = inspect.signature(create_deep_agent)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    supported = (
        kwargs
        if accepts_kwargs
        else {name: value for name, value in kwargs.items() if name in signature.parameters}
    )
    return create_deep_agent(**supported), [backend_note, checkpoint_note]


def build_run_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}
