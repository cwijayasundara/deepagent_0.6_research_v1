from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.profiles import register_harness_profile
from deepagents.profiles.harness.harness_profiles import HarnessProfile
from langchain.chat_models import init_chat_model

from deepagents_06_lab.prompts import SYSTEM_PROMPT
from deepagents_06_lab.tools import build_tools


@dataclass(frozen=True)
class AgentConfig:
    model: str | None = None
    model_provider: str | None = None
    memory: str = "local"
    thread_id: str = "demo"
    project_root: Path = field(default_factory=lambda: Path.cwd())


def load_env_file(project_root: Path) -> None:
    env_file = project_root / ".env"
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_model(config: AgentConfig) -> Any:
    load_env_file(config.project_root)
    choice_provider, choice_model = _choice_defaults(os.getenv("LLM_CHOICE", "moonshot_kimi"))
    configured_provider = (config.model_provider or os.getenv("LLM_PROVIDER") or choice_provider).lower()
    model_provider = "openai" if configured_provider == "moonshot" else configured_provider
    model = config.model or os.getenv("LLM_MODEL") or choice_model or _default_model_for_provider(configured_provider)

    if configured_provider == "ollama":
        return init_chat_model(
            model,
            model_provider=model_provider,
            temperature=0,
        )

    moonshot_api_key = os.getenv("MOONSHOT_API_KEY")
    if not moonshot_api_key:
        raise RuntimeError("MOONSHOT_API_KEY is required for Moonshot. Add it to .env or export it.")
    return init_chat_model(
        model,
        model_provider=model_provider,
        api_key=moonshot_api_key,
        base_url="https://api.moonshot.ai/v1",
        temperature=0.6,
        extra_body={"thinking": {"type": "disabled"}},
    )


def _default_model_for_provider(provider: str) -> str:
    if provider == "ollama":
        return "nemotron3:33b"
    return "kimi-k2.6"


def _choice_defaults(choice: str) -> tuple[str, str | None]:
    normalized = choice.strip().lower().replace("-", "_")
    choices: dict[str, tuple[str, str]] = {
        "moonshot_kimi": ("moonshot", "kimi-k2.6"),
        "kimi": ("moonshot", "kimi-k2.6"),
        "ollama_nemotron": ("ollama", "nemotron3:33b"),
        "nemotron": ("ollama", "nemotron3:33b"),
        "ollama_qwen": ("ollama", "qwen3.6:latest"),
        "qwen": ("ollama", "qwen3.6:latest"),
        "ollama_qwan": ("ollama", "qwen3.6:latest"),
        "qwan": ("ollama", "qwen3.6:latest"),
    }
    return choices.get(normalized, (normalized or "moonshot", None))


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
        from langgraph.channels.delta import DeltaChannel  # noqa: F401
        from langgraph.checkpoint.memory import DeltaChannelHistory  # noqa: F401

        note += " DeltaChannel and DeltaChannelHistory are available in this LangGraph install."
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
    profile_note = register_sample_harness_profiles()
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
    return create_deep_agent(**supported), [profile_note, backend_note, checkpoint_note]


def build_run_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def register_sample_harness_profiles() -> str:
    suffix = (
        "\nHarness profile guidance for Deep Agents 0.6 sample:\n"
        "- Prefer concise JSON-shaped tool arguments.\n"
        "- Use QuickJS JavaScript for PTC fan-out, filtering, and recursive queues.\n"
        "- Return compact evidence to model context; keep intermediate state in the interpreter.\n"
    )
    profiles = {
        "ollama": HarnessProfile(system_prompt_suffix=suffix),
        "qwen3.5:9b": HarnessProfile(system_prompt_suffix=suffix),
        "openai:kimi-k2.6": HarnessProfile(system_prompt_suffix=suffix),
    }
    for key, profile in profiles.items():
        register_harness_profile(key, profile)
    return "Harness profiles registered for Ollama Qwen and Moonshot Kimi."


def deepagents_06_feature_matrix() -> dict[str, dict[str, str]]:
    return {
        "code_interpreter": {
            "implemented_by": "CodeInterpreterMiddleware",
            "where": "build_interpreter_middleware(ptc=tools)",
        },
        "harness_profiles": {
            "implemented_by": "register_harness_profile",
            "where": "register_sample_harness_profiles()",
        },
        "streaming": {
            "implemented_by": "stream_events(version='v3')",
            "where": "run_with_streaming()",
        },
        "delta_channel": {
            "implemented_by": "langgraph.channels.delta.DeltaChannel",
            "where": "build_checkpointer() feature detection",
        },
        "context_hub_backend": {
            "implemented_by": "deepagents.backends.ContextHubBackend",
            "where": "build_backend(memory='context-hub')",
        },
    }
