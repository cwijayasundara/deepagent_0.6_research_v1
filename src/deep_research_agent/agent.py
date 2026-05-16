from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepagents import AsyncSubAgent, SubAgent, create_deep_agent
from deepagents.profiles import register_harness_profile
from deepagents.profiles.harness.harness_profiles import HarnessProfile
from langchain.chat_models import init_chat_model

from deep_research_agent.prompts import CRITIC_PROMPT, RESEARCHER_PROMPT, SUPERVISOR_PROMPT, SYNTHESIZER_PROMPT
from deep_research_agent.serpapi_client import SerpApiClient
from deep_research_agent.tools import build_tools


@dataclass(frozen=True)
class DeepResearchConfig:
    model: str | None = None
    model_provider: str | None = None
    project_root: Path = field(default_factory=lambda: Path.cwd())
    async_subagents: bool = False
    sync_subagents: bool = False


def load_env_file(project_root: Path) -> None:
    env_file = project_root / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def build_model(config: DeepResearchConfig) -> Any:
    load_env_file(config.project_root)
    choice_provider, choice_model = _choice_defaults(os.getenv("LLM_CHOICE", "moonshot_kimi"))
    configured_provider = (config.model_provider or os.getenv("LLM_PROVIDER") or choice_provider).lower()
    provider = "openai" if configured_provider == "moonshot" else configured_provider
    model = config.model or os.getenv("LLM_MODEL") or choice_model
    if provider == "ollama":
        return init_chat_model(model, model_provider="ollama", temperature=0)
    if configured_provider == "moonshot":
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise RuntimeError("MOONSHOT_API_KEY is required for Moonshot.")
        return init_chat_model(
            model,
            model_provider="openai",
            api_key=api_key,
            base_url="https://api.moonshot.ai/v1",
            temperature=0.6,
            extra_body={"thinking": {"type": "disabled"}},
        )
    return init_chat_model(model, model_provider=provider, temperature=0)


def _choice_defaults(choice: str) -> tuple[str, str]:
    normalized = choice.strip().lower().replace("-", "_")
    choices = {
        "moonshot_kimi": ("moonshot", "kimi-k2.6"),
        "kimi": ("moonshot", "kimi-k2.6"),
        "ollama_qwen": ("ollama", "qwen3.5:9b"),
        "qwen": ("ollama", "qwen3.5:9b"),
        "ollama_qwan": ("ollama", "qwen3.5:9b"),
        "qwan": ("ollama", "qwen3.5:9b"),
    }
    return choices.get(normalized, (normalized or "moonshot", "kimi-k2.6"))


def build_serpapi_client(config: DeepResearchConfig) -> SerpApiClient:
    load_env_file(config.project_root)
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY is required for web research. Add it to .env.")
    return SerpApiClient(api_key=api_key)


def build_interpreter_middleware(tools: list[Any]) -> Any:
    from langchain_quickjs.middleware import CodeInterpreterMiddleware

    return CodeInterpreterMiddleware(ptc=tools)


def build_sync_subagents(tools: list[Any]) -> list[SubAgent]:
    return [
        {
            "name": "researcher",
            "description": "Focused SerpApi researcher. Use for source-backed investigation.",
            "system_prompt": RESEARCHER_PROMPT,
            "tools": tools,
        },
        {
            "name": "critic",
            "description": "Evidence critic. Use to find weak claims and follow-up questions.",
            "system_prompt": CRITIC_PROMPT,
            "tools": tools,
        },
        {
            "name": "synthesizer",
            "description": "Report synthesizer. Use to consolidate findings and citations.",
            "system_prompt": SYNTHESIZER_PROMPT,
            "tools": tools,
        },
    ]


def build_async_subagents() -> list[AsyncSubAgent]:
    return [
        {
            "name": "researcher",
            "description": "Conducts in-depth web research using SerpApi and returns cited findings.",
            "graph_id": "researcher",
        },
        {
            "name": "critic",
            "description": "Checks evidence quality, gaps, contradictions, and follow-up questions.",
            "graph_id": "critic",
        },
        {
            "name": "synthesizer",
            "description": "Synthesizes completed research tracks into report-ready sections.",
            "graph_id": "synthesizer",
        },
    ]


def build_skill_paths(project_root: Path) -> list[str]:
    skill_root = project_root / "src" / "deep_research_agent" / "skills"
    return [
        str(skill_root / "research-planning"),
        str(skill_root / "source-quality"),
        str(skill_root / "citation-dedup"),
        str(skill_root / "report-synthesis"),
    ]


def build_agent(config: DeepResearchConfig) -> tuple[Any, list[str]]:
    profile_note = register_deep_research_harness_profiles()
    model = build_model(config)
    tools = build_tools(config.project_root, build_serpapi_client(config))
    middleware = [build_interpreter_middleware(tools)]
    if config.async_subagents:
        subagents: list[Any] = build_async_subagents()
        subagent_note = "Using async subagents."
    elif config.sync_subagents:
        subagents = build_sync_subagents(tools)
        subagent_note = "Using synchronous local subagents."
    else:
        subagents = []
        subagent_note = "Using supervisor-only bounded research; pass --sync-subagents or --async-subagents to delegate."
    kwargs: dict[str, Any] = {
        "model": model,
        "tools": tools,
        "system_prompt": SUPERVISOR_PROMPT,
        "middleware": middleware,
        "subagents": subagents,
        "skills": build_skill_paths(config.project_root),
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
    return create_deep_agent(**supported), [
        profile_note,
        "Using SerpApi Google search.",
        "Using interpreter PTC for search fan-out and recursive frontier queues.",
        subagent_note,
    ]


def build_run_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def register_deep_research_harness_profiles() -> str:
    profile = HarnessProfile(
        system_prompt_suffix=(
            "For this deep research CLI, do research planning in prose or in the "
            "QuickJS interpreter. Do not rely on the write_todos tool as a terminal action."
        ),
        excluded_middleware=frozenset({"TodoListMiddleware"}),
    )
    for key in ("ollama", "qwen3.5:9b", "openai:kimi-k2.6"):
        register_harness_profile(key, profile)
    return "Registered deep research harness profiles without TodoListMiddleware."
