import os
from pathlib import Path

from deepagents_06_lab import agent as agent_module
from deepagents_06_lab.agent import AgentConfig


def test_agent_config_defaults_to_qwen() -> None:
    assert AgentConfig().model == "qwen3.6:latest"
    assert AgentConfig().memory == "local"


def test_build_model_creates_chat_ollama(monkeypatch) -> None:
    captured = {}

    class FakeChatOllama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(agent_module, "ChatOllama", FakeChatOllama)

    model = agent_module.build_model(AgentConfig(model="custom-qwen"))

    assert isinstance(model, FakeChatOllama)
    assert captured["model"] == "custom-qwen"
    assert captured["temperature"] == 0


def test_build_backend_falls_back_to_state_without_langsmith_key(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    backend, note = agent_module.build_backend(AgentConfig(memory="context-hub"))

    assert backend.__class__.__name__ == "StateBackend"
    assert "LANGSMITH_API_KEY" in note


def test_build_backend_uses_context_hub_when_requested(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")

    backend, note = agent_module.build_backend(AgentConfig(memory="context-hub"))

    assert backend.__class__.__name__ == "ContextHubBackend"
    assert "ContextHubBackend" in note


def test_build_interpreter_middleware_uses_available_quickjs() -> None:
    middleware = agent_module.build_interpreter_middleware([])

    assert middleware.__class__.__name__ in {"CodeInterpreterMiddleware", "REPLMiddleware"}


def test_build_agent_passes_model_tools_prompt_middleware_and_backend(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeModel:
        pass

    class FakeMiddleware:
        pass

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "agent"

    monkeypatch.setattr(agent_module, "build_model", lambda config: FakeModel())
    monkeypatch.setattr(agent_module, "build_tools", lambda project_root: ["tool-a"])
    monkeypatch.setattr(agent_module, "build_interpreter_middleware", lambda tools: FakeMiddleware())
    monkeypatch.setattr(agent_module, "create_deep_agent", fake_create_deep_agent)

    built, notes = agent_module.build_agent(AgentConfig(project_root=tmp_path))

    assert built == "agent"
    assert captured["model"].__class__ is FakeModel
    assert captured["tools"] == ["tool-a"]
    assert "Qwen" in captured["system_prompt"]
    assert captured["middleware"][0].__class__ is FakeMiddleware
    assert captured["backend"].__class__.__name__ == "StateBackend"
    assert "checkpointer" in captured
    assert notes


def test_build_run_config_uses_thread_id() -> None:
    assert agent_module.build_run_config("demo") == {"configurable": {"thread_id": "demo"}}
