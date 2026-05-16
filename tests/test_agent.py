from pathlib import Path

from deepagents_06_lab import agent as agent_module
from deepagents_06_lab.agent import AgentConfig


def test_agent_config_defaults_to_kimi() -> None:
    assert AgentConfig().model == "kimi-k2.6"
    assert AgentConfig().model_provider == "openai"
    assert AgentConfig().memory == "local"


def test_load_env_file_sets_moonshot_api_key(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MOONSHOT_API_KEY=test-moonshot-key\n", encoding="utf-8")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)

    agent_module.load_env_file(tmp_path)

    assert agent_module.os.environ["MOONSHOT_API_KEY"] == "test-moonshot-key"


def test_load_env_file_does_not_override_existing_env(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MOONSHOT_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.setenv("MOONSHOT_API_KEY", "existing-key")

    agent_module.load_env_file(tmp_path)

    assert agent_module.os.environ["MOONSHOT_API_KEY"] == "existing-key"


def test_build_model_uses_init_chat_model_with_moonshot_key(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    (tmp_path / ".env").write_text("MOONSHOT_API_KEY=test-key\n", encoding="utf-8")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)

    def fake_init_chat_model(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "model"

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)

    model = agent_module.build_model(AgentConfig(model="custom-kimi", project_root=tmp_path))

    assert model == "model"
    assert captured["args"] == ("custom-kimi",)
    assert captured["kwargs"]["model_provider"] == "openai"
    assert captured["kwargs"]["api_key"] == "test-key"
    assert captured["kwargs"]["base_url"] == "https://api.moonshot.ai/v1"
    assert captured["kwargs"]["temperature"] == 0.6
    assert captured["kwargs"]["extra_body"] == {"thinking": {"type": "disabled"}}


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
    assert "Kimi" in captured["system_prompt"]
    assert captured["middleware"][0].__class__ is FakeMiddleware
    assert captured["backend"].__class__.__name__ == "StateBackend"
    assert "checkpointer" in captured
    assert notes


def test_build_run_config_uses_thread_id() -> None:
    assert agent_module.build_run_config("demo") == {"configurable": {"thread_id": "demo"}}
