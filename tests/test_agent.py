from pathlib import Path

from deepagents_06_lab import agent as agent_module
from deepagents_06_lab.agent import AgentConfig


def test_agent_config_defaults_to_kimi() -> None:
    assert AgentConfig().model is None
    assert AgentConfig().model_provider is None
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


def test_build_model_uses_ollama_from_env(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    (tmp_path / ".env").write_text(
        "LLM_PROVIDER=ollama\nLLM_MODEL=nemotron3:33b\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    def fake_init_chat_model(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "model"

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)

    model = agent_module.build_model(AgentConfig(project_root=tmp_path))

    assert model == "model"
    assert captured["args"] == ("nemotron3:33b",)
    assert captured["kwargs"] == {"model_provider": "ollama", "temperature": 0}


def test_build_model_uses_single_llm_choice_switch_for_ollama(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    (tmp_path / ".env").write_text("LLM_CHOICE=ollama_nemotron\n", encoding="utf-8")
    monkeypatch.delenv("LLM_CHOICE", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    def fake_init_chat_model(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "model"

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)

    model = agent_module.build_model(AgentConfig(project_root=tmp_path))

    assert model == "model"
    assert captured["args"] == ("nemotron3:33b",)
    assert captured["kwargs"] == {"model_provider": "ollama", "temperature": 0}


def test_build_model_uses_single_llm_choice_switch_for_ollama_qwen(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    (tmp_path / ".env").write_text("LLM_CHOICE=ollama_qwen\n", encoding="utf-8")
    monkeypatch.delenv("LLM_CHOICE", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    def fake_init_chat_model(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "model"

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)

    model = agent_module.build_model(AgentConfig(project_root=tmp_path))

    assert model == "model"
    assert captured["args"] == ("qwen3.6:latest",)
    assert captured["kwargs"] == {"model_provider": "ollama", "temperature": 0}


def test_build_model_accepts_qwan_alias_for_qwen(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    (tmp_path / ".env").write_text("LLM_CHOICE=ollama_qwan\n", encoding="utf-8")
    monkeypatch.delenv("LLM_CHOICE", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    def fake_init_chat_model(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "model"

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)

    agent_module.build_model(AgentConfig(project_root=tmp_path))

    assert captured["args"] == ("qwen3.6:latest",)
    assert captured["kwargs"]["model_provider"] == "ollama"


def test_register_sample_harness_profiles(monkeypatch) -> None:
    registered = {}

    def fake_register_harness_profile(key, profile):
        registered[key] = profile

    monkeypatch.setattr(agent_module, "register_harness_profile", fake_register_harness_profile)

    notes = agent_module.register_sample_harness_profiles()

    assert "ollama" in registered
    assert "qwen3.6:latest" in registered
    assert "qwen3.5:9b" in registered
    assert "nemotron3:33b" in registered
    assert "openai:kimi-k2.6" in registered
    assert "Harness profiles" in notes
    assert "JavaScript" in registered["ollama"].system_prompt_suffix


def test_build_model_uses_single_llm_choice_switch_for_moonshot(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    (tmp_path / ".env").write_text(
        "LLM_CHOICE=moonshot_kimi\nMOONSHOT_API_KEY=test-key\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("LLM_CHOICE", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)

    def fake_init_chat_model(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "model"

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)

    model = agent_module.build_model(AgentConfig(project_root=tmp_path))

    assert model == "model"
    assert captured["args"] == ("kimi-k2.6",)
    assert captured["kwargs"]["model_provider"] == "openai"
    assert captured["kwargs"]["api_key"] == "test-key"


def test_llm_provider_and_model_override_llm_choice(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    (tmp_path / ".env").write_text(
        "LLM_CHOICE=moonshot_kimi\nLLM_PROVIDER=ollama\nLLM_MODEL=nemotron3:33b\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("LLM_CHOICE", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    def fake_init_chat_model(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "model"

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)

    agent_module.build_model(AgentConfig(project_root=tmp_path))

    assert captured["args"] == ("nemotron3:33b",)
    assert captured["kwargs"]["model_provider"] == "ollama"


def test_build_model_cli_override_keeps_env_provider(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    (tmp_path / ".env").write_text("LLM_PROVIDER=ollama\nLLM_MODEL=llama3.2\n", encoding="utf-8")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    def fake_init_chat_model(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "model"

    monkeypatch.setattr(agent_module, "init_chat_model", fake_init_chat_model)

    agent_module.build_model(AgentConfig(model="nemotron3:33b", project_root=tmp_path))

    assert captured["args"] == ("nemotron3:33b",)
    assert captured["kwargs"]["model_provider"] == "ollama"


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


def test_feature_matrix_covers_deepagents_06_concepts() -> None:
    matrix = agent_module.deepagents_06_feature_matrix()

    assert matrix["code_interpreter"]["implemented_by"] == "CodeInterpreterMiddleware"
    assert matrix["harness_profiles"]["implemented_by"] == "register_harness_profile"
    assert matrix["streaming"]["implemented_by"] == "stream_events(version='v3')"
    assert matrix["delta_channel"]["implemented_by"] == "langgraph.channels.delta.DeltaChannel"
    assert matrix["context_hub_backend"]["implemented_by"] == "deepagents.backends.ContextHubBackend"


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
    monkeypatch.setattr(agent_module, "register_sample_harness_profiles", lambda: "profiles registered")

    built, notes = agent_module.build_agent(AgentConfig(project_root=tmp_path))

    assert built == "agent"
    assert captured["model"].__class__ is FakeModel
    assert captured["tools"] == ["tool-a"]
    assert "configured chat model" in captured["system_prompt"]
    assert captured["middleware"][0].__class__ is FakeMiddleware
    assert captured["backend"].__class__.__name__ == "StateBackend"
    assert "checkpointer" in captured
    assert "profiles registered" in notes


def test_build_run_config_uses_thread_id() -> None:
    assert agent_module.build_run_config("demo") == {"configurable": {"thread_id": "demo"}}
