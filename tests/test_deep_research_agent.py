from pathlib import Path

import pytest

from deep_research_agent import agent as agent_module
from deep_research_agent import cli, prompts
from deep_research_agent.serpapi_client import SerpApiClient, SerpApiResult
from deep_research_agent.tools import DeepResearchRuntime


def test_serpapi_client_formats_organic_results(monkeypatch) -> None:
    captured = {}

    def fake_get_json(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return {
            "organic_results": [
                {
                    "title": "Result A",
                    "link": "https://example.com/a",
                    "snippet": "Useful evidence.",
                    "source": "Example",
                },
                {"title": "Missing link", "snippet": "Skip me."},
            ]
        }

    client = SerpApiClient(api_key="test-key", get_json=fake_get_json)

    results = client.search("deep agents", max_results=5, gl="gb", hl="en")

    assert captured["url"] == "https://serpapi.com/search.json"
    assert captured["params"]["engine"] == "google"
    assert captured["params"]["q"] == "deep agents"
    assert captured["params"]["api_key"] == "test-key"
    assert captured["params"]["gl"] == "gb"
    assert results == [
        SerpApiResult(
            title="Result A",
            link="https://example.com/a",
            snippet="Useful evidence.",
            source="Example",
        )
    ]


def test_serpapi_client_surfaces_api_errors() -> None:
    client = SerpApiClient(api_key="test-key", get_json=lambda url, params, timeout: {"error": "bad key"})

    with pytest.raises(RuntimeError, match="bad key"):
        client.search("deep agents")


def test_deep_research_runtime_search_web_returns_citations(tmp_path: Path) -> None:
    client = SerpApiClient(
        api_key="test-key",
        get_json=lambda url, params, timeout: {
            "organic_results": [
                {
                    "title": "Deep Agents",
                    "link": "https://example.com/deep-agents",
                    "snippet": "Research agents use subagents.",
                }
            ]
        },
    )
    runtime = DeepResearchRuntime(tmp_path, client)

    result = runtime.search_web("deep agents", max_results=3)

    assert "[1] Deep Agents" in result
    assert "https://example.com/deep-agents" in result
    assert "Research agents use subagents." in result


def test_deep_research_runtime_search_web_returns_compact_errors(tmp_path: Path) -> None:
    client = SerpApiClient(
        api_key="test-key",
        get_json=lambda url, params, timeout: {"error": "Invalid API key"},
    )
    runtime = DeepResearchRuntime(tmp_path, client)

    result = runtime.search_web("deep agents")

    assert result == "SerpApi search failed for 'deep agents': Invalid API key"


def test_deep_research_runtime_writes_report(tmp_path: Path) -> None:
    runtime = DeepResearchRuntime(tmp_path, SerpApiClient(api_key="test-key"))

    path = runtime.write_report("AI Market", "## Findings\nEvidence [1].")

    assert path == "reports/ai-market.md"
    assert (tmp_path / path).read_text(encoding="utf-8").startswith("# AI Market")


def test_deep_research_tools_are_ptc_allowlisted(tmp_path: Path) -> None:
    runtime = DeepResearchRuntime(tmp_path, SerpApiClient(api_key="test-key"))

    tools = runtime.build_tools()

    assert {tool.name for tool in tools} == {"search_web", "write_report"}


def test_prompts_describe_async_subagents_ptc_recursion_and_skills() -> None:
    text = prompts.SUPERVISOR_PROMPT.lower()

    assert "start_async_task" in text
    assert "programmatic tool calling" in text
    assert "frontier" in text
    assert "skills" in text


def test_build_async_subagents_returns_research_roles() -> None:
    subagents = agent_module.build_async_subagents()

    assert {subagent["name"] for subagent in subagents} == {
        "researcher",
        "critic",
        "synthesizer",
    }
    assert all("graph_id" in subagent for subagent in subagents)


def test_register_deep_research_harness_profiles_excludes_todos(monkeypatch) -> None:
    registered = {}

    def fake_register_harness_profile(key, profile):
        registered[key] = profile

    monkeypatch.setattr(agent_module, "register_harness_profile", fake_register_harness_profile)

    note = agent_module.register_deep_research_harness_profiles()

    assert "qwen3.5:9b" in registered
    assert "TodoListMiddleware" in registered["ollama"].excluded_middleware
    assert "without TodoListMiddleware" in note


def test_build_sync_subagents_include_interpreter_guidance() -> None:
    subagents = agent_module.build_sync_subagents([])

    researcher = next(subagent for subagent in subagents if subagent["name"] == "researcher")
    assert "SerpApi" in researcher["system_prompt"]
    assert "Promise.all" in researcher["system_prompt"]
    assert "frontier" in researcher["system_prompt"]


def test_build_agent_passes_tools_middleware_subagents_and_skills(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "agent"

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr(agent_module, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(agent_module, "build_model", lambda config: "model")
    monkeypatch.setattr(agent_module, "build_interpreter_middleware", lambda tools: "interpreter")

    built, notes = agent_module.build_agent(
        agent_module.DeepResearchConfig(project_root=tmp_path, async_subagents=True)
    )

    assert built == "agent"
    assert captured["model"] == "model"
    assert captured["middleware"] == ["interpreter"]
    assert captured["subagents"][0]["name"] == "researcher"
    assert captured["skills"]
    assert "TodoListMiddleware" in notes[0]
    assert "SerpApi" in notes[1]


def test_build_agent_defaults_to_supervisor_only(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "agent"

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr(agent_module, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(agent_module, "build_model", lambda config: "model")
    monkeypatch.setattr(agent_module, "build_interpreter_middleware", lambda tools: "interpreter")

    agent_module.build_agent(agent_module.DeepResearchConfig(project_root=tmp_path))

    assert captured["subagents"] == []


def test_deep_research_model_uses_moonshot_kimi_choice(monkeypatch, tmp_path: Path) -> None:
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

    model = agent_module.build_model(agent_module.DeepResearchConfig(project_root=tmp_path))

    assert model == "model"
    assert captured["args"] == ("kimi-k2.6",)
    assert captured["kwargs"]["model_provider"] == "openai"
    assert captured["kwargs"]["api_key"] == "test-key"
    assert captured["kwargs"]["base_url"] == "https://api.moonshot.ai/v1"
    assert captured["kwargs"]["temperature"] == 0.6
    assert captured["kwargs"]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_deep_research_model_uses_ollama_qwen_choice(monkeypatch, tmp_path: Path) -> None:
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

    agent_module.build_model(agent_module.DeepResearchConfig(project_root=tmp_path))

    assert captured["args"] == ("qwen3.5:9b",)
    assert captured["kwargs"] == {"model_provider": "ollama", "temperature": 0}


def test_cli_parses_topic_and_async_subagents() -> None:
    args = cli.parse_args(["--topic", "AI infrastructure", "--async-subagents"])

    assert args.topic == "AI infrastructure"
    assert args.async_subagents is True
    assert args.sync_subagents is False
    assert args.stream is False


def test_cli_parses_sync_subagents() -> None:
    args = cli.parse_args(["--topic", "AI infrastructure", "--sync-subagents"])

    assert args.sync_subagents is True
    assert args.async_subagents is False


def test_cli_accepts_topic_split_by_smart_quotes() -> None:
    args = cli.parse_args(
        [
            "--topic",
            "“AI",
            "for",
            "small",
            "and",
            "medium",
            "business",
            "market",
            "in",
            "2026”",
        ]
    )

    assert args.topic == "AI for small and medium business market in 2026"


def test_cli_prints_only_final_response(monkeypatch, tmp_path: Path, capsys) -> None:
    class Args:
        topic = "AI infrastructure"
        model = None
        thread_id = "thread-1"
        async_subagents = False
        sync_subagents = False
        stream = False
        skip_serpapi_check = False

    monkeypatch.setattr(cli, "parse_args", lambda argv=None: Args())
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_agent", lambda config: ("agent", ["runtime note"]))
    monkeypatch.setattr(cli, "check_serpapi_access", lambda config: None)

    def noisy_run(agent, payload, config, stream):
        print("Updated todo list to [{'content': 'Plan research', 'status': 'in_progress'}]")
        return {"output": "final research report"}

    monkeypatch.setattr(cli, "run_with_streaming", noisy_run)

    assert cli.main([]) == 0

    assert capsys.readouterr().out == "final research report\n"


def test_cli_fails_fast_when_serpapi_check_fails(monkeypatch, tmp_path: Path, capsys) -> None:
    class Args:
        topic = "AI infrastructure"
        model = None
        thread_id = "thread-1"
        async_subagents = False
        sync_subagents = False
        stream = False
        skip_serpapi_check = False

    monkeypatch.setattr(cli, "parse_args", lambda argv=None: Args())
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "check_serpapi_access", lambda config: "HTTP 401 from SerpApi.")

    assert cli.main([]) == 1

    assert capsys.readouterr().out == "SerpApi check failed: HTTP 401 from SerpApi.\n"
