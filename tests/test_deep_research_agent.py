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
    assert "SerpApi" in notes[0]


def test_cli_parses_topic_and_async_subagents() -> None:
    args = cli.parse_args(["--topic", "AI infrastructure", "--async-subagents"])

    assert args.topic == "AI infrastructure"
    assert args.async_subagents is True
    assert args.stream is False


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
        stream = False

    monkeypatch.setattr(cli, "parse_args", lambda argv=None: Args())
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_agent", lambda config: ("agent", ["runtime note"]))

    def noisy_run(agent, payload, config, stream):
        print("Updated todo list to [{'content': 'Plan research', 'status': 'in_progress'}]")
        return {"output": "final research report"}

    monkeypatch.setattr(cli, "run_with_streaming", noisy_run)

    assert cli.main([]) == 0

    assert capsys.readouterr().out == "final research report\n"
