from pathlib import Path

import pytest

from deepagents_06_lab import cli


def test_parse_args_defaults_to_qwen_and_streaming() -> None:
    args = cli.parse_args(["--task", "Analyze this"])

    assert args.model == "qwen3.6:latest"
    assert args.task == "Analyze this"
    assert args.stream is True
    assert args.memory == "local"


def test_parse_args_supports_example_thread_memory_and_no_stream() -> None:
    args = cli.parse_args(
        ["--example", "--thread-id", "abc", "--memory", "context-hub", "--no-stream"]
    )

    assert args.example is True
    assert args.thread_id == "abc"
    assert args.memory == "context-hub"
    assert args.stream is False


def test_load_task_reads_example(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "task.txt").write_text("Example task", encoding="utf-8")

    assert cli.load_task(cli.parse_args(["--example"]), tmp_path) == "Example task"


def test_load_task_requires_task_or_example(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.load_task(cli.parse_args([]), tmp_path)


def test_check_ollama_model_accepts_installed_model(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"models": [{"name": "qwen3.6:latest"}]}

    monkeypatch.setattr(cli.requests, "get", lambda url, timeout: FakeResponse())

    ok, message = cli.check_ollama_model("qwen3.6:latest")

    assert ok is True
    assert "available" in message


def test_check_ollama_model_reports_missing_model(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"models": [{"name": "llama3.2:latest"}]}

    monkeypatch.setattr(cli.requests, "get", lambda url, timeout: FakeResponse())

    ok, message = cli.check_ollama_model("qwen3.6:latest")

    assert ok is False
    assert "ollama pull qwen3.6:latest" in message


def test_check_ollama_model_reports_connection_failure(monkeypatch) -> None:
    def fail(url, timeout):
        raise cli.requests.RequestException("offline")

    monkeypatch.setattr(cli.requests, "get", fail)

    ok, message = cli.check_ollama_model("qwen3.6:latest")

    assert ok is False
    assert "Ollama is not reachable" in message


def test_main_runs_agent(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class Args:
        task = "Do work"
        example = False
        model = "qwen3.6:latest"
        memory = "local"
        thread_id = "thread-1"
        stream = False
        skip_ollama_check = False

    monkeypatch.setattr(cli, "parse_args", lambda argv=None: Args())
    monkeypatch.setattr(cli, "check_ollama_model", lambda model: (True, "available"))
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_agent", lambda config: ("agent", ["note"]))

    def fake_run(agent, payload, config, stream):
        captured["agent"] = agent
        captured["payload"] = payload
        captured["config"] = config
        captured["stream"] = stream
        return {"output": "done"}

    monkeypatch.setattr(cli, "run_with_streaming", fake_run)

    assert cli.main([]) == 0
    assert captured["agent"] == "agent"
    assert captured["payload"]["messages"][0]["content"].endswith("Task: Do work")
    assert captured["config"] == {"configurable": {"thread_id": "thread-1"}}
    assert captured["stream"] is False
