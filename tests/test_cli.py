from pathlib import Path

import pytest

from deepagents_06_lab import cli


def test_parse_args_defaults_to_kimi_and_streaming() -> None:
    args = cli.parse_args(["--task", "Analyze this"])

    assert args.model == "kimi-k2.6"
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


def test_main_runs_agent(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class Args:
        task = "Do work"
        example = False
        model = "kimi-k2.6"
        memory = "local"
        thread_id = "thread-1"
        stream = False

    monkeypatch.setattr(cli, "parse_args", lambda argv=None: Args())
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
