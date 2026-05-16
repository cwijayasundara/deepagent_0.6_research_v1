from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from deepagents_06_lab.agent import AgentConfig, build_agent, build_run_config
from deepagents_06_lab.prompts import build_user_prompt
from deepagents_06_lab.streaming import run_with_streaming


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="deepagents-06-lab",
        description="Run a Deep Agents 0.6 CLI demo with Moonshot Kimi.",
    )
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--task", help="Task for the agent to run.")
    task_group.add_argument("--example", action="store_true", help="Run examples/task.txt.")
    parser.add_argument("--model", default="kimi-k2.6", help="Moonshot/OpenAI-compatible model name.")
    parser.add_argument("--thread-id", default="demo", help="Stable thread id for checkpointing.")
    parser.add_argument(
        "--memory",
        choices=["local", "context-hub"],
        default="local",
        help="Memory/backend mode.",
    )
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Use invoke fallback.")
    parser.set_defaults(stream=True)
    return parser.parse_args(argv)


def load_task(args: argparse.Namespace, project_root: Path = PROJECT_ROOT) -> str:
    if args.example:
        return (project_root / "examples" / "task.txt").read_text(encoding="utf-8").strip()
    if args.task:
        return args.task.strip()
    raise SystemExit("Provide --task or --example.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    task = load_task(args, PROJECT_ROOT)

    config = AgentConfig(
        model=args.model,
        memory=args.memory,
        thread_id=args.thread_id,
        project_root=PROJECT_ROOT,
    )
    agent, notes = build_agent(config)
    for note in notes:
        print(f"[runtime] {note}")

    payload = {"messages": [{"role": "user", "content": build_user_prompt(task)}]}
    run_config = build_run_config(args.thread_id)
    result = run_with_streaming(agent, payload, run_config, stream=args.stream)
    if result is not None and not args.stream:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
