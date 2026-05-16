from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Sequence

from deepagents_06_lab.agent import AgentConfig, build_agent, build_run_config
from deepagents_06_lab.prompts import build_user_prompt
from deepagents_06_lab.streaming import extract_final_response, run_with_streaming


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="deepagents-06-lab",
        description="Run a Deep Agents 0.6 CLI demo with an .env-configured chat model.",
    )
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--task", help="Task for the agent to run.")
    task_group.add_argument("--example", action="store_true", help="Run examples/task.txt.")
    parser.add_argument("--model", help="Override LLM_MODEL from .env.")
    parser.add_argument("--thread-id", default="demo", help="Stable thread id for checkpointing.")
    parser.add_argument(
        "--memory",
        choices=["local", "context-hub"],
        default="local",
        help="Memory/backend mode.",
    )
    parser.add_argument("--stream", dest="stream", action="store_true", help="Use v3 event streaming.")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Use invoke fallback.")
    parser.set_defaults(stream=False)
    return parser.parse_args(argv)


def load_task(args: argparse.Namespace, project_root: Path = PROJECT_ROOT) -> str:
    if args.example:
        return (project_root / "examples" / "task.txt").read_text(encoding="utf-8").strip()
    if args.task:
        return args.task.strip()
    raise SystemExit("Provide --task or --example.")


def main(argv: Sequence[str] | None = None) -> int:
    _suppress_known_beta_warnings()
    args = parse_args(argv)
    task = load_task(args, PROJECT_ROOT)

    config = AgentConfig(
        model=args.model,
        memory=args.memory,
        thread_id=args.thread_id,
        project_root=PROJECT_ROOT,
    )
    agent, _notes = build_agent(config)

    payload = {"messages": [{"role": "user", "content": build_user_prompt(task)}]}
    run_config = build_run_config(args.thread_id)
    result = run_with_streaming(agent, payload, run_config, stream=args.stream)
    final_response = extract_final_response(result)
    if final_response:
        print(final_response)
    return 0


def _suppress_known_beta_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*CodeInterpreterMiddleware.*beta.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*v3 streaming protocol.*experimental.*",
    )


if __name__ == "__main__":
    raise SystemExit(main())
