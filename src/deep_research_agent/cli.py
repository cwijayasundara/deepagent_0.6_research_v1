from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Sequence

from deep_research_agent.agent import DeepResearchConfig, build_agent, build_run_config
from deep_research_agent.prompts import build_user_prompt
from deepagents_06_lab.streaming import extract_final_response, run_with_streaming


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="deep-research-agent",
        description="Run a SerpApi-backed Deep Agents 0.6 deep research agent.",
    )
    parser.add_argument("--topic", required=True, help="Research topic or question.")
    parser.add_argument("--model", help="Override LLM_MODEL from .env.")
    parser.add_argument("--thread-id", default="deep-research", help="Stable thread id.")
    parser.add_argument("--async-subagents", action="store_true", help="Use AsyncSubAgent specs.")
    parser.add_argument("--stream", dest="stream", action="store_true", help="Use v3 event streaming.")
    parser.set_defaults(stream=False)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _suppress_known_beta_warnings()
    args = parse_args(argv)
    config = DeepResearchConfig(
        model=args.model,
        project_root=PROJECT_ROOT,
        async_subagents=args.async_subagents,
    )
    agent, _notes = build_agent(config)
    payload = {"messages": [{"role": "user", "content": build_user_prompt(args.topic)}]}
    result = run_with_streaming(agent, payload, build_run_config(args.thread_id), stream=args.stream)
    final_response = extract_final_response(result)
    if final_response:
        print(final_response)
    return 0


def _suppress_known_beta_warnings() -> None:
    warnings.filterwarnings("ignore", message=r".*CodeInterpreterMiddleware.*beta.*")
    warnings.filterwarnings("ignore", message=r".*v3 streaming protocol.*experimental.*")


if __name__ == "__main__":
    raise SystemExit(main())

