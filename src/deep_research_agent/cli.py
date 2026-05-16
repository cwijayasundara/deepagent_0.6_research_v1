from __future__ import annotations

import argparse
import contextlib
import io
import warnings
from pathlib import Path
from typing import Sequence

from deep_research_agent.agent import DeepResearchConfig, build_agent, build_run_config, build_serpapi_client
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
    parser.add_argument("--sync-subagents", action="store_true", help="Use local synchronous subagents.")
    parser.add_argument("--stream", dest="stream", action="store_true", help="Use v3 event streaming.")
    parser.add_argument(
        "--skip-serpapi-check",
        action="store_true",
        help="Skip the startup SerpApi credentials check.",
    )
    parser.set_defaults(stream=False)
    args, extras = parser.parse_known_args(argv)
    if extras:
        args.topic = " ".join([args.topic, *extras])
    args.topic = _normalize_topic(args.topic)
    return args


def _normalize_topic(topic: str) -> str:
    return topic.strip().strip("\"'“”‘’").strip()


def main(argv: Sequence[str] | None = None) -> int:
    _suppress_known_beta_warnings()
    args = parse_args(argv)
    config = DeepResearchConfig(
        model=args.model,
        project_root=PROJECT_ROOT,
        async_subagents=args.async_subagents,
        sync_subagents=args.sync_subagents,
    )
    if not args.skip_serpapi_check:
        if error := check_serpapi_access(config):
            print(f"SerpApi check failed: {error}")
            return 1
    agent, _notes = build_agent(config)
    payload = {"messages": [{"role": "user", "content": build_user_prompt(args.topic)}]}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = run_with_streaming(agent, payload, build_run_config(args.thread_id), stream=args.stream)
    final_response = extract_final_response(result)
    if final_response:
        print(final_response)
    return 0


def check_serpapi_access(config: DeepResearchConfig) -> str | None:
    try:
        build_serpapi_client(config).search("test", max_results=1)
    except Exception as exc:
        return str(exc)
    return None


def _suppress_known_beta_warnings() -> None:
    warnings.filterwarnings("ignore", message=r".*CodeInterpreterMiddleware.*beta.*")
    warnings.filterwarnings("ignore", message=r".*v3 streaming protocol.*experimental.*")


if __name__ == "__main__":
    raise SystemExit(main())
