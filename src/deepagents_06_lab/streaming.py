from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def render_event(event: Any) -> str | None:
    if not isinstance(event, Mapping):
        return None

    event_name = str(event.get("event", ""))
    name = event.get("name", "")
    data = event.get("data", {})
    if not isinstance(data, Mapping):
        data = {}

    if event_name in {"messages", "on_chat_model_stream"}:
        delta = data.get("delta") or data.get("chunk") or {}
        if isinstance(delta, Mapping):
            content = delta.get("content")
            if isinstance(content, str):
                return content
        if isinstance(delta, str):
            return delta

    if event_name in {"tool_call", "on_tool_start"}:
        tool_name = name or data.get("name", "tool")
        args = data.get("args") or data.get("input") or {}
        return f"\n[tool] {tool_name} {args}\n"

    if event_name in {"subagent", "subagent_status"}:
        subagent_name = name or data.get("name", "subagent")
        status = data.get("status", "update")
        return f"\n[subagent] {subagent_name} {status}\n"

    if event_name in {"final", "on_chain_end"}:
        output = data.get("output") or data.get("result")
        if output is not None:
            return f"\n[final]\n{output}\n"

    return None


def run_with_streaming(agent: Any, payload: dict[str, Any], config: dict[str, Any], stream: bool) -> Any:
    if not stream or not hasattr(agent, "stream_events"):
        return agent.invoke(payload, config=config)

    final: Any = None
    try:
        events = agent.stream_events(payload, config=config, version="v3")
        for event in events:
            rendered = render_event(event)
            if rendered:
                print(rendered, end="", flush=True)
            if isinstance(event, Mapping) and event.get("event") in {"final", "on_chain_end"}:
                data = event.get("data", {})
                if isinstance(data, Mapping):
                    final = data.get("output") or data.get("result")
    except (AttributeError, TypeError, NotImplementedError):
        return agent.invoke(payload, config=config)

    return {"output": final} if final is not None else None
