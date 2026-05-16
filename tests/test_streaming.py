from deepagents_06_lab.streaming import render_event, run_with_streaming


def test_render_event_message_text_delta() -> None:
    event = {"event": "messages", "data": {"delta": {"content": "hello"}}}

    assert render_event(event) == "hello"


def test_render_event_tool_call() -> None:
    event = {"event": "tool_call", "name": "search_notes", "data": {"args": {"query": "risk"}}}

    assert render_event(event) == "\n[tool] search_notes {'query': 'risk'}\n"


def test_render_event_subagent_status() -> None:
    event = {"event": "subagent", "name": "analyst", "data": {"status": "running"}}

    assert render_event(event) == "\n[subagent] analyst running\n"


def test_render_event_final_output() -> None:
    event = {"event": "final", "data": {"output": "done"}}

    assert render_event(event) == "\n[final]\ndone\n"


def test_render_event_ignores_unknown_event() -> None:
    assert render_event({"event": "unknown", "data": {}}) is None


def test_run_with_streaming_prefers_stream_events(capsys) -> None:
    class FakeAgent:
        def stream_events(self, payload, config=None, version=None):
            assert payload == {"messages": []}
            assert config == {"configurable": {"thread_id": "x"}}
            assert version == "v3"
            yield {"event": "messages", "data": {"delta": {"content": "hi"}}}
            yield {"event": "final", "data": {"output": "done"}}

    result = run_with_streaming(
        FakeAgent(),
        {"messages": []},
        {"configurable": {"thread_id": "x"}},
        stream=True,
    )

    assert result == {"output": "done"}
    assert "hi" in capsys.readouterr().out


def test_run_with_streaming_falls_back_to_invoke() -> None:
    class FakeAgent:
        def invoke(self, payload, config=None):
            assert payload == {"messages": []}
            assert config == {"configurable": {"thread_id": "x"}}
            return {"output": "done"}

    result = run_with_streaming(
        FakeAgent(),
        {"messages": []},
        {"configurable": {"thread_id": "x"}},
        stream=True,
    )

    assert result == {"output": "done"}
