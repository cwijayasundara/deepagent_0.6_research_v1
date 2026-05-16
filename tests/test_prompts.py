from deepagents_06_lab.prompts import QWEN_HARNESS_PROFILE, SYSTEM_PROMPT, build_user_prompt


def test_system_prompt_names_kimi_and_moonshot() -> None:
    text = SYSTEM_PROMPT.lower()

    assert "kimi" in text
    assert "moonshot" in text


def test_system_prompt_requires_interpreter_side_ptc() -> None:
    text = SYSTEM_PROMPT.lower()

    assert "javascript" in text
    assert "programmatic tool calling" in text
    assert "promise.all" in text
    assert "tools.search_notes" in text


def test_system_prompt_describes_recursive_follow_up_queue() -> None:
    text = SYSTEM_PROMPT.lower()

    assert "frontier" in text
    assert "follow-up" in text
    assert "recursive" in text


def test_system_prompt_limits_returned_context_and_requires_report() -> None:
    text = SYSTEM_PROMPT.lower()

    assert "return only" in text
    assert "compact" in text
    assert "tools.write_report" in text


def test_kimi_profile_sets_conservative_harness_rules() -> None:
    profile = QWEN_HARNESS_PROFILE

    assert profile["model_family"] == "kimi"
    assert "concise" in profile["tool_style"]
    assert "json" in profile["tool_style"].lower()


def test_build_user_prompt_wraps_task() -> None:
    prompt = build_user_prompt("Review Northwind Robotics")

    assert "Review Northwind Robotics" in prompt
    assert "Use the Deep Agents 0.6 workflow" in prompt
