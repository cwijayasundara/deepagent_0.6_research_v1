from pathlib import Path

import pytest

from deepagents_06_lab.tools import ToolRuntime


def make_runtime(tmp_path: Path) -> ToolRuntime:
    notes = tmp_path / "examples" / "source_notes"
    notes.mkdir(parents=True)
    (notes / "company.md").write_text(
        "# Northwind Robotics\n\nRevenue grew 42 percent after launching field service robots.\n",
        encoding="utf-8",
    )
    (notes / "product.md").write_text(
        "# Product Notes\n\nThe autonomy module reduced warehouse downtime by 18 percent.\n",
        encoding="utf-8",
    )
    return ToolRuntime(tmp_path)


def test_search_notes_returns_matching_snippets(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    result = runtime.search_notes("downtime")

    assert "product.md" in result
    assert "18 percent" in result
    assert "company.md" not in result


def test_read_note_rejects_paths_outside_source_notes(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    secret = tmp_path / "secret.md"
    secret.write_text("hidden", encoding="utf-8")

    with pytest.raises(ValueError, match="source_notes"):
        runtime.read_note("../secret.md")


def test_calculate_supports_safe_arithmetic(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    assert runtime.calculate("(42 + 18) / 3") == "20"


def test_calculate_rejects_function_calls(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    with pytest.raises(ValueError, match="Only arithmetic"):
        runtime.calculate("__import__('os').system('echo unsafe')")


def test_write_report_creates_markdown_under_reports(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    path = runtime.write_report("Northwind Review", "## Finding\nGrowth is strong.")

    report_path = tmp_path / path
    assert path.startswith("reports/")
    assert report_path.read_text(encoding="utf-8").startswith("# Northwind Review")


def test_build_tools_returns_langchain_tools(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    tools = runtime.build_tools()

    assert {tool.name for tool in tools} == {
        "search_notes",
        "read_note",
        "calculate",
        "write_report",
    }
