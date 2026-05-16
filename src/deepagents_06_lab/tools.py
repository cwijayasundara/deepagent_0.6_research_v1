from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import tool


_OPERATORS: dict[type[ast.operator | ast.unaryop], Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


@dataclass(frozen=True)
class ToolRuntime:
    project_root: Path

    @property
    def source_dir(self) -> Path:
        return self.project_root / "examples" / "source_notes"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"

    def search_notes(self, query: str, max_results: int = 5) -> str:
        """Search local example notes for matching snippets."""
        normalized = query.strip().lower()
        if not normalized:
            return "No query provided."

        matches: list[str] = []
        for note in sorted(self.source_dir.glob("*.md")):
            text = note.read_text(encoding="utf-8")
            for paragraph in re.split(r"\n\s*\n", text):
                if normalized in paragraph.lower():
                    snippet = " ".join(paragraph.split())
                    matches.append(f"{note.name}: {snippet}")
                    break
            if len(matches) >= max_results:
                break

        return "\n".join(matches) if matches else f"No matches for {query!r}."

    def read_note(self, relative_path: str) -> str:
        """Read a note from examples/source_notes."""
        path = self._safe_source_path(relative_path)
        return path.read_text(encoding="utf-8")

    def calculate(self, expression: str) -> str:
        """Evaluate a safe arithmetic expression."""
        try:
            tree = ast.parse(expression, mode="eval")
            value = self._eval_arithmetic(tree.body)
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError("Only arithmetic expressions are allowed.") from exc
        if float(value).is_integer():
            return str(int(value))
        return str(value)

    def write_report(self, title: str, markdown_body: str) -> str:
        """Write a Markdown report under reports/ and return its relative path."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "report"
        path = self.reports_dir / f"{slug}.md"
        path.write_text(f"# {title}\n\n{markdown_body.strip()}\n", encoding="utf-8")
        return path.relative_to(self.project_root).as_posix()

    def build_tools(self) -> list[Any]:
        runtime = self

        @tool
        def search_notes(query: str, max_results: int = 5) -> str:
            """Search the local source notes for short evidence snippets."""
            return runtime.search_notes(query, max_results=max_results)

        @tool
        def read_note(relative_path: str) -> str:
            """Read one Markdown note from examples/source_notes."""
            return runtime.read_note(relative_path)

        @tool
        def calculate(expression: str) -> str:
            """Evaluate safe arithmetic needed for report calculations."""
            return runtime.calculate(expression)

        @tool
        def write_report(title: str, markdown_body: str) -> str:
            """Write the final Markdown report and return its path."""
            return runtime.write_report(title, markdown_body)

        return [search_notes, read_note, calculate, write_report]

    def _safe_source_path(self, relative_path: str) -> Path:
        base = self.source_dir.resolve()
        path = (base / relative_path).resolve()
        if base != path and base not in path.parents:
            raise ValueError("Can only read files inside examples/source_notes.")
        if path.suffix != ".md" or not path.exists():
            raise ValueError(f"Note not found inside examples/source_notes: {relative_path}")
        return path

    def _eval_arithmetic(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _OPERATORS:
                raise ValueError("Only arithmetic operators are allowed.")
            return _OPERATORS[op_type](
                self._eval_arithmetic(node.left),
                self._eval_arithmetic(node.right),
            )
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _OPERATORS:
                raise ValueError("Only arithmetic operators are allowed.")
            return _OPERATORS[op_type](self._eval_arithmetic(node.operand))
        raise ValueError("Only arithmetic expressions are allowed.")


def build_tools(project_root: Path) -> list[Any]:
    return ToolRuntime(project_root).build_tools()
