from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from deep_research_agent.serpapi_client import SerpApiClient


@dataclass(frozen=True)
class DeepResearchRuntime:
    project_root: Path
    serpapi: SerpApiClient

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"

    def search_web(self, query: str, max_results: int = 5) -> str:
        """Search Google through SerpApi and return compact cited results."""
        max_results = min(max_results, 5)
        try:
            results = self.serpapi.search(query, max_results=max_results)
        except Exception as exc:
            return f"SerpApi search failed for {query!r}: {exc}"
        if not results:
            return f"No SerpApi organic results for {query!r}."
        lines: list[str] = []
        for index, result in enumerate(results, start=1):
            source = f" ({result.source})" if result.source else ""
            lines.append(
                f"[{index}] {result.title}{source}\n"
                f"URL: {result.link}\n"
                f"Snippet: {result.snippet or 'No snippet returned.'}"
            )
        return "\n\n".join(lines)

    def write_report(self, title: str, markdown_body: str) -> str:
        """Write a Markdown research report and return its relative path."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "research-report"
        path = self.reports_dir / f"{slug}.md"
        path.write_text(f"# {title}\n\n{markdown_body.strip()}\n", encoding="utf-8")
        return path.relative_to(self.project_root).as_posix()

    def build_tools(self) -> list[Any]:
        runtime = self

        @tool
        def search_web(query: str, max_results: int = 5) -> str:
            """Search Google through SerpApi and return compact cited results."""
            return runtime.search_web(query, max_results=max_results)

        @tool
        def write_report(title: str, markdown_body: str) -> str:
            """Write the final Markdown report and return its path."""
            return runtime.write_report(title, markdown_body)

        return [search_web, write_report]


def build_tools(project_root: Path, serpapi_client: SerpApiClient) -> list[Any]:
    return DeepResearchRuntime(project_root, serpapi_client).build_tools()
