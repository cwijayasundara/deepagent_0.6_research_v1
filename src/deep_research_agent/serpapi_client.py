from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SerpApiResult:
    title: str
    link: str
    snippet: str
    source: str = ""


GetJson = Callable[[str, dict[str, Any], float], Mapping[str, Any]]


class SerpApiClient:
    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = "https://serpapi.com/search.json",
        timeout: float = 20.0,
        get_json: GetJson | None = None,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout
        self._get_json = get_json or _urllib_get_json

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        gl: str = "us",
        hl: str = "en",
    ) -> list[SerpApiResult]:
        params: dict[str, Any] = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "gl": gl,
            "hl": hl,
        }
        payload = self._get_json(self.endpoint, params, self.timeout)
        if error := payload.get("error"):
            raise RuntimeError(str(error))

        results: list[SerpApiResult] = []
        for item in payload.get("organic_results", []):
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("title") or "").strip()
            link = str(item.get("link") or "").strip()
            if not title or not link:
                continue
            results.append(
                SerpApiResult(
                    title=title,
                    link=link,
                    snippet=str(item.get("snippet") or "").strip(),
                    source=str(item.get("source") or "").strip(),
                )
            )
            if len(results) >= max_results:
                break
        return results


def _urllib_get_json(url: str, params: dict[str, Any], timeout: float) -> Mapping[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": "deep-research-agent/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

