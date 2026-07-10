from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin


def _beautifulsoup(html: str):
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("beautifulsoup4 is required for API detection") from exc
    return BeautifulSoup(html, "html.parser")


def detect_api_candidates(html: str, base_url: str) -> list[dict[str, Any]]:
    soup = _beautifulsoup(html)
    candidates: list[dict[str, Any]] = []
    for script in soup.find_all("script"):
        src = script.get("src")
        if src:
            absolute = urljoin(base_url, src)
            if any(token in absolute.lower() for token in ["/api/", "graphql", "jobs", "career", "recruit"]):
                candidates.append({"kind": "script_src", "url": absolute})
        text = script.get_text(" ", strip=False)
        if not text:
            continue
        lowered = text.lower()
        for token in ["/api/", "graphql", "jobs", "career", "recruit", "position", "vacanc"]:
            if token in lowered:
                candidates.append({"kind": "script_text", "token": token, "snippet": text[:500]})
                break
    return candidates

