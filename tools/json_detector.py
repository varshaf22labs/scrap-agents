from __future__ import annotations

import json
from typing import Any


def _beautifulsoup(html: str):
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("beautifulsoup4 is required for JSON detection") from exc
    return BeautifulSoup(html, "html.parser")


def detect_embedded_json(html: str) -> list[dict[str, Any]]:
    soup = _beautifulsoup(html)
    payloads: list[dict[str, Any]] = []
    for script in soup.find_all("script"):
        script_type = (script.get("type") or "").lower()
        text = script.string or script.get_text(strip=False) or ""
        if not text.strip():
            continue
        if script_type == "application/ld+json":
            try:
                payloads.append({"kind": "ld+json", "data": json.loads(text)})
            except json.JSONDecodeError:
                continue
        elif "__NEXT_DATA__" in text or "window.__INITIAL_STATE__" in text or "self.__next_f.push" in text:
            payloads.append({"kind": "embedded_state", "data": text.strip()})
    return payloads

