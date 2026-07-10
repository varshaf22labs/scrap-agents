from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin


def _beautifulsoup(html: str):
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("beautifulsoup4 is required for HTML parsing") from exc
    return BeautifulSoup(html, "html.parser")


def extract_links(html: str, base_url: str) -> list[str]:
    soup = _beautifulsoup(html)
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if href:
            links.append(urljoin(base_url, href))
    return list(dict.fromkeys(links))


def extract_text_by_selector(html: str, selector: str) -> str | None:
    soup = _beautifulsoup(html)
    node = soup.select_one(selector)
    if node is None:
        return None
    return " ".join(node.get_text(" ", strip=True).split())


def extract_elements_text(html: str, selector: str) -> list[str]:
    soup = _beautifulsoup(html)
    nodes = soup.select(selector)
    return [" ".join(node.get_text(" ", strip=True).split()) for node in nodes]


def extract_attrs(html: str, selector: str, attr_name: str) -> list[str]:
    soup = _beautifulsoup(html)
    values: list[str] = []
    for node in soup.select(selector):
        value = node.get(attr_name)
        if value:
            values.append(str(value))
    return values

