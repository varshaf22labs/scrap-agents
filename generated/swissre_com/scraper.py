#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

ENTRY_URL = 'https://swissre.com'
DEFAULT_OUTPUT = 'C:\\Users\\varsh\\scrap agent\\generated\\swissre_com.jsonl'
PAGINATION = {'type': 'page', 'param': 'page', 'start': 1, 'step': 1, 'max_pages': 50}
LISTING = {'items_selector': 'a', 'title_selector': 'a', 'url_selector': 'a', 'location_selector': None, 'description_selector': None, 'json_items_path': ['jobs']}
DETAIL = {'url_path': ['url'], 'description_selector': None, 'location_selector': None}
FILTERS = {'country': 'India', 'country_code': 'IN'}

def read_json_path(value: Any, path: list[Any]) -> Any:
    current = value
    for part in path:
        if current is None:
            return None
        if isinstance(part, int):
            if not isinstance(current, list) or part >= len(current):
                return None
            current = current[part]
        else:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
    return current

def string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None

def parse_location(text: Any, city: Any = None, state: Any = None, country: Any = None, country_code: Any = None) -> dict[str, Any]:
    location = {
        "city": normalize_text(city),
        "state": normalize_text(state),
        "country": normalize_text(country),
        "country_code": normalize_text(country_code),
    }
    if text and not location["country"]:
        parts = [part.strip() for part in str(text).split(",") if part.strip()]
        if len(parts) >= 1 and not location["city"]:
            location["city"] = parts[0]
        if len(parts) >= 2 and not location["state"]:
            location["state"] = parts[1]
        if parts:
            last = parts[-1]
            if last.lower() in {"india", "in"}:
                location["country"] = "India"
                location["country_code"] = "IN"
            elif not location["country"]:
                location["country"] = last
    if not location["country"] and FILTERS.get("country"):
        location["country"] = FILTERS.get("country")
    if not location["country_code"] and FILTERS.get("country_code"):
        location["country_code"] = FILTERS.get("country_code")
    return location

def pick_first(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None

def make_job(
    title: Any,
    job_id: Any,
    location: dict[str, Any],
    url: Any,
    apply_url: Any,
    date_posted: Any = None,
    date_posted_text: Any = None,
    job_description: Any = None,
    employment_type: Any = None,
    work_type: Any = None,
    salary_range: Any = None,
) -> dict[str, Any]:
    return {
        "title": normalize_text(title),
        "job_id": string_or_none(job_id),
        "location": location,
        "url": string_or_none(url),
        "apply_url": string_or_none(apply_url),
        "date_posted": string_or_none(date_posted),
        "date_posted_text": string_or_none(date_posted_text),
        "job_description": normalize_text(job_description),
        "employment_type": string_or_none(employment_type),
        "work_type": string_or_none(work_type),
        "salary_range": string_or_none(salary_range),
    }

def is_india_job(job: dict[str, Any]) -> bool:
    location = job.get("location") or {}
    country = str(location.get("country") or "").strip().lower()
    country_code = str(location.get("country_code") or "").strip().upper()
    return country in {"india", "in"} or country_code == "IN"

def _bs(html: str):
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        raise RuntimeError("beautifulsoup4 is required to run HTML scrapers") from exc
    return BeautifulSoup(html, "html.parser")

def fetch_html(url: str) -> str:
    response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return response.text

def absolute_url(base: str, href: str | None) -> str | None:
    if not href:
        return None
    if href.startswith("http://") or href.startswith("https://"):
        return href
    from urllib.parse import urljoin
    return urljoin(base, href)

def set_query_param(url: str, key: str, value: Any) -> str:
    from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query[key] = str(value)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, urlencode(query), parsed.fragment))

def find_next_url(soup: Any, current_url: str) -> str | None:
    next_selector = PAGINATION.get("next_link_selector")
    if next_selector:
        next_node = soup.select_one(str(next_selector))
        if next_node and getattr(next_node, "get", None):
            return absolute_url(current_url, next_node.get("href"))
    return None

def extract_jobs_from_page(html: str, current_url: str):
    soup = _bs(html)
    items = soup.select('a')
    for item in items:
        title_node = item.select_one('a') if 'a' else item
        url_node = item.select_one('a') if 'a' else item
        location_node = item.select_one(None) if None else None
        description_node = item.select_one(None) if None else None
        title = pick_first(title_node.get_text(" ", strip=True) if title_node else None)
        link = None
        if url_node and getattr(url_node, "get", None):
            link = url_node.get("href")
        apply_url = absolute_url(current_url, link) or current_url
        location_text = location_node.get_text(" ", strip=True) if location_node else None
        location = parse_location(location_text)
        job = make_job(
            title=title,
            job_id=apply_url,
            location=location,
            url=apply_url,
            apply_url=apply_url,
            date_posted=None,
            date_posted_text=None,
            job_description=description_node.get_text(" ", strip=True) if description_node else None,
            employment_type=None,
            work_type=None,
            salary_range=None,
        )
        if is_india_job(job):
            yield job

def iter_jobs():
    page_type = str(PAGINATION.get("type") or "single")
    param = str(PAGINATION.get("param") or "page")
    start = int(PAGINATION.get("start") or 1)
    step = int(PAGINATION.get("step") or 1)
    max_pages = int(PAGINATION.get("max_pages") or 50)
    current_url = ENTRY_URL
    for index in range(max_pages):
        if page_type in ('page', 'offset'):
            page_url = set_query_param(current_url, param, start + index * step)
        else:
            page_url = current_url
        page_html = fetch_html(page_url)
        yielded_any = False
        for item in extract_jobs_from_page(page_html, page_url):
            yielded_any = True
            yield item
        if page_type == "next_link":
            next_url = find_next_url(_bs(page_html), page_url)
            if not next_url or next_url == current_url:
                break
            current_url = next_url
            continue
        if not yielded_any:
            break


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = list(iter_jobs())
    with output_path.open("w", encoding="utf-8") as fh:
        for job in jobs:
            if job and is_india_job(job):
                fh.write(json.dumps(job, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output_path), "jobs": len(jobs)}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
