#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

ENTRY_URL = 'https://www.zoho.com/careers/'
DEFAULT_OUTPUT = 'output\\jobs.jsonl'
PAGINATION = {'type': 'single'}
LISTING = {'items_selector': 'div[class~="zgh-accounts"]', 'title_selector': 'a[class~="zgh-signup"]', 'url_selector': 'a[class~="zgh-signup"]', 'json_items_path': [], 'items_path': [], 'title_path': ['selectors', 'listing', 'title_selector'], 'url_path': ['selectors', 'listing', 'url_selector'], 'job_selector': 'div[class~="zgh-accounts"]', 'listing_selector': 'div[class~="zgh-accounts"]'}
DETAIL = {'description_path': ['description'], 'url_path': ['url'], 'detail_strategy': 'structural', 'other_fields': {}}
FILTERS = {}

def read_json_path(value: Any, path: list[Any]) -> Any:
    if not path:
        return None
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

def parse_date_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    text = normalize_text(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        return text

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
            last = parts[-1].lower()
            if "india" in last or last == "in" or last.startswith("india "):
                location["country"] = "India"
                location["country_code"] = "IN"
            elif not location["country"]:
                location["country"] = parts[-1]
    if not location["country"]:
        lowered_text = str(text or "").lower()
        if "india" in lowered_text:
            location["country"] = "India"
            location["country_code"] = "IN"
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
        "date_posted": parse_date_value(date_posted),
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
    return country in {"india", "in"} or "india" in country or country_code == "IN"

LISTING_SELECTOR_TEXT = 'div[class~="zgh-accounts"]'

def _bs(html: str):
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        raise RuntimeError("beautifulsoup4 is required to run SPA scrapers") from exc
    return BeautifulSoup(html, "html.parser")

def fetch_rendered_html(url: str, wait_selector: str | None = None) -> tuple[str, str]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        raise RuntimeError("playwright is required to run generated website scrapers")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_load_state("networkidle", timeout=30000)
            if wait_selector:
                page.wait_for_selector(wait_selector, timeout=15000)
            return page.content(), page.url
        finally:
            browser.close()

def fetch_detail_fields(url: str) -> tuple[Any, Any, Any, Any, Any]:
    detail_html, detail_url = fetch_rendered_html(url, wait_selector=None)
    detail_soup = _bs(detail_html)
    description_node = detail_soup.select_one(None) if None else None
    date_node = detail_soup.select_one(None) if None else None
    employment_node = detail_soup.select_one(None) if None else None
    work_type_node = detail_soup.select_one(None) if None else None
    salary_node = detail_soup.select_one(None) if None else None
    return (
        description_node.get_text(" ", strip=True) if description_node else None,
        date_node.get_text(" ", strip=True) if date_node else None,
        employment_node.get_text(" ", strip=True) if employment_node else None,
        work_type_node.get_text(" ", strip=True) if work_type_node else None,
        salary_node.get_text(" ", strip=True) if salary_node else None,
    )

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

def extract_jobs_from_page(html: str, current_url: str, page_index: int):
    soup = _bs(html)
    items = soup.select('div[class~="zgh-accounts"]')
    if not items and page_index == 0:
        raise RuntimeError(
            f"items_selector {LISTING_SELECTOR_TEXT!r} matched 0 elements on {current_url}. "
            "The investigation-provided selector does not match the rendered page "
            "(possible causes: page needs longer JS wait, selector is stale, or Playwright wasn't available so we fell back to raw HTML)."
        )
    for item in items:
        title_node = item.select_one('a[class~="zgh-signup"]')
        url_node = item.select_one('a[class~="zgh-signup"]')
        location_node = item.select_one(None) if None else None
        description_node = item.select_one(None) if None else None
        date_node = item.select_one(None) if None else None
        employment_node = item.select_one(None) if None else None
        work_type_node = item.select_one(None) if None else None
        salary_node = item.select_one(None) if None else None
        title = pick_first(title_node.get_text(" ", strip=True) if title_node else None)
        link = None
        if url_node and getattr(url_node, "get", None):
            link = url_node.get("href")
        apply_url = absolute_url(current_url, link) or current_url
        location_text = location_node.get_text(" ", strip=True) if location_node else None
        location = parse_location(location_text)
        description_text = description_node.get_text(" ", strip=True) if description_node else None
        date_text = date_node.get_text(" ", strip=True) if date_node else None
        employment_text = employment_node.get_text(" ", strip=True) if employment_node else None
        work_type_text = work_type_node.get_text(" ", strip=True) if work_type_node else None
        salary_text = salary_node.get_text(" ", strip=True) if salary_node else None
        if apply_url and (description_text is None or date_text is None or employment_text is None or work_type_text is None or salary_text is None):
            detail_description, detail_date, detail_employment, detail_work_type, detail_salary = fetch_detail_fields(apply_url)
            description_text = description_text or detail_description
            date_text = date_text or detail_date
            employment_text = employment_text or detail_employment
            work_type_text = work_type_text or detail_work_type
            salary_text = salary_text or detail_salary
        job = make_job(
            title=title,
            job_id=apply_url,
            location=location,
            url=apply_url,
            apply_url=apply_url,
            date_posted=date_text,
            date_posted_text=date_text,
            job_description=description_text,
            employment_type=employment_text,
            work_type=work_type_text,
            salary_range=salary_text,
        )
        if is_india_job(job):
            yield job

def iter_jobs():
    page_type = str(PAGINATION.get("type") or "single")
    param = str(PAGINATION.get("param") or "page")
    start = int(PAGINATION.get("start") or 1)
    step = int(PAGINATION.get("step") or 1)
    max_pages = int(PAGINATION.get("max_pages") or 50)
    single_page = page_type not in ('page', 'offset', 'next_link')
    if single_page:
        max_pages = 1
    current_url = ENTRY_URL
    for index in range(max_pages):
        if page_type in ('page', 'offset'):
            page_url = set_query_param(current_url, param, start + index * step)
        else:
            page_url = current_url
        page_html, rendered_url = fetch_rendered_html(page_url, wait_selector='div[class~="zgh-accounts"]')
        yielded_any = False
        for item in extract_jobs_from_page(page_html, rendered_url, index):
            yielded_any = True
            yield item
        if page_type == "next_link":
            next_url = find_next_url(_bs(page_html), rendered_url)
            if not next_url or next_url == current_url:
                break
            current_url = next_url
            continue
        if not yielded_any:
            break
        if single_page:
            break


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_path = Path(args.output)
    if output_path.suffix.lower() != ".jsonl":
        output_path = output_path.with_suffix(".jsonl")
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
