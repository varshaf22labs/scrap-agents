from __future__ import annotations

import json
import threading
from pathlib import Path
from textwrap import dedent
from typing import Any

from agents.context import AgentContext
from llm_client import LLMClient, parse_json_object
from models import ScrapePlan


class ScraperPlanError(ValueError):
    """Raised when the investigation-provided plan lacks a field code generation
    needs, so we refuse to fabricate a selector/path instead of guessing wrong."""


class CodeGenerationAgent:
    def __init__(self, llm_client: LLMClient | None = None, review_timeout_seconds: float = 15.0):
        self.llm_client = llm_client
        self.review_timeout_seconds = review_timeout_seconds

    def run(self, context: AgentContext) -> Path:
        if not context.plan:
            raise ValueError("Plan is required before code generation")
        script = self._render_script(context.plan)
        script = self._finalize_script(script)
        output_dir = context.generated_dir / context.normalized_domain.replace(".", "_")
        output_dir.mkdir(parents=True, exist_ok=True)
        script_path = output_dir / "scraper.py"
        script_path.write_text(script, encoding="utf-8")
        context.generated_script_path = script_path
        return script_path

    def _review_with_llm(self, plan: ScrapePlan, script: str) -> str | None:
        prompt = (
            "Review the following standalone Python scraper and return the full corrected script only if you can improve it.\n"
            "Requirements: no regex for field extraction, no LLM calls at runtime, preserve JSONL schema, and keep missing fields null.\n"
            "If no changes are needed, return the exact script unchanged.\n"
            f"Plan: {json.dumps(plan.__dict__)[:8000]}\n"
            f"Script:\n{script[:20000]}"
        )
        response_box: dict[str, str] = {}
        error_box: dict[str, Exception] = {}

        def _call_llm() -> None:
            try:
                response_box["response"] = self.llm_client.chat(
                    [
                        {"role": "system", "content": "Return only a full Python script. No markdown fences."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )
            except Exception as exc:
                error_box["error"] = exc

        worker = threading.Thread(target=_call_llm, daemon=True)
        worker.start()
        worker.join(timeout=self.review_timeout_seconds)
        if worker.is_alive():
            return None
        if error_box:
            return None
        response = response_box.get("response")
        if not response:
            return None
        try:
            cleaned = response.strip()
        except Exception:
            return None
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        if "def main()" not in cleaned or "jsonl" not in cleaned.lower():
            return None
        cleaned = self._finalize_script(cleaned)
        if not self._is_valid_python(cleaned):
            return None
        return cleaned

    def _render_script(self, plan: ScrapePlan) -> str:
        if self._should_use_api_script(plan):
            try:
                return self._finalize_script(self._render_api_script(plan))
            except ScraperPlanError:
                if self._has_structural_listing(plan):
                    return self._finalize_script(self._render_spa_script(plan))
                raise
        return self._finalize_script(self._render_spa_script(plan))

    def _should_use_api_script(self, plan: ScrapePlan) -> bool:
        if plan.strategy not in {"api", "json"}:
            return False
        listing = self._coerce_mapping(plan.listing)
        detail = self._coerce_mapping(plan.detail)
        has_json_items = bool(
            self._path_value(listing.get("json_items_path"), [])
            or self._path_value(listing.get("items_path"), [])
            or self._path_value(listing.get("content_path"), [])
        )
        has_structural_listing = bool(
            self._selector_value(listing.get("items_selector"), None)
            or self._selector_value(listing.get("job_selector"), None)
            or self._selector_value(listing.get("listing_selector"), None)
        )
        has_json_detail = bool(
            self._path_value(detail.get("url_path"), [])
            or self._path_value(detail.get("description_path"), [])
        )
        if has_json_items:
            return True
        if has_structural_listing and not has_json_detail:
            return False
        return has_json_items

    def _has_structural_listing(self, plan: ScrapePlan) -> bool:
        listing = self._coerce_mapping(plan.listing)
        return bool(
            self._selector_value(listing.get("items_selector"), None)
            or self._selector_value(listing.get("job_selector"), None)
            or self._selector_value(listing.get("listing_selector"), None)
        )

    def _finalize_script(self, script: str) -> str:
        cleaned = dedent("\n" + script).lstrip("\ufeff").lstrip()
        return cleaned

    def _is_valid_python(self, script: str) -> bool:
        try:
            compile(script, "<generated_scraper>", "exec")
        except SyntaxError:
            return False
        return True

    def _selector_value(self, value: Any, default: str | None = None) -> str | None:
        if isinstance(value, str):
            text = value.strip()
            return text or default
        if isinstance(value, (list, tuple)):
            for item in value:
                resolved = self._selector_value(item, None)
                if resolved:
                    return resolved
            return default
        if isinstance(value, dict):
            for key in ("selector", "css", "value", "text", "path"):
                if key in value:
                    resolved = self._selector_value(value[key], None)
                    if resolved:
                        return resolved
            return default
        if value is None:
            return default
        text = str(value).strip()
        return text or default

    def _require_selector(self, value: Any, field_name: str) -> str:
        """Resolve a selector from the investigation plan only. Never falls back
        to a hardcoded guess — if the plan doesn't have it, generation stops here
        with a clear error rather than emitting a scraper that silently matches 0 elements."""
        resolved = self._selector_value(value, None)
        if not resolved:
            raise ScraperPlanError(
                f"Investigation plan is missing a required selector for '{field_name}'. "
                "Code generation will not fabricate a default selector for this field — "
                "re-run investigation or add it to the plan manually."
            )
        return resolved

    def _path_value(self, value: Any, default: list[Any] | None = None) -> list[Any]:
        if default is None:
            default = []
        if value is None:
            return list(default)
        if isinstance(value, dict):
            for key in ("path", "json_path", "value"):
                if key in value:
                    return self._path_value(value[key], default)
            return list(default)
        if isinstance(value, list):
            if not value:
                return list(default)
            if value[0] == "investigation":
                return list(default)
            return [item for item in value if item is not None]
        if isinstance(value, tuple):
            return [item for item in value if item is not None]
        if isinstance(value, str):
            text = value.strip()
            if not text or text.startswith("investigation."):
                return list(default)
            return [text]
        return [value]

    def _require_path(self, value: Any, field_name: str) -> list[Any]:
        """Same idea as _require_selector, but for JSON extraction paths. No more
        silently defaulting to a guessed path like ["jobs"] or ["title"]."""
        resolved = self._path_value(value, [])
        if not resolved:
            raise ScraperPlanError(
                f"Investigation plan is missing a required JSON path for '{field_name}'. "
                "Code generation will not fabricate a default path for this field — "
                "re-run investigation or add it to the plan manually."
            )
        return resolved

    def _coerce_mapping(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        coerced: dict[str, Any] = {}
        for key, item in value.items():
            if item is None:
                continue
            if key.endswith("selector") or key.endswith("_selector"):
                coerced[key] = self._selector_value(item, None)
            elif key.endswith("path") or key.endswith("_path"):
                coerced[key] = self._path_value(item, [])
            elif isinstance(item, dict):
                coerced[key] = self._coerce_mapping(item)
            elif isinstance(item, list):
                coerced[key] = [self._coerce_value(item_value) for item_value in item]
            else:
                coerced[key] = self._coerce_value(item)
        return coerced

    def _coerce_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return self._coerce_mapping(value)
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return self._coerce_value(value[0])
            return [self._coerce_value(item) for item in value]
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    def _has_selector(self, value: Any) -> bool:
        return bool(self._selector_value(value, None))

    def _header(self, plan: ScrapePlan) -> str:
        pagination = self._coerce_mapping(plan.pagination)
        listing = self._coerce_mapping(plan.listing)
        detail = self._coerce_mapping(plan.detail)
        filters = self._coerce_mapping(plan.filters)
        return dedent(
            f'''\
            #!/usr/bin/env python3
            from __future__ import annotations

            import argparse
            import json
            import sys
            from datetime import datetime
            from pathlib import Path
            from typing import Any

            import requests

            ENTRY_URL = {plan.entry_url!r}
            DEFAULT_OUTPUT = {plan.output_path!r}
            PAGINATION = {pagination!r}
            LISTING = {listing!r}
            DETAIL = {detail!r}
            FILTERS = {filters!r}

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
                location = {{
                    "city": normalize_text(city),
                    "state": normalize_text(state),
                    "country": normalize_text(country),
                    "country_code": normalize_text(country_code),
                }}
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
                return {{
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
                }}

            def is_india_job(job: dict[str, Any]) -> bool:
                location = job.get("location") or {{}}
                country = str(location.get("country") or "").strip().lower()
                country_code = str(location.get("country_code") or "").strip().upper()
                return country in {{"india", "in"}} or "india" in country or country_code == "IN"
            '''
        )

    def _footer(self) -> str:
        return dedent(
            '''
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
                            fh.write(json.dumps(job, ensure_ascii=False) + "\\n")
                print(json.dumps({"output": str(output_path), "jobs": len(jobs)}, ensure_ascii=False))
                return 0

            if __name__ == "__main__":
                raise SystemExit(main())
            '''
        )

    def _render_api_script(self, plan: ScrapePlan) -> str:
        listing_items_path = json.dumps(self._require_path(plan.listing.get("json_items_path"), "listing.json_items_path"))
        title_path = json.dumps(self._path_value(plan.listing.get("title_path"), []))
        name_path = json.dumps(self._path_value(plan.listing.get("name_path"), []))
        job_id_path = json.dumps(self._path_value(plan.listing.get("job_id_path"), []))
        id_path = json.dumps(self._path_value(plan.listing.get("id_path"), []))
        location_path = json.dumps(self._path_value(plan.listing.get("location_path"), []))
        city_path = json.dumps(self._path_value(plan.listing.get("city_path"), []))
        state_path = json.dumps(self._path_value(plan.listing.get("state_path"), []))
        country_path = json.dumps(self._path_value(plan.listing.get("country_path"), []))
        country_code_path = json.dumps(self._path_value(plan.listing.get("country_code_path"), []))
        date_posted_path = json.dumps(self._path_value(plan.listing.get("date_posted_path"), []))
        date_posted_text_path = json.dumps(self._path_value(plan.listing.get("date_posted_text_path"), []))
        employment_type_path = json.dumps(self._path_value(plan.listing.get("employment_type_path"), []))
        work_type_path = json.dumps(self._path_value(plan.listing.get("work_type_path"), []))
        salary_range_path = json.dumps(self._path_value(plan.listing.get("salary_range_path"), []))
        detail_url_path = json.dumps(self._path_value(plan.detail.get("url_path"), []))
        detail_desc_path = json.dumps(self._path_value(plan.detail.get("description_path"), []))
        return (
            self._header(plan)
            + dedent(
                f'''
                def fetch_json(url: str, params: dict[str, Any] | None = None) -> Any:
                    response = requests.get(url, params=params or {{}}, timeout=30, headers={{"User-Agent": "Mozilla/5.0"}})
                    response.raise_for_status()
                    return response.json()

                def extract_jobs_from_payload(payload: Any) -> list[Any]:
                    items = read_json_path(payload, {listing_items_path})
                    if isinstance(items, list):
                        return items
                    return []

                def iter_jobs():
                    page_type = str(PAGINATION.get("type") or "page")
                    param = str(PAGINATION.get("param") or "page")
                    start = int(PAGINATION.get("start") or 1)
                    step = int(PAGINATION.get("step") or 1)
                    max_pages = int(PAGINATION.get("max_pages") or 50)
                    single_page = page_type not in {"page", "offset", "cursor"}
                    if single_page:
                        max_pages = 1
                    url = ENTRY_URL
                    cursor = PAGINATION.get("cursor")
                    for index in range(max_pages):
                        params: dict[str, Any] = {{}}
                        if page_type == "offset":
                            params[param] = start + index * step
                        elif page_type == "cursor":
                            if cursor:
                                params[param] = cursor
                        else:
                            params[param] = start + index * step
                        payload = fetch_json(url, params=params)
                        items = extract_jobs_from_payload(payload)
                        if not items:
                            if index == 0:
                                top_level = list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
                                raise RuntimeError(
                                    f"json_items_path {listing_items_path} matched 0 items in the API response "
                                    f"from {{url}} (params={{params}}). Top-level payload shape: {{top_level}}. "
                                    "The investigation-provided path does not match this response."
                                )
                            break
                        for item in items:
                            title = pick_first(
                                read_json_path(item, {title_path}),
                                read_json_path(item, {name_path}),
                            )
                            job_id = pick_first(
                                read_json_path(item, {job_id_path}),
                                read_json_path(item, {id_path}),
                            )
                            url_value = read_json_path(item, {detail_url_path})
                            apply_url = url_value
                            location_value = read_json_path(item, {location_path})
                            city = read_json_path(item, {city_path})
                            state = read_json_path(item, {state_path})
                            country = read_json_path(item, {country_path})
                            country_code = read_json_path(item, {country_code_path})
                            description = read_json_path(item, {detail_desc_path})
                            job = make_job(
                                title=title,
                                job_id=job_id,
                                location=parse_location(location_value, city=city, state=state, country=country, country_code=country_code),
                                url=url_value,
                                apply_url=apply_url,
                                date_posted=read_json_path(item, {date_posted_path}),
                                date_posted_text=read_json_path(item, {date_posted_text_path}),
                                job_description=description,
                                employment_type=read_json_path(item, {employment_type_path}),
                                work_type=read_json_path(item, {work_type_path}),
                                salary_range=read_json_path(item, {salary_range_path}),
                            )
                            yield job
                        if page_type == "cursor":
                            cursor_path = PAGINATION.get("next_cursor_path") or []
                            next_cursor = read_json_path(payload, cursor_path) if cursor_path else None
                            if not next_cursor or next_cursor == cursor:
                                break
                            cursor = next_cursor
                            continue
                        if len(items) == 0:
                            break
                        if single_page:
                            break

                '''
            )
            + self._footer()
        )

    def _render_html_script(self, plan: ScrapePlan) -> str:
        listing_selector = self._selector_value(
            plan.listing.get("items_selector")
            or plan.listing.get("item_selector")
            or plan.listing.get("job_selector"),
            None,
        )
        if not listing_selector:
            raise ScraperPlanError(
                "Investigation plan is missing a required selector for 'listing.items_selector'. "
                "Code generation will not fabricate a default selector for this field."
            )
        title_selector = self._selector_value(plan.listing.get("title_selector") or plan.listing.get("job_selector"), None)
        url_selector = self._selector_value(plan.listing.get("url_selector") or plan.listing.get("job_selector"), None)
        location_selector = self._selector_value(plan.listing.get("location_selector"), None)
        description_selector = self._selector_value(plan.detail.get("description_selector"), None) or self._selector_value(plan.listing.get("description_selector"), None)
        date_posted_text_selector = self._selector_value(plan.listing.get("date_posted_text_selector"), None) or self._selector_value(plan.detail.get("date_posted_text_selector"), None)
        employment_type_selector = self._selector_value(plan.listing.get("employment_type_selector"), None) or self._selector_value(plan.detail.get("employment_type_selector"), None)
        work_type_selector = self._selector_value(plan.listing.get("work_type_selector"), None) or self._selector_value(plan.detail.get("work_type_selector"), None)
        salary_range_selector = self._selector_value(plan.listing.get("salary_range_selector"), None) or self._selector_value(plan.detail.get("salary_range_selector"), None)
        return (
            self._header(plan)
            + dedent(
                f'''
                LISTING_SELECTOR_TEXT = {listing_selector!r}

                def _bs(html: str):
                    try:
                        from bs4 import BeautifulSoup
                    except Exception as exc:
                        raise RuntimeError("beautifulsoup4 is required to run HTML scrapers") from exc
                    return BeautifulSoup(html, "html.parser")

                def fetch_html(url: str) -> str:
                    response = requests.get(url, timeout=30, headers={{"User-Agent": "Mozilla/5.0"}})
                    response.raise_for_status()
                    return response.text

                def fetch_detail_fields(url: str) -> tuple[Any, Any, Any, Any, Any]:
                    detail_html = fetch_html(url)
                    detail_soup = _bs(detail_html)
                    description_node = detail_soup.select_one({description_selector!r}) if {description_selector!r} else None
                    date_node = detail_soup.select_one({date_posted_text_selector!r}) if {date_posted_text_selector!r} else None
                    employment_node = detail_soup.select_one({employment_type_selector!r}) if {employment_type_selector!r} else None
                    work_type_node = detail_soup.select_one({work_type_selector!r}) if {work_type_selector!r} else None
                    salary_node = detail_soup.select_one({salary_range_selector!r}) if {salary_range_selector!r} else None
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
                    items = soup.select({listing_selector!r})
                    if not items and page_index == 0:
                        raise RuntimeError(
                            f"items_selector {{LISTING_SELECTOR_TEXT!r}} matched 0 elements on {{current_url}}. "
                            "The investigation-provided selector does not match this page's structure "
                            "(page may render via JS, or the markup has changed)."
                        )
                    for item in items:
                        title_node = item.select_one({title_selector!r}) if {title_selector!r} else None
                        url_node = item.select_one({url_selector!r}) if {url_selector!r} else None
                        location_node = item.select_one({location_selector!r}) if {location_selector!r} else None
                        description_node = item.select_one({description_selector!r}) if {description_selector!r} else None
                        date_node = item.select_one({date_posted_text_selector!r}) if {date_posted_text_selector!r} else None
                        employment_node = item.select_one({employment_type_selector!r}) if {employment_type_selector!r} else None
                        work_type_node = item.select_one({work_type_selector!r}) if {work_type_selector!r} else None
                        salary_node = item.select_one({salary_range_selector!r}) if {salary_range_selector!r} else None
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
                    single_page = page_type not in {"page", "offset", "next_link"}
                    if single_page:
                        max_pages = 1
                    current_url = ENTRY_URL
                    for index in range(max_pages):
                        if page_type in {"page", "offset"}:
                            page_url = set_query_param(current_url, param, start + index * step)
                        else:
                            page_url = current_url
                        page_html = fetch_html(page_url)
                        yielded_any = False
                        for item in extract_jobs_from_page(page_html, page_url, index):
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
                        if single_page:
                            break

                '''
            )
            + self._footer()
        )

    def _render_spa_script(self, plan: ScrapePlan) -> str:
        listing_selector = self._selector_value(
            plan.listing.get("items_selector")
            or plan.listing.get("item_selector")
            or plan.listing.get("job_selector"),
            None,
        )
        if not listing_selector:
            raise ScraperPlanError(
                "Investigation plan is missing a required selector for 'listing.items_selector'. "
                "Code generation will not fabricate a default selector for this field."
            )
        title_selector = self._selector_value(plan.listing.get("title_selector") or plan.listing.get("job_selector"), None)
        url_selector = self._selector_value(plan.listing.get("url_selector") or plan.listing.get("job_selector"), None)
        location_selector = self._selector_value(plan.listing.get("location_selector"), None)
        description_selector = self._selector_value(plan.detail.get("description_selector"), None) or self._selector_value(plan.listing.get("description_selector"), None)
        date_posted_text_selector = self._selector_value(plan.listing.get("date_posted_text_selector"), None) or self._selector_value(plan.detail.get("date_posted_text_selector"), None)
        employment_type_selector = self._selector_value(plan.listing.get("employment_type_selector"), None) or self._selector_value(plan.detail.get("employment_type_selector"), None)
        work_type_selector = self._selector_value(plan.listing.get("work_type_selector"), None) or self._selector_value(plan.detail.get("work_type_selector"), None)
        salary_range_selector = self._selector_value(plan.listing.get("salary_range_selector"), None) or self._selector_value(plan.detail.get("salary_range_selector"), None)
        return (
            self._header(plan)
            + dedent(
                f'''
                LISTING_SELECTOR_TEXT = {listing_selector!r}

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
                    detail_html, detail_url = fetch_rendered_html(url, wait_selector={description_selector!r})
                    detail_soup = _bs(detail_html)
                    description_node = detail_soup.select_one({description_selector!r}) if {description_selector!r} else None
                    date_node = detail_soup.select_one({date_posted_text_selector!r}) if {date_posted_text_selector!r} else None
                    employment_node = detail_soup.select_one({employment_type_selector!r}) if {employment_type_selector!r} else None
                    work_type_node = detail_soup.select_one({work_type_selector!r}) if {work_type_selector!r} else None
                    salary_node = detail_soup.select_one({salary_range_selector!r}) if {salary_range_selector!r} else None
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
                    items = soup.select({listing_selector!r})
                    if not items and page_index == 0:
                        raise RuntimeError(
                            f"items_selector {{LISTING_SELECTOR_TEXT!r}} matched 0 elements on {{current_url}}. "
                            "The investigation-provided selector does not match the rendered page "
                            "(possible causes: page needs longer JS wait, selector is stale, or Playwright wasn't available so we fell back to raw HTML)."
                        )
                    for item in items:
                        title_node = item.select_one({title_selector!r}) if {title_selector!r} else None
                        url_node = item.select_one({url_selector!r}) if {url_selector!r} else None
                        location_node = item.select_one({location_selector!r}) if {location_selector!r} else None
                        description_node = item.select_one({description_selector!r}) if {description_selector!r} else None
                        date_node = item.select_one({date_posted_text_selector!r}) if {date_posted_text_selector!r} else None
                        employment_node = item.select_one({employment_type_selector!r}) if {employment_type_selector!r} else None
                        work_type_node = item.select_one({work_type_selector!r}) if {work_type_selector!r} else None
                        salary_node = item.select_one({salary_range_selector!r}) if {salary_range_selector!r} else None
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
                    single_page = page_type not in {"page", "offset", "next_link"}
                    if single_page:
                        max_pages = 1
                    current_url = ENTRY_URL
                    for index in range(max_pages):
                        if page_type in {"page", "offset"}:
                            page_url = set_query_param(current_url, param, start + index * step)
                        else:
                            page_url = current_url
                        page_html, rendered_url = fetch_rendered_html(page_url, wait_selector={listing_selector!r})
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

                '''
            )
            + self._footer()
        )
