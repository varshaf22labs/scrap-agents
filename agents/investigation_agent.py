from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

import requests

from agents.context import AgentContext
from llm_client import LLMClient, parse_json_object
from models import InvestigationFinding
from tools.api_detector import detect_api_candidates
from tools.html_parser import extract_links
from tools.json_detector import detect_embedded_json
from tools.playwright import PlaywrightTool


# Known ATS/careers-hosting platforms, keyed by the domain (or domain suffix)
# their pages/APIs are served from. This lets us recognize the platform
# deterministically instead of relying on the LLM to guess it from HTML alone.
KNOWN_ATS_PLATFORMS: dict[str, str] = {
    "zohorecruit.in": "zoho_recruit",
    "zohorecruit.com": "zoho_recruit",
    "boards.greenhouse.io": "greenhouse",
    "greenhouse.io": "greenhouse",
    "jobs.lever.co": "lever",
    "lever.co": "lever",
    "myworkdayjobs.com": "workday",
    "ashbyhq.com": "ashby",
    "smartrecruiters.com": "smartrecruiters",
    "bamboohr.com": "bamboohr",
    "workable.com": "workable",
    "recruitee.com": "recruitee",
    "breezy.hr": "breezyhr",
    "icims.com": "icims",
    "successfactors.com": "successfactors",
    "personio.de": "personio",
    "freshteam.com": "freshteam",
    "keka.com": "keka",
    "darwinbox.com": "darwinbox",
}

# Platforms with a stable, documented public JSON API we can actually trust.
# Everything else in KNOWN_ATS_PLATFORMS has no reliable public API, so an
# LLM-"detected" api_url for those is almost always a guess, not a fact.
ATS_WITH_RELIABLE_API = {"greenhouse", "lever", "smartrecruiters", "recruitee", "workable"}


def detect_known_platform(url: str | None) -> str | None:
    """Return a known ATS platform name if `url` is hosted on one, else None."""
    if not url:
        return None
    host = (urlparse(url).netloc or "").lower()
    for domain_suffix, platform in KNOWN_ATS_PLATFORMS.items():
        if host == domain_suffix or host.endswith("." + domain_suffix):
            return platform
    return None


class InvestigationAgent:
    def __init__(self, llm_client: LLMClient | None = None, timeout_seconds: int = 30):
        self.llm_client = llm_client
        self.timeout_seconds = timeout_seconds

    def run(self, context: AgentContext) -> InvestigationFinding:
        evidence: list[dict[str, Any]] = []
        target_candidate = self._select_target_candidate(context.candidate_pages)
        if not target_candidate and context.candidate_pages:
            target_candidate = context.candidate_pages[0]

        chosen_url = (target_candidate.final_url or target_candidate.url) if target_candidate else None
        source_type = "unknown"
        platform: str | None = None
        api_url: str | None = None
        json_paths: list[list[Any]] = []
        selectors: dict[str, Any] = {}
        listing: dict[str, Any] = {}
        pagination: dict[str, Any] = {}
        detail: dict[str, Any] = {}
        job_titles: list[str] = []
        job_count_detected = 0
        jobs_found = False
        listing_selector: str | None = None
        job_selector: str | None = None
        detail_selector: str | None = None

        inspected_urls = self._inspection_candidates(context.candidate_pages, target_candidate)

        for candidate in inspected_urls:
            url = candidate.final_url or candidate.url
            snapshot = self._fetch_with_playwright(url)
            if snapshot:
                html = snapshot.html or ""
                response_url = snapshot.final_url or url
                response_status = snapshot.status_code
                response_title = snapshot.title
                signals = snapshot.signals or {}
                embedded_json = signals.get("embedded_json") or detect_embedded_json(html)
                api_candidates = signals.get("api_candidates") or detect_api_candidates(html, response_url)
                sample_links = signals.get("sample_links") or extract_links(html, response_url)[:50]
                pagination = self._merge_dicts(pagination, self._normalize_pagination(signals.get("pagination_hints")))
                listing = self._merge_dicts(listing, self._build_listing_hints(html, response_url, signals))
                detail = self._merge_dicts(detail, self._build_detail_hints())
                selectors = self._merge_dicts(selectors, self._build_selector_hints(listing, detail, pagination))
                try:
                    visible_jobs = self._extract_visible_jobs(html, listing)
                except Exception as exc:
                    evidence.append({"url": response_url, "selector_error": str(exc), "stage": "visible_jobs"})
                    visible_jobs = {"titles": [], "count": 0, "listing_selector": None, "job_selector": None, "detail_selector": None}
                job_titles = visible_jobs["titles"] or job_titles
                job_count_detected = max(job_count_detected, visible_jobs["count"])
                jobs_found = jobs_found or visible_jobs["count"] > 0
                listing_selector = listing_selector or visible_jobs["listing_selector"]
                job_selector = job_selector or visible_jobs["job_selector"]
                detail_selector = detail_selector or visible_jobs["detail_selector"]
                api_url = api_url or self._extract_api_url(api_candidates)
                json_paths.extend(self._extract_json_paths(embedded_json))
                snippet = snapshot.text[:2000] if snapshot.text else html[:2000]
            else:
                try:
                    response = requests.get(
                        url,
                        timeout=self.timeout_seconds,
                        allow_redirects=True,
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                except requests.RequestException as exc:
                    evidence.append({"url": candidate.url, "error": str(exc)})
                    continue

                html = response.text or ""
                response_url = response.url
                response_status = response.status_code
                response_title = self._extract_title(html)
                embedded_json = detect_embedded_json(html)
                api_candidates = detect_api_candidates(html, response_url)
                sample_links = extract_links(html, response_url)[:50]
                pagination = self._merge_dicts(pagination, self._derive_pagination(html))
                listing = self._merge_dicts(listing, self._build_listing_hints(html, response_url, {}))
                detail = self._merge_dicts(detail, self._build_detail_hints())
                selectors = self._merge_dicts(selectors, self._build_selector_hints(listing, detail, pagination))
                try:
                    visible_jobs = self._extract_visible_jobs(html, listing)
                except Exception as exc:
                    evidence.append({"url": response_url, "selector_error": str(exc), "stage": "visible_jobs"})
                    visible_jobs = {"titles": [], "count": 0, "listing_selector": None, "job_selector": None, "detail_selector": None}
                job_titles = visible_jobs["titles"] or job_titles
                job_count_detected = max(job_count_detected, visible_jobs["count"])
                jobs_found = jobs_found or visible_jobs["count"] > 0
                listing_selector = listing_selector or visible_jobs["listing_selector"]
                job_selector = job_selector or visible_jobs["job_selector"]
                detail_selector = detail_selector or visible_jobs["detail_selector"]
                api_url = api_url or self._extract_api_url(api_candidates)
                json_paths.extend(self._extract_json_paths(embedded_json))
                snippet = html[:2000]

            evidence.append(
                {
                    "url": response_url,
                    "status_code": response_status,
                    "title": response_title,
                    "embedded_json_kinds": [item.get("kind") for item in embedded_json],
                    "api_candidates": api_candidates[:5],
                    "sample_links": sample_links[:10],
                    "snippet": snippet,
                }
            )

            has_json_signal = self._has_json_api_signal(snapshot.signals if snapshot else None, embedded_json, api_candidates)
            if embedded_json and source_type == "unknown":
                source_type = "json"
                chosen_url = response_url
            if has_json_signal and source_type == "unknown":
                source_type = "api"
                chosen_url = response_url
            if "jobs" in response_url.lower() or "careers" in response_url.lower():
                chosen_url = chosen_url or response_url

        if not chosen_url and context.candidate_pages:
            chosen_url = context.candidate_pages[0].final_url or context.candidate_pages[0].url

        if source_type == "unknown":
            source_type = "html"

        llm_summary = self._summarize_with_llm(context.domain, evidence)
        if llm_summary:
            platform = llm_summary.get("platform")
            source_type = llm_summary.get("source_type", source_type)
            listing = self._merge_missing_dicts(listing, llm_summary.get("listing") or {})
            pagination = self._merge_missing_dicts(pagination, llm_summary.get("pagination") or {})
            detail = self._merge_missing_dicts(detail, llm_summary.get("detail") or {})
            selectors = self._merge_missing_dicts(selectors, llm_summary.get("selectors") or {})
            api_url = llm_summary.get("api_url") or api_url
            json_paths.extend(self._normalize_json_paths(llm_summary.get("json_paths") or []))
            if llm_summary.get("careers_url"):
                chosen_url = llm_summary["careers_url"]

        # Deterministic ATS detection overrides any LLM/heuristic guesswork here,
        # because neither can actually verify that a guessed API endpoint exists.
        # This is what prevents the scraper from hitting a fabricated API URL for
        # platforms like Zoho Recruit, Workday, Ashby, etc. and crashing on a
        # non-JSON response.
        detected_platform = detect_known_platform(chosen_url) or detect_known_platform(api_url)
        if detected_platform:
            platform = detected_platform
            api_hosted_on_platform = detect_known_platform(api_url) == detected_platform
            if detected_platform in ATS_WITH_RELIABLE_API and api_url and api_hosted_on_platform:
                source_type = "api"
            else:
                source_type = "spa"
                api_url = None

        if jobs_found and listing_selector:
            if source_type in {"api", "json"} and not (
                platform in ATS_WITH_RELIABLE_API and api_url and detect_known_platform(api_url) == platform
            ):
                source_type = "spa"
                api_url = None

        if not platform and chosen_url:
            platform = detect_known_platform(chosen_url)

        recommended_strategy = self._recommend_strategy(source_type, jobs_found, api_url, listing_selector, job_count_detected)

        finding = InvestigationFinding(
            source_type=source_type,
            careers_url=chosen_url,
            platform=platform,
            api_url=api_url,
            jobs_found_during_investigation=jobs_found,
            job_count_detected=job_count_detected,
            job_titles=job_titles[:10],
            listing_selector=listing_selector,
            job_selector=job_selector,
            detail_selector=detail_selector,
            recommended_strategy=recommended_strategy,
            json_paths=self._dedupe_json_paths(json_paths),
            selectors=selectors,
            pagination=pagination,
            listing=listing,
            detail=detail,
            evidence=evidence,
        )
        context.investigation = finding
        return finding

    def _summarize_with_llm(self, domain: str, evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not self.llm_client:
            return None
        prompt = (
            "You are inspecting a company's careers source. Return strict JSON only.\n"
            "Pick the best source type and describe listing/detail extraction hints.\n"
            "Allowed source_type values: api, json, html, spa.\n"
            "Use structural selectors or JSON paths only. No regex.\n"
            "Try to identify selectors or paths for title, url/apply_url, location, date_posted_text, "
            "employment_type, work_type, salary_range, and job_description whenever the source exposes them.\n"
            f"Domain: {domain}\n"
            f"Evidence: {json.dumps(self._structured_evidence(evidence[:4]))}\n"
        )
        try:
            response = self.llm_client.chat(
                [
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            return parse_json_object(response)
        except Exception:
            return None

    def _fetch_with_playwright(self, url: str):
        try:
            tool = PlaywrightTool(timeout_seconds=self.timeout_seconds)
            return tool.fetch(url)
        except Exception:
            return None

    def _select_target_candidate(self, candidates: list[Any]) -> Any | None:
        if not candidates:
            return None
        return max(candidates, key=self._candidate_score)

    def _inspection_candidates(self, candidates: list[Any], target_candidate: Any | None) -> list[Any]:
        if not candidates:
            return []
        ranked = sorted(candidates, key=self._candidate_score, reverse=True)
        ordered: list[Any] = []
        seen: set[str] = set()
        for candidate in ([target_candidate] if target_candidate else []) + ranked:
            if not candidate:
                continue
            key = candidate.final_url or candidate.url
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(candidate)
            if len(ordered) >= 3:
                break
        return ordered

    def _candidate_score(self, candidate: Any) -> tuple[int, int, int, int]:
        url = (candidate.final_url or candidate.url or "").lower()
        title = (candidate.title or "").lower()
        source = (candidate.source or "").lower()
        careers = 2 if any(token in url for token in ["/careers", "/jobs", "career", "job"]) else 0
        search = 1 if source == "search" else 0
        status = 1 if getattr(candidate, "status_code", None) and int(candidate.status_code) < 400 else 0
        homepage = 1 if url.rstrip("/").count("/") <= 2 else 0
        title_hint = 1 if "career" in title or "job" in title else 0
        return (careers + title_hint + search + status, -homepage, search, status)

    def _extract_api_url(self, api_candidates: list[dict[str, Any]]) -> str | None:
        for candidate in api_candidates:
            if candidate.get("url"):
                return str(candidate["url"])
        return None

    def _extract_json_paths(self, embedded_json: list[dict[str, Any]]) -> list[list[Any]]:
        paths: list[list[Any]] = []
        for payload in embedded_json:
            paths.extend(self._collect_json_paths(payload.get("data"), []))
        return paths

    def _collect_json_paths(self, value: Any, prefix: list[Any]) -> list[list[Any]]:
        paths: list[list[Any]] = []
        if isinstance(value, dict):
            interesting_keys = {
                "title",
                "name",
                "url",
                "id",
                "job_id",
                "location",
                "description",
                "apply_url",
                "date_posted",
                "date_posted_text",
                "employment_type",
                "work_type",
                "salary_range",
            }
            if interesting_keys.intersection(value.keys()):
                paths.append(prefix.copy())
            for key, item in value.items():
                paths.extend(self._collect_json_paths(item, prefix + [key]))
        elif isinstance(value, list):
            if value:
                paths.extend(self._collect_json_paths(value[0], prefix + [0]))
        return paths

    def _dedupe_json_paths(self, paths: list[list[Any]]) -> list[list[Any]]:
        seen: set[tuple[Any, ...]] = set()
        unique: list[list[Any]] = []
        for path in paths:
            key = tuple(path)
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique

    def _normalize_json_paths(self, paths: Any) -> list[list[Any]]:
        normalized: list[list[Any]] = []
        if not isinstance(paths, list):
            return normalized
        for path in paths:
            if isinstance(path, list):
                normalized.append(path)
            elif isinstance(path, tuple):
                normalized.append(list(path))
            elif path is not None:
                normalized.append([path])
        return normalized

    def _derive_pagination(self, html: str) -> dict[str, Any]:
        lower = html.lower()
        pagination: dict[str, Any] = {}
        if "load more" in lower or "show more" in lower:
            pagination["type"] = "next_link"
        elif "?page=" in lower or "&page=" in lower:
            pagination.update({"type": "page", "param": "page", "start": 1, "step": 1})
        elif "?offset=" in lower or "&offset=" in lower:
            pagination.update({"type": "offset", "param": "offset", "start": 0, "step": 10})
        elif "cursor" in lower:
            pagination.update({"type": "cursor", "param": "cursor"})
        return pagination

    def _normalize_pagination(self, pagination: Any) -> dict[str, Any]:
        if isinstance(pagination, dict):
            return pagination
        if isinstance(pagination, list):
            for item in pagination:
                if isinstance(item, dict):
                    return item
        return {}

    def _build_listing_hints(self, html: str, base_url: str, signals: dict[str, Any]) -> dict[str, Any]:
        try:
            from bs4 import BeautifulSoup
        except Exception:
            sample_links = signals.get("sample_links") or extract_links(html, base_url)[:50]
            job_links = [link for link in sample_links if any(token in link.lower() for token in ["/career", "/jobs", "/job", "vacanc", "position"])]
            selector = "main a[href]" if job_links else "a[href]"
            return {
                "items_selector": selector,
                "title_selector": "a",
                "url_selector": "a[href]",
                "json_items_path": self._dedupe_json_paths(self._extract_json_paths(signals.get("embedded_json") or [])),
            }

        soup = BeautifulSoup(html, "html.parser")
        sample_links = signals.get("sample_links") or extract_links(html, base_url)[:50]
        job_links = [link for link in sample_links if any(token in link.lower() for token in ["/career", "/jobs", "/job", "vacanc", "position"])]
        candidate_anchor = None
        candidate_score = -10**9
        for anchor in soup.select("a[href]"):
            score = self._anchor_score(anchor, job_links)
            if score > candidate_score:
                candidate_score = score
                candidate_anchor = anchor

        container = self._best_container_for_anchor(candidate_anchor) if candidate_anchor else None

        items_selector = self._selector_for_node(container) if container else None
        title_selector = self._selector_for_node(candidate_anchor) if candidate_anchor else None
        url_selector = title_selector
        location_selector = self._find_structural_child_selector(container, ["location", "subhead", "city", "place", "where"])
        description_selector = self._find_structural_child_selector(container, ["description", "summary", "details", "about", "overview"])
        date_posted_text_selector = self._find_structural_child_selector(container, ["date", "posted", "opened", "published"])
        employment_type_selector = self._find_structural_child_selector(container, ["full time", "part time", "intern", "contract", "temporary", "freelance"])
        work_type_selector = self._find_structural_child_selector(container, ["remote", "hybrid", "onsite", "on site", "work from home"])
        salary_range_selector = self._find_structural_child_selector(container, ["salary", "compensation", "ctc", "pay", "range"])
        return {
            "items_selector": items_selector or ("main a[href]" if job_links else "a[href]"),
            "title_selector": title_selector or "a",
            "url_selector": url_selector or "a[href]",
            "location_selector": location_selector,
            "description_selector": description_selector,
            "date_posted_text_selector": date_posted_text_selector,
            "employment_type_selector": employment_type_selector,
            "work_type_selector": work_type_selector,
            "salary_range_selector": salary_range_selector,
            "json_items_path": self._dedupe_json_paths(self._extract_json_paths(signals.get("embedded_json") or [])),
        }

    def _anchor_score(self, anchor: Any, job_links: list[str]) -> int:
        if anchor is None:
            return -10**9
        href = str(anchor.get("href") or "").lower() if hasattr(anchor, "get") else ""
        text = " ".join(getattr(anchor, "get_text", lambda *args, **kwargs: "")(" ", strip=True).split()).lower()
        if not href or not text:
            return -10**9
        banned_text = {
            "learn more",
            "sign in",
            "sign up",
            "skip to content",
            "search jobs",
            "browse all",
            "view all",
            "privacy",
            "cookie",
            "terms",
        }
        score = 0
        if any(token in href for token in ["/career", "/jobs", "/job", "/vacanc", "position", "opening"]):
            score += 8
        if any(token in text for token in ["job", "career", "opening", "position", "vacanc", "apply"]):
            score += 6
        if job_links and any(token in href for token in ["/career", "/jobs", "/job", "/vacanc", "position", "opening"]):
            score += 4
        if text in banned_text:
            score -= 10
        if len(text) > 120:
            score -= 4
        if len(text) < 3:
            score -= 4
        return score

    def _build_detail_hints(self) -> dict[str, Any]:
        return {
            "url_path": ["url"],
            "description_path": ["description"],
        }

    def _extract_visible_jobs(self, html: str, listing: dict[str, Any]) -> dict[str, Any]:
        try:
            from bs4 import BeautifulSoup
        except Exception:
            return {"titles": [], "count": 0, "listing_selector": None, "job_selector": None, "detail_selector": None}

        soup = BeautifulSoup(html, "html.parser")
        items_selector = str(listing.get("items_selector") or "").strip()
        title_selector = str(listing.get("title_selector") or "").strip()
        location_selector = str(listing.get("location_selector") or "").strip()
        description_selector = str(listing.get("description_selector") or "").strip()
        if not items_selector:
            return {"titles": [], "count": 0, "listing_selector": None, "job_selector": None, "detail_selector": None}

        titles: list[str] = []
        for item in soup.select(items_selector):
            node = item.select_one(title_selector) if title_selector else item
            text = " ".join((node.get_text(" ", strip=True) if node else "").split())
            if text and text.lower() not in {"loading...", "search"} and text not in titles:
                titles.append(text)
        return {
            "titles": titles,
            "count": len(titles),
            "listing_selector": items_selector or None,
            "job_selector": items_selector or None,
            "detail_selector": description_selector or location_selector or None,
        }

    def _recommend_strategy(
        self,
        source_type: str,
        jobs_found: bool,
        api_url: str | None,
        listing_selector: str | None,
        job_count_detected: int,
    ) -> str:
        if jobs_found:
            return "beautifulsoup"
        if source_type == "api" and api_url:
            return "requests"
        if source_type == "spa" and job_count_detected == 0:
            return "playwright"
        return "beautifulsoup"

    def _build_selector_hints(self, listing: dict[str, Any], detail: dict[str, Any], pagination: dict[str, Any]) -> dict[str, Any]:
        return {
            "listing": {
                "items_selector": listing.get("items_selector"),
                "title_selector": listing.get("title_selector"),
                "url_selector": listing.get("url_selector"),
                "location_selector": listing.get("location_selector"),
                "description_selector": listing.get("description_selector"),
                "date_posted_text_selector": listing.get("date_posted_text_selector"),
                "employment_type_selector": listing.get("employment_type_selector"),
                "work_type_selector": listing.get("work_type_selector"),
                "salary_range_selector": listing.get("salary_range_selector"),
            },
            "detail": {
                "description_path": detail.get("description_path"),
            },
            "pagination": {
                "type": pagination.get("type"),
                "page_param": pagination.get("param"),
            },
        }

    def _selector_for_node(self, node: Any) -> str | None:
        if node is None:
            return None
        name = str(getattr(node, "name", "") or "").strip()
        if not name:
            return None
        selector = [name]
        if hasattr(node, "get"):
            node_id = str(node.get("id") or "").strip()
            if node_id:
                selector.append(f'[id="{self._css_attr_value(node_id)}"]')
            classes = [part.strip() for part in (node.get("class") or []) if str(part).strip()]
            for class_name in classes[:3]:
                selector.append(f'[class~="{self._css_attr_value(class_name)}"]')
        if len(selector) > 1:
            return "".join(selector)
        return name

    def _css_attr_value(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", " ")

    def _has_json_api_signal(self, snapshot_signals: Any, embedded_json: list[dict[str, Any]], api_candidates: list[dict[str, Any]]) -> bool:
        if embedded_json:
            return True
        if not isinstance(snapshot_signals, dict):
            return False
        for response in snapshot_signals.get("network_responses") or []:
            content_type = str(response.get("content_type") or "").lower()
            if "json" in content_type:
                return True
        for candidate in api_candidates:
            if candidate.get("kind") == "network" and candidate.get("url"):
                url = str(candidate.get("url") or "").lower()
                if any(token in url for token in ["/api/", "graphql", "json", "jobs", "career", "recruit"]):
                    return True
        return False

    def _find_structural_child_selector(self, node: Any, tokens: list[str]) -> str | None:
        if node is None or not getattr(node, "select", None):
            return None
        token_set = [token.lower() for token in tokens if token]
        for child in node.select("*"):
            text = " ".join(getattr(child, "get_text", lambda *args, **kwargs: "")(" ", strip=True).split()).lower()
            class_text = " ".join(str(part).lower() for part in (child.get("class") or [])) if hasattr(child, "get") else ""
            if any(token in text for token in token_set) or any(token in class_text for token in token_set):
                selector = self._selector_for_node(child)
                if selector:
                    return selector
        return None

    def _structured_evidence(self, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        structured: list[dict[str, Any]] = []
        for item in evidence:
            structured.append(
                {
                    "url": item.get("url"),
                    "status_code": item.get("status_code"),
                    "title": item.get("title"),
                    "embedded_json_kinds": item.get("embedded_json_kinds") or [],
                    "api_candidates": [
                        {k: candidate.get(k) for k in ["kind", "url", "token"] if candidate.get(k) is not None}
                        for candidate in (item.get("api_candidates") or [])[:3]
                    ],
                    "sample_links": (item.get("sample_links") or [])[:5],
                }
            )
        return structured

    def _merge_dicts(self, base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in extra.items():
            if value is None:
                continue
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _merge_missing_dicts(self, base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in extra.items():
            if value is None:
                continue
            if key not in merged or merged[key] in (None, "", [], {}):
                merged[key] = value
                continue
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_missing_dicts(merged[key], value)
        return merged

    def _best_container_for_anchor(self, anchor: Any) -> Any | None:
        if anchor is None:
            return None
        best = None
        best_score = -1
        ancestor = anchor.parent
        while ancestor and getattr(ancestor, "name", None) not in {"html", "body"}:
            score = self._container_score(ancestor, anchor)
            if score > best_score:
                best_score = score
                best = ancestor
            ancestor = ancestor.parent
        return best or anchor.parent

    def _container_score(self, node: Any, anchor: Any) -> int:
        class_text = " ".join(str(part).lower() for part in (node.get("class") or [])) if hasattr(node, "get") else ""
        id_text = str(node.get("id") or "").lower() if hasattr(node, "get") else ""
        text = f"{class_text} {id_text}"
        matched_token = False
        for token, token_score in (
            ("job", 4),
            ("jobs", 4),
            ("career", 3),
            ("opening", 3),
            ("position", 3),
            ("vacanc", 3),
            ("listing", 3),
            ("list", 2),
            ("card", 2),
            ("item", 2),
            ("result", 2),
        ):
            if token in text:
                matched_token = True
        if not matched_token:
            return 0

        score = 0
        for token, token_score in (
            ("job", 4),
            ("jobs", 4),
            ("career", 3),
            ("opening", 3),
            ("position", 3),
            ("vacanc", 3),
            ("listing", 3),
            ("list", 2),
            ("card", 2),
            ("item", 2),
            ("result", 2),
        ):
            if token in text:
                score += token_score
        for token in ("left", "right", "top", "bottom", "header", "footer", "sidebar", "aside", "nav", "menu"):
            if token in text:
                score -= 5
        for token in ("joblist", "job-card", "jobcard", "job-item", "jobitem", "listing-item", "result-item"):
            if token in text:
                score += 4
        if getattr(node, "select", None):
            anchors = node.select("a[href]")
            anchor_count = len(anchors)
            if anchor_count >= 1:
                score += 12 if anchor_count == 1 else max(1, 6 - anchor_count)
            if anchor_count > 1:
                score -= min(anchor_count, 4)
        if anchor and getattr(node, "get_text", None):
            anchor_text = " ".join(anchor.get_text(" ", strip=True).split()).lower()
            node_text = " ".join(node.get_text(" ", strip=True).split()).lower()
            if anchor_text and anchor_text in node_text:
                score += 2
        return score

    def _extract_title(self, html: str) -> str | None:
        lower = html.lower()
        start = lower.find("<title>")
        end = lower.find("</title>")
        if start == -1 or end == -1 or end <= start:
            return None
        return html[start + 7 : end].strip()
