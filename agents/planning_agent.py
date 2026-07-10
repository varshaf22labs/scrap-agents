from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.context import AgentContext
from llm_client import LLMClient, parse_json_object
from models import ScrapePlan


class PlanningAgent:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client

    def run(self, context: AgentContext) -> ScrapePlan:
        if self.llm_client and context.investigation:
            plan = self._plan_with_llm(context)
        else:
            raise ValueError("Planning requires an LLM client and investigation findings")
        context.plan = plan
        return plan

    def _plan_with_llm(self, context: AgentContext) -> ScrapePlan:
        payload = {
            "domain": context.domain,
            "investigation": context.investigation.planning_payload() if context.investigation else None,
        }
        prompt = (
            "Create a strict JSON scrape plan for a job scraper script.\n"
            "The plan must support one of these strategies: api, json, html, spa.\n"
            "Use the investigation payload as the source of truth. Do not infer new selectors or URLs unless the payload is empty.\n"
            "Use structural extraction only. No regex. Missing fields should be null.\n"
            "Return keys: strategy, entry_url, output_path, pagination, listing, detail, filters, notes.\n"
            "Use JSON paths as arrays of keys/indices. Use CSS selectors for HTML extraction.\n"
            f"Context: {json.dumps(payload)[:12000]}"
        )
        response = self.llm_client.chat(
            [
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        data = parse_json_object(response)
        return self._to_plan(context, data)

    def _to_plan(self, context: AgentContext, data: dict[str, Any]) -> ScrapePlan:
        output_path = self._normalize_output_path(
            data.get("output_path") or str(context.generated_dir / f"{context.normalized_domain.replace('.', '_')}.jsonl")
        )
        investigation_source = context.investigation.source_type if context.investigation else None
        strategy = str(data.get("strategy") or "html")
        investigation = context.investigation or None
        if investigation:
            if investigation_source in {"api", "json"} and investigation.api_url:
                strategy = "api"
            elif investigation_source == "spa":
                strategy = "spa"
        if investigation_source in {"api", "json"} and strategy == "html" and investigation and investigation.api_url:
            strategy = "api"
        investigation_selectors = (investigation.selectors if investigation else {}) or {}
        investigation_listing = (investigation.listing if investigation else {}) or {}
        investigation_detail = (investigation.detail if investigation else {}) or {}
        investigation_pagination = (investigation.pagination if investigation else {}) or {}
        browser_required = bool(
            data.get("browser_required")
            if data.get("browser_required") is not None
            else (investigation and investigation_source == "spa")
        )
        if investigation_source == "spa":
            browser_required = True

        listing = self._merge_sections(
            self._coerce_section(data.get("listing")),
            investigation_listing,
            investigation_selectors.get("listing") if isinstance(investigation_selectors, dict) else {},
        )
        detail = self._merge_sections(
            self._coerce_section(data.get("detail")),
            investigation_detail,
            investigation_selectors.get("detail") if isinstance(investigation_selectors, dict) else {},
        )
        pagination = self._merge_sections(
            self._coerce_section(data.get("pagination")),
            investigation_pagination,
            investigation_selectors.get("pagination") if isinstance(investigation_selectors, dict) else {},
        )
        if investigation:
            root_listing_selector = self._coerce_value("items_selector", investigation.listing_selector or investigation.job_selector)
            if root_listing_selector and not listing.get("items_selector"):
                listing["items_selector"] = root_listing_selector
            if investigation.job_selector and not listing.get("job_selector"):
                listing["job_selector"] = self._coerce_value("job_selector", investigation.job_selector)
            if investigation.listing_selector and not listing.get("listing_selector"):
                listing["listing_selector"] = self._coerce_value("listing_selector", investigation.listing_selector)
            if investigation.detail_selector and not detail.get("description_selector"):
                detail["description_selector"] = self._coerce_value("description_selector", investigation.detail_selector)
            if investigation.job_selector and not listing.get("title_selector"):
                listing["title_selector"] = self._coerce_value("title_selector", investigation.job_selector)
            if investigation.job_selector and not listing.get("url_selector"):
                listing["url_selector"] = self._coerce_value("url_selector", investigation.job_selector)
        has_json_paths = bool(
            (investigation and investigation.json_paths)
            or self._coerce_value("json_items_path", listing.get("json_items_path") if isinstance(listing, dict) else None)
        )
        has_structural_listing = bool(
            (listing.get("items_selector") if isinstance(listing, dict) else None)
            or (investigation and (investigation.listing_selector or investigation.job_selector))
        )
        if strategy in {"api", "json"} and not has_json_paths and has_structural_listing:
            strategy = "spa"
            browser_required = True
        if strategy == "spa" and not pagination:
            pagination = {"type": "single"}
        return ScrapePlan(
            strategy=str(strategy),
            entry_url=self._normalize_url_value(data.get("entry_url") or f"https://{context.domain}"),
            output_path=str(output_path),
            browser_required=browser_required,
            pagination=pagination,
            listing=listing,
            detail=detail,
            filters=dict(data.get("filters") or {"country": "India", "country_code": "IN"}),
            notes=str(data.get("notes") or ""),
        )

    def _normalize_output_path(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if text.startswith("investigation."):
            return text
        path = Path(text)
        if path.suffix.lower() != ".jsonl":
            path = path.with_suffix(".jsonl")
        return str(path)

    def _normalize_url_value(self, value: Any) -> str:
        if isinstance(value, list):
            for item in value:
                normalized = self._normalize_url_value(item)
                if normalized:
                    return normalized
            return ""
        if isinstance(value, dict):
            for key in ("url", "href", "value", "text"):
                if key in value:
                    normalized = self._normalize_url_value(value[key])
                    if normalized:
                        return normalized
            return ""
        text = str(value or "").strip()
        if not text:
            return ""
        if text.startswith("http://") or text.startswith("https://"):
            return text
        return text

    def _coerce_section(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        return {
            key: self._coerce_value(key, section_value)
            for key, section_value in value.items()
            if section_value is not None
        }

    def _merge_sections(self, *sections: Any) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for section in sections:
            if not isinstance(section, dict):
                continue
            for key, value in section.items():
                if value is None:
                    continue
                merged[key] = self._coerce_value(key, value)
        return merged

    def _coerce_value(self, key: str, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                sub_key: self._coerce_value(sub_key, sub_value)
                for sub_key, sub_value in value.items()
                if sub_value is not None
            }
        if isinstance(value, list):
            if not value:
                return [] if key.endswith("path") or key.endswith("_path") else None
            if key.endswith("selector") or key.endswith("_selector"):
                if len(value) == 1:
                    return self._coerce_value(key, value[0])
                return " ".join(str(item).strip() for item in value if str(item).strip()) or None
            if value[0] == "investigation":
                return [] if key.endswith("path") or key.endswith("_path") else None
            return [self._coerce_value(key, item) for item in value]
        if isinstance(value, str):
            text = value.strip()
            if not text or text.startswith("investigation."):
                return None
            return text
        return value
