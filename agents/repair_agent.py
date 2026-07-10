from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.context import AgentContext
from llm_client import LLMClient, parse_json_object
from models import ScrapePlan


class RepairAgent:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client

    def run(
        self,
        context: AgentContext,
        issues: list[dict[str, Any]],
        execution_result: Any | None = None,
        generated_script: str | None = None,
        verification_report: dict[str, Any] | None = None,
    ) -> ScrapePlan | None:
        if not self.llm_client:
            raise ValueError("Repair requires an LLM client")
        traceback_text = ""
        stdout_text = ""
        if execution_result is not None:
            traceback_text = str(getattr(execution_result, "stderr", "") or "")
            stdout_text = str(getattr(execution_result, "stdout", "") or "")

        prompt = (
            "The generated job scraper failed execution or validation. Produce a corrected strict JSON scrape plan.\n"
            "Use the original investigation output, the current plan, the generated scraper, the traceback, and the validation report to make targeted fixes.\n"
            "Do not guess new selectors. Modify only the failing parts.\n"
            "Keep the output schema the same. No regex. Structural extraction only.\n"
            f"Domain: {context.domain}\n"
            f"Investigation: {json.dumps(context.investigation.planning_payload() if context.investigation else None)[:12000]}\n"
            f"Current plan: {json.dumps(context.plan.__dict__ if context.plan else None)[:12000]}\n"
            f"Generated scraper: {generated_script[:20000] if generated_script else ''}\n"
            f"Execution stdout: {stdout_text[:12000]}\n"
            f"Execution traceback: {traceback_text[:12000]}\n"
            f"Verification report: {json.dumps(verification_report)[:12000] if verification_report else ''}\n"
            f"Issues: {json.dumps(issues)[:12000]}\n"
        )
        response = self.llm_client.chat(
            [
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        data = parse_json_object(response)
        if not isinstance(data, dict):
            return None
        plan = ScrapePlan(
            strategy=str(data.get("strategy") or (context.plan.strategy if context.plan else "html")),
            entry_url=self._normalize_url_value(data.get("entry_url") or (context.plan.entry_url if context.plan else f"https://{context.domain}")),
            output_path=self._normalize_output_path(data.get("output_path") or (context.plan.output_path if context.plan else context.generated_dir / "jobs.jsonl")),
            browser_required=bool(data.get("browser_required") if data.get("browser_required") is not None else (context.plan.browser_required if context.plan else False)),
            pagination=dict(data.get("pagination") or (context.plan.pagination if context.plan else {})),
            listing=dict(data.get("listing") or (context.plan.listing if context.plan else {})),
            detail=dict(data.get("detail") or (context.plan.detail if context.plan else {})),
            filters=dict(data.get("filters") or (context.plan.filters if context.plan else {})),
            notes=str(data.get("notes") or ""),
        )

        if context.investigation:
            if context.investigation.careers_url:
                plan.entry_url = context.investigation.careers_url
            if context.investigation.listing:
                plan.listing = self._merge_section(plan.listing, context.investigation.listing)
            if context.investigation.selectors.get("listing") if context.investigation.selectors else None:
                plan.listing = self._merge_section(plan.listing, context.investigation.selectors.get("listing") or {})
            if context.investigation.detail:
                plan.detail = self._merge_section(plan.detail, context.investigation.detail)
            if context.investigation.selectors.get("detail") if context.investigation.selectors else None:
                plan.detail = self._merge_section(plan.detail, context.investigation.selectors.get("detail") or {})
            if context.investigation.pagination:
                plan.pagination = self._merge_section(plan.pagination, context.investigation.pagination)
            if context.investigation.selectors.get("pagination") if context.investigation.selectors else None:
                plan.pagination = self._merge_section(plan.pagination, context.investigation.selectors.get("pagination") or {})
            if context.investigation.listing_selector and not plan.listing.get("items_selector"):
                plan.listing["items_selector"] = context.investigation.listing_selector
            if context.investigation.job_selector and not plan.listing.get("job_selector"):
                plan.listing["job_selector"] = context.investigation.job_selector
            if context.investigation.job_selector and not plan.listing.get("title_selector"):
                plan.listing["title_selector"] = context.investigation.job_selector
            if context.investigation.job_selector and not plan.listing.get("url_selector"):
                plan.listing["url_selector"] = context.investigation.job_selector
            if context.investigation.detail_selector and not plan.detail.get("description_selector"):
                plan.detail["description_selector"] = context.investigation.detail_selector
            if self._has_structural_listing(plan) and not self._has_json_items(plan):
                plan.strategy = "spa"
                plan.browser_required = True
            elif context.investigation.source_type == "spa":
                plan.strategy = "spa"
                plan.browser_required = True
            plan.notes = (plan.notes + " [repair preserved investigation selectors]").strip()

        context.plan = plan
        return plan

    def _merge_section(self, base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = dict(base or {})
        for key, value in (overlay or {}).items():
            if value is None:
                continue
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_section(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _normalize_output_path(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
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

    def _has_json_items(self, plan: ScrapePlan) -> bool:
        listing = plan.listing or {}
        for key in ("json_items_path", "items_path", "content_path"):
            value = listing.get(key)
            if isinstance(value, list) and value:
                return True
            if isinstance(value, dict) and value:
                return True
        return False

    def _has_structural_listing(self, plan: ScrapePlan) -> bool:
        listing = plan.listing or {}
        for key in ("items_selector", "job_selector", "listing_selector"):
            value = listing.get(key)
            if isinstance(value, str) and value.strip():
                return True
            if isinstance(value, list) and any(str(item).strip() for item in value):
                return True
        return False
