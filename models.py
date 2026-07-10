from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class CandidatePage:
    url: str
    source: str
    status_code: int | None = None
    final_url: str | None = None
    title: str | None = None
    snippet: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class InvestigationFinding:
    source_type: str
    careers_url: str | None
    platform: str | None
    api_url: str | None = None
    jobs_found_during_investigation: bool = False
    job_count_detected: int = 0
    job_titles: list[str] = field(default_factory=list)
    listing_selector: str | None = None
    job_selector: str | None = None
    detail_selector: str | None = None
    recommended_strategy: str | None = None
    json_paths: list[list[Any]] = field(default_factory=list)
    selectors: dict[str, Any] = field(default_factory=dict)
    pagination: dict[str, Any] = field(default_factory=dict)
    listing: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def planning_payload(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "careers_url": self.careers_url,
            "platform": self.platform,
            "api_url": self.api_url,
            "jobs_found_during_investigation": self.jobs_found_during_investigation,
            "job_count_detected": self.job_count_detected,
            "job_titles": self.job_titles,
            "listing_selector": self.listing_selector,
            "job_selector": self.job_selector,
            "detail_selector": self.detail_selector,
            "recommended_strategy": self.recommended_strategy,
            "json_paths": self.json_paths,
            "selectors": self.selectors,
            "pagination": self.pagination,
            "listing": self.listing,
            "detail": self.detail,
        }


@dataclass
class ScrapePlan:
    strategy: str
    entry_url: str
    output_path: str
    browser_required: bool = False
    pagination: dict[str, Any] = field(default_factory=dict)
    listing: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class ExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    output_path: str | None
    duration_seconds: float


@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: str = "error"
    sample: dict[str, Any] | None = None


@dataclass
class ValidationReport:
    ok: bool
    output_path: str
    job_count: int
    india_job_count: int
    issues: list[ValidationIssue] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value
