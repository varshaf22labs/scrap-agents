from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models import ValidationIssue, ValidationReport


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
    return rows


def _location_is_india(location: Any) -> bool:
    if not isinstance(location, dict):
        return False
    country = str(location.get("country") or "").strip().lower()
    country_code = str(location.get("country_code") or "").strip().upper()
    return country in {"india", "in"} or country_code == "IN"


def validate_jsonl_jobs(path: str | Path, min_jobs: int = 1) -> ValidationReport:
    output_path = Path(path)
    if not output_path.exists():
        return ValidationReport(
            ok=False,
            output_path=str(output_path),
            job_count=0,
            india_job_count=0,
            issues=[ValidationIssue(code="missing_output", message="Output file does not exist")],
        )

    rows = _load_jsonl(output_path)
    issues: list[ValidationIssue] = []
    samples = rows[:3]

    required_keys = {
        "title",
        "job_id",
        "location",
        "url",
        "apply_url",
        "date_posted",
        "date_posted_text",
        "job_description",
        "employment_type",
        "work_type",
        "salary_range",
    }
    for index, row in enumerate(rows, start=1):
        missing = sorted(required_keys - set(row.keys()))
        if missing:
            issues.append(
                ValidationIssue(
                    code="missing_keys",
                    message=f"Row {index} is missing required keys: {', '.join(missing)}",
                    sample=row,
                )
            )
        location = row.get("location")
        if not isinstance(location, dict):
            issues.append(
                ValidationIssue(
                    code="location_shape",
                    message=f"Row {index} location must be an object",
                    sample=row,
                )
            )
        else:
            for key in ["city", "state", "country", "country_code"]:
                if key not in location:
                    issues.append(
                        ValidationIssue(
                            code="location_missing_key",
                            message=f"Row {index} location missing {key}",
                            sample=row,
                        )
                    )
        if location is not None and not _location_is_india(location):
            issues.append(
                ValidationIssue(
                    code="non_india_job",
                    message=f"Row {index} is not marked as India",
                    severity="warning",
                    sample=row,
                )
            )

    india_count = sum(1 for row in rows if _location_is_india(row.get("location")))
    if len(rows) < min_jobs:
        issues.append(
            ValidationIssue(
                code="too_few_jobs",
                message=f"Expected at least {min_jobs} jobs, found {len(rows)}",
            )
        )
    if india_count == 0:
        issues.append(ValidationIssue(code="no_india_jobs", message="No India jobs were found"))

    ok = not any(issue.severity == "error" for issue in issues)
    return ValidationReport(
        ok=ok and len(rows) >= min_jobs and india_count > 0,
        output_path=str(output_path),
        job_count=len(rows),
        india_job_count=india_count,
        issues=issues,
        samples=samples,
    )

