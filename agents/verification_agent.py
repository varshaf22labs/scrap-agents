from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.context import AgentContext
from models import ValidationIssue, ValidationReport
from tools.validator import validate_jsonl_jobs


class VerificationAgent:
    def run(self, context: AgentContext, execution_result: Any) -> ValidationReport:
        if not context.output_path:
            return ValidationReport(
                ok=False,
                output_path="",
                job_count=0,
                india_job_count=0,
                issues=[],
            )
        if getattr(execution_result, "exit_code", 1) != 0:
            return ValidationReport(
                ok=False,
                output_path=str(context.output_path),
                job_count=0,
                india_job_count=0,
                issues=[
                    ValidationIssue(
                        code="execution_failed",
                        message=f"Generated scraper exited with code {getattr(execution_result, 'exit_code', None)}.",
                    )
                ],
            )
        report = validate_jsonl_jobs(context.output_path)
        investigation = context.investigation
        if investigation and investigation.jobs_found_during_investigation:
            if report.job_count == 0:
                report.issues.append(
                    ValidationIssue(
                        code="investigation_mismatch",
                        message=(
                            "Investigation already observed jobs, but the generated scraper returned 0 jobs."
                        ),
                    )
                )
                report.ok = False
            elif investigation.job_count_detected and report.job_count < investigation.job_count_detected:
                report.issues.append(
                    ValidationIssue(
                        code="undercounted_jobs",
                        message=(
                            f"Investigation detected {investigation.job_count_detected} jobs, but the scraper returned {report.job_count}."
                        ),
                    )
                )
                report.ok = False
        context.verification = {
            "execution_exit_code": getattr(execution_result, "exit_code", None),
            "stdout": getattr(execution_result, "stdout", ""),
            "stderr": getattr(execution_result, "stderr", ""),
            "duration_seconds": getattr(execution_result, "duration_seconds", None),
            "validation": {
                "ok": report.ok,
                "job_count": report.job_count,
                "india_job_count": report.india_job_count,
                "issues": [issue.__dict__ for issue in report.issues],
            },
            "investigation": {
                "jobs_found_during_investigation": investigation.jobs_found_during_investigation if investigation else None,
                "job_count_detected": investigation.job_count_detected if investigation else None,
                "job_titles": investigation.job_titles if investigation else [],
                "recommended_strategy": investigation.recommended_strategy if investigation else None,
            },
        }
        return report
