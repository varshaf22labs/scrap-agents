from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from models import CandidatePage, InvestigationFinding, ScrapePlan


@dataclass
class AgentContext:
    domain: str
    normalized_domain: str
    root_dir: Path
    generated_dir: Path
    traces_dir: Path
    memory_dir: Path
    candidate_pages: list[CandidatePage] = field(default_factory=list)
    raw_artifacts: list[dict[str, Any]] = field(default_factory=list)
    investigation: InvestigationFinding | None = None
    plan: ScrapePlan | None = None
    generated_script_path: Path | None = None
    output_path: Path | None = None
    verification: dict[str, Any] = field(default_factory=dict)

