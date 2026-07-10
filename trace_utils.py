from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models import to_jsonable


@dataclass
class TraceEvent:
    ts: str
    actor: str
    action: str
    input: Any
    output: Any
    note: str | None = None


@dataclass
class TraceRecorder:
    domain: str
    trace_dir: Path
    events: list[TraceEvent] = field(default_factory=list)

    def record(self, actor: str, action: str, input_data: Any, output_data: Any, note: str | None = None) -> None:
        self.events.append(
            TraceEvent(
                ts=datetime.now(timezone.utc).isoformat(),
                actor=actor,
                action=action,
                input=to_jsonable(input_data),
                output=to_jsonable(output_data),
                note=note,
            )
        )

    def save(self) -> Path:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"{self.domain.replace('/', '_')}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        payload = {
            "domain": self.domain,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "events": [to_jsonable(event) for event in self.events],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def snapshot(self, **extra: Any) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "event_count": len(self.events),
            **{k: to_jsonable(v) for k, v in extra.items()},
        }
