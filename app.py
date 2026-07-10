from __future__ import annotations

import argparse
import json
import sys

from agents.code_generation_agent import CodeGenerationAgent
from agents.context import AgentContext
from agents.discovery_agent import DiscoveryAgent
from agents.execution_agent import ExecutionAgent
from agents.investigation_agent import InvestigationAgent
from agents.planning_agent import PlanningAgent
from agents.repair_agent import RepairAgent
from agents.trace_agent import TraceAgent
from agents.verification_agent import VerificationAgent
from config import ensure_directories, load_config
from llm_client import LLMClient
from models import ValidationReport
from site_utils import normalize_domain
from tools.firecrawl import FirecrawlTool
from tools.search import SearchTool


def build_llm_client(profile, trace=None):
    if not profile.base_url:
        return None
    return LLMClient(
        provider=profile.provider,
        base_url=profile.base_url,
        api_key=profile.api_key,
        model=profile.model,
        reasoning_effort=profile.reasoning_effort,
        timeout_seconds=30,
        trace_recorder=trace.recorder if trace else None,
    )


def build_search_tool(config, trace=None):
    if not config.search_provider:
        return None
    api_key = config.serper_api_key if config.search_provider == "serper" else config.serpapi_api_key
    if not api_key:
        return None
    return SearchTool(
        provider=config.search_provider,
        api_key=api_key,
        timeout_seconds=config.request_timeout_seconds,
        trace_recorder=trace.recorder if trace else None,
    )


def build_firecrawl_tool(config, trace=None):
    if not config.firecrawl_api_key:
        return None
    return FirecrawlTool(
        api_key=config.firecrawl_api_key,
        timeout_seconds=config.request_timeout_seconds,
        trace_recorder=trace.recorder if trace else None,
    )


def run_domain(domain: str) -> dict:
    config = load_config()
    ensure_directories(config)
    normalized_domain = normalize_domain(domain)
    def log_stage(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    errors: list[dict[str, str]] = []

    def record_error(stage: str, exc: Exception) -> None:
        error = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
        errors.append(error)
        trace.record(
            stage,
            "error",
            {"domain": domain},
            error,
            note=f"{stage} stage failed",
        )

    log_stage(f"[startup] loading configuration for {domain}")
    trace = TraceAgent(normalized_domain, config.traces_dir)
    trace.llm_client = build_llm_client(config.trace_llm, trace=trace)
    context = AgentContext(
        domain=domain,
        normalized_domain=normalized_domain,
        root_dir=config.root_dir,
        generated_dir=config.generated_dir,
        traces_dir=config.traces_dir,
        memory_dir=config.memory_dir,
    )

    discovery_llm = build_llm_client(config.discovery_llm, trace=trace)
    investigation_llm = build_llm_client(config.investigation_llm, trace=trace)
    planning_llm = build_llm_client(config.planning_llm, trace=trace)
    codegen_llm = build_llm_client(config.codegen_llm, trace=trace)
    repair_llm = build_llm_client(config.repair_llm, trace=trace)
    search_tool = build_search_tool(config, trace=trace)
    firecrawl_tool = build_firecrawl_tool(config, trace=trace)

    trace.record(
        "system",
        "startup",
        {"domain": domain},
        {
            "env_file_loaded": config.env_file_loaded,
            "discovery_model": config.discovery_llm.model,
            "investigation_model": config.investigation_llm.model,
            "planning_model": config.planning_llm.model,
            "codegen_model": config.codegen_llm.model,
            "repair_model": config.repair_llm.model,
            "trace_model": config.trace_llm.model,
            "search_provider": config.search_provider,
            "firecrawl_enabled": bool(config.firecrawl_api_key),
        },
        note="Resolved runtime configuration",
    )

    discovery = DiscoveryAgent(search_tool=search_tool, firecrawl_tool=firecrawl_tool, timeout_seconds=config.request_timeout_seconds)
    investigation = InvestigationAgent(llm_client=investigation_llm, timeout_seconds=config.request_timeout_seconds)
    planning = PlanningAgent(llm_client=planning_llm)
    codegen = CodeGenerationAgent(llm_client=codegen_llm)
    execution = ExecutionAgent(timeout_seconds=config.request_timeout_seconds * 6)
    verification = VerificationAgent()
    repair = RepairAgent(llm_client=repair_llm)

    log_stage("[discovery] finding candidate careers pages")
    try:
        candidates = discovery.run(context)
    except Exception as exc:
        record_error("discovery", exc)
        trace.save()
        raise
    else:
        trace.record("discovery", "candidate_pages", {"domain": domain}, [page.__dict__ for page in candidates], note="Candidate careers URLs and probes")
    log_stage(f"[discovery] found {len(candidates)} candidate pages")

    log_stage("[investigation] analyzing source type and page evidence")
    try:
        finding = investigation.run(context)
    except Exception as exc:
        record_error("investigation", exc)
        trace.save()
        raise
    else:
        investigation_note = (
            f"Detected source type {finding.source_type}; "
            f"jobs_found={finding.jobs_found_during_investigation}; "
            f"job_count={finding.job_count_detected}; "
            f"recommended_strategy={finding.recommended_strategy}"
        )
        trace.record("investigation", "source_analysis", {"domain": domain}, finding.__dict__, note=investigation_note)
    log_stage(f"[investigation] source type: {finding.source_type}")

    log_stage("[planning] generating scrape plan")
    try:
        plan = planning.run(context)
    except Exception as exc:
        record_error("planning", exc)
        trace.save()
        raise
    else:
        plan_note = (
            f"Planned strategy={plan.strategy}; browser_required={plan.browser_required}; "
            f"jobs_found={finding.jobs_found_during_investigation}"
        )
        trace.record("planning", "scrape_plan", {"domain": domain}, plan.__dict__, note=plan_note)
    log_stage(f"[planning] strategy: {plan.strategy}")

    log_stage("[code_generation] rendering scraper script")
    try:
        script_path = codegen.run(context)
    except Exception as exc:
        record_error("code_generation", exc)
        trace.save()
        raise
    else:
        codegen_note = "Generated BeautifulSoup scraper" if plan.strategy in {"html", "spa"} and not plan.browser_required else "Generated browser-capable scraper"
        trace.record("code_generation", "render_script", {"plan": plan.__dict__}, {"script_path": str(script_path)}, note=codegen_note)
    log_stage(f"[code_generation] wrote {script_path}")
    generated_script_text = script_path.read_text(encoding="utf-8") if script_path.exists() else ""

    log_stage("[execution] running generated scraper")
    result = None
    try:
        result = execution.run(context)
    except Exception as exc:
        record_error("execution", exc)
    else:
        trace.record(
            "execution",
            "run_script",
            {"script_path": str(script_path), "output_path": context.output_path},
            result.__dict__,
            note="Executed generated scraper",
        )
        log_stage(f"[execution] exit code {result.exit_code}")

    log_stage("[verification] validating scraper output")
    if result is not None:
        try:
            report = verification.run(context, result)
        except Exception as exc:
            record_error("verification", exc)
            report = ValidationReport(
                ok=False,
                output_path=str(context.output_path or ""),
                job_count=0,
                india_job_count=0,
                issues=[],
            )
        else:
            trace.record(
                "verification",
                "validate_output",
                {"output_path": context.output_path},
                {
                    "ok": report.ok,
                    "job_count": report.job_count,
                    "india_job_count": report.india_job_count,
                    "issues": [issue.__dict__ for issue in report.issues],
                },
                note="Validated output JSONL",
            )
            log_stage(f"[verification] ok={report.ok}, jobs={report.job_count}, india_jobs={report.india_job_count}")
    else:
        report = ValidationReport(
            ok=False,
            output_path=str(context.output_path or ""),
            job_count=0,
            india_job_count=0,
            issues=[],
        )

    attempts = 0
    while not report.ok and attempts < 2 and repair_llm is not None and result is not None:
        attempts += 1
        issue_payload = [issue.__dict__ for issue in report.issues]
        log_stage(f"[repair] attempt {attempts}: fixing validation issues")
        try:
            new_plan = repair.run(
                context,
                issue_payload,
                execution_result=result,
                generated_script=generated_script_text,
                verification_report=context.verification,
            )
        except Exception as exc:
            record_error("repair", exc)
            break
        if not new_plan:
            break
        trace.record("repair", "revise_plan", {"issues": issue_payload}, new_plan.__dict__, note="Repair iteration")
        log_stage("[code_generation] re-rendering repaired scraper")
        script_path = codegen.run(context)
        generated_script_text = script_path.read_text(encoding="utf-8") if script_path.exists() else ""
        trace.record("code_generation", "render_script", {"plan": new_plan.__dict__}, {"script_path": str(script_path)}, note="Re-rendered after repair")
        log_stage("[execution] re-running repaired scraper")
        result = execution.run(context)
        trace.record(
            "execution",
            "run_script",
            {"script_path": str(script_path), "output_path": context.output_path},
            result.__dict__,
            note="Re-executed repaired scraper",
        )
        report = verification.run(context, result)
        trace.record(
            "verification",
            "validate_output",
            {"output_path": context.output_path},
            {
                "ok": report.ok,
                "job_count": report.job_count,
                "india_job_count": report.india_job_count,
                "issues": [issue.__dict__ for issue in report.issues],
            },
            note="Validated repaired output",
        )
        log_stage(f"[verification] repair ok={report.ok}, jobs={report.job_count}, india_jobs={report.india_job_count}")

    trace_path = trace.save()
    log_stage(f"[done] trace saved to {trace_path}")
    summary = {
        "ok": report.ok and not errors,
        "domain": domain,
        "normalized_domain": normalized_domain,
        "script_path": str(script_path),
        "output_path": str(context.output_path) if context.output_path else None,
        "trace_path": str(trace_path),
        "verification": context.verification,
        "errors": errors,
        "candidate_count": len(candidates),
        "env_file_loaded": config.env_file_loaded,
        "discovery_model": config.discovery_llm.model,
        "investigation_model": config.investigation_llm.model,
        "planning_model": config.planning_llm.model,
        "codegen_model": config.codegen_llm.model,
        "repair_model": config.repair_llm.model,
        "trace_model": config.trace_llm.model,
        "search_provider": config.search_provider,
        "firecrawl_enabled": bool(config.firecrawl_api_key),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a standalone India job scraper from only a domain.")
    parser.add_argument("domain", help="Company domain, e.g. swissre.com")
    parser.add_argument("--json", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--plain", action="store_true", help="Print a human-readable summary instead of JSON")
    args = parser.parse_args()
    try:
        summary = run_domain(args.domain)
    except Exception as exc:
        summary = {
            "ok": False,
            "domain": args.domain,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        exit_code = 1
    else:
        exit_code = 0 if summary.get("ok", False) else 1

    if args.plain:
        print(f"ok: {summary.get('ok', False)}")
        print(f"domain: {summary.get('domain', args.domain)}")
        print(f"script_path: {summary.get('script_path')}")
        print(f"output_path: {summary.get('output_path')}")
        print(f"trace_path: {summary.get('trace_path')}")
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
