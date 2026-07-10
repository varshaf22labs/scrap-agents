from __future__ import annotations

import json
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agents.code_generation_agent import CodeGenerationAgent
from agents.investigation_agent import InvestigationAgent
from agents.planning_agent import PlanningAgent
from agents.verification_agent import VerificationAgent
from agents.trace_agent import TraceAgent
from config import _llm_profile
from models import CandidatePage, InvestigationFinding, ScrapePlan
from llm_client import LLMClient
from tools.validator import validate_jsonl_jobs
from trace_utils import TraceEvent
from trace_utils import TraceRecorder


class ValidatorTests(unittest.TestCase):
    def test_validate_jsonl_jobs_accepts_india_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "jobs.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "title": "Engineer",
                        "job_id": "123",
                        "location": {"city": "Pune", "state": "MH", "country": "India", "country_code": "IN"},
                        "url": "https://example.com/job/123",
                        "apply_url": "https://example.com/job/123",
                        "date_posted": None,
                        "date_posted_text": None,
                        "job_description": "Build things",
                        "employment_type": None,
                        "work_type": None,
                        "salary_range": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = validate_jsonl_jobs(output)

            self.assertTrue(report.ok)
            self.assertEqual(report.job_count, 1)
            self.assertEqual(report.india_job_count, 1)

    def test_validate_jsonl_jobs_flags_missing_country(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "jobs.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "title": "Engineer",
                        "job_id": "123",
                        "location": {"city": "Pune", "state": "MH", "country": None, "country_code": None},
                        "url": "https://example.com/job/123",
                        "apply_url": "https://example.com/job/123",
                        "date_posted": None,
                        "date_posted_text": None,
                        "job_description": "Build things",
                        "employment_type": None,
                        "work_type": None,
                        "salary_range": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = validate_jsonl_jobs(output)

            self.assertFalse(report.ok)
            self.assertTrue(any(issue.code == "no_india_jobs" for issue in report.issues))


class CodeGenerationTests(unittest.TestCase):
    def test_codegen_does_not_emit_regex(self) -> None:
        plan = ScrapePlan(
            strategy="html",
            entry_url="https://example.com/careers",
            output_path="generated/example.com/jobs.jsonl",
            pagination={"type": "page"},
            listing={"items_selector": ".job-card", "title_selector": "a", "url_selector": "a"},
            detail={"description_selector": ".description"},
            filters={"country": "India", "country_code": "IN"},
        )

        script = CodeGenerationAgent()._render_script(plan)

        self.assertNotIn("re.", script)
        self.assertNotIn("regex", script.lower())
        self.assertIn("BeautifulSoup", script)
        compile(script, "<generated>", "exec")
        self.assertFalse(script.startswith("    "))

    def test_codegen_run_does_not_call_llm_review(self) -> None:
        plan = ScrapePlan(
            strategy="html",
            entry_url="https://example.com/careers",
            output_path="generated/example.com/jobs.jsonl",
            pagination={"type": "page"},
            listing={"items_selector": ".job-card", "title_selector": "a", "url_selector": "a"},
            detail={"description_selector": ".description"},
            filters={"country": "India", "country_code": "IN"},
        )
        llm = Mock()
        agent = CodeGenerationAgent(llm_client=llm)
        with tempfile.TemporaryDirectory() as temp_dir:
            context = type(
                "Context",
                (),
                {
                    "plan": plan,
                    "generated_dir": Path(temp_dir),
                    "normalized_domain": "example.com",
                },
            )()
            script_path = agent.run(context)
            self.assertTrue(script_path.exists())
        self.assertFalse(llm.chat.called)

    def test_spa_codegen_compiles(self) -> None:
        plan = ScrapePlan(
            strategy="spa",
            entry_url="https://example.com/careers",
            output_path="generated/example.com/jobs.jsonl",
            pagination={"type": "page"},
            listing={"items_selector": [".job-card"], "title_selector": ["a"], "url_selector": ["a"]},
            detail={"description_selector": ".description"},
            filters={"country": "India", "country_code": "IN"},
        )

        script = CodeGenerationAgent()._render_script(plan)

        compile(script, "<generated_spa>", "exec")
        self.assertIn("soup.select('.job-card')", script)

    def test_spa_parse_location_handles_india_with_experience_text(self) -> None:
        plan = ScrapePlan(
            strategy="spa",
            entry_url="https://example.com/careers",
            output_path="generated/example.com/jobs.jsonl",
            pagination={"type": "page"},
            listing={"items_selector": [".job-card"], "title_selector": ["a"], "url_selector": ["a"]},
            detail={"description_selector": ".description"},
            filters={"country": "India", "country_code": "IN"},
        )

        script = CodeGenerationAgent()._render_script(plan)
        namespace: dict[str, object] = {}
        exec(script, namespace)

        parse_location = namespace["parse_location"]
        is_india_job = namespace["is_india_job"]

        location = parse_location("Chennai, Tamil Nadu, India 4-5 years")
        self.assertEqual(location["country"], "India")
        self.assertEqual(location["country_code"], "IN")
        self.assertTrue(is_india_job({"location": location}))

    def test_spa_codegen_single_page_guard(self) -> None:
        plan = ScrapePlan(
            strategy="spa",
            entry_url="https://example.com/careers",
            output_path="generated/example.com/jobs.jsonl",
            pagination={"type": "none"},
            listing={"items_selector": [".job-card"], "title_selector": ["a"], "url_selector": ["a"]},
            detail={"description_selector": ".description"},
            filters={"country": "India", "country_code": "IN"},
        )

        script = CodeGenerationAgent()._render_script(plan)

        self.assertIn("single_page = page_type not in", script)
        self.assertIn("max_pages = 1", script)

    def test_codegen_review_times_out_without_blocking(self) -> None:
        class SlowClient:
            def chat(self, messages, temperature=0.2):
                time.sleep(0.05)
                return "print('still running')"

        plan = ScrapePlan(
            strategy="html",
            entry_url="https://example.com/careers",
            output_path="generated/example.com/jobs.jsonl",
            pagination={"type": "page"},
            listing={"items_selector": ".job-card", "title_selector": "a", "url_selector": "a"},
            detail={"description_selector": ".description"},
            filters={"country": "India", "country_code": "IN"},
        )

        agent = CodeGenerationAgent(llm_client=SlowClient(), review_timeout_seconds=0.01)
        start = time.perf_counter()
        reviewed = agent._review_with_llm(plan, "def main():\n    pass\n")
        elapsed = time.perf_counter() - start

        self.assertIsNone(reviewed)
        self.assertLess(elapsed, 0.2)

    def test_codegen_rejects_invalid_reviewed_script(self) -> None:
        class BadClient:
            def chat(self, messages, temperature=0.2):
                return "    from __future__ import annotations\n    def main():\n        pass\n"

        plan = ScrapePlan(
            strategy="json",
            entry_url="https://example.com/api",
            output_path="generated/example.com/jobs.jsonl",
            pagination={"type": "page"},
            listing={"json_items_path": ["jobs"]},
            detail={"url_path": ["url"]},
            filters={"country": "India", "country_code": "IN"},
        )

        agent = CodeGenerationAgent(llm_client=BadClient())
        reviewed = agent._review_with_llm(plan, agent._render_script(plan))

        self.assertIsNone(reviewed)

    def test_api_plan_without_json_items_falls_back_to_spa(self) -> None:
        plan = ScrapePlan(
            strategy="api",
            entry_url="https://example.com/careers",
            output_path="generated/example.com/jobs.jsonl",
            pagination={"type": "page"},
            listing={
                "items_selector": ".job-card",
                "title_selector": "a",
                "url_selector": "a",
                "json_items_path": [],
            },
            detail={"description_selector": ".description"},
            filters={"country": "India", "country_code": "IN"},
        )

        script = CodeGenerationAgent()._render_script(plan)

        self.assertIn("fetch_rendered_html", script)
        self.assertIn("soup.select('.job-card')", script)
        self.assertNotIn("fetch_json(", script)
        compile(script, "<generated_api_fallback>", "exec")


class TraceTests(unittest.TestCase):
    def test_trace_recorder_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TraceRecorder(domain="example.com", trace_dir=Path(temp_dir))
            recorder.record("agent", "action", {"x": 1}, {"y": 2})
            path = recorder.save()

            self.assertTrue(path.exists())
            self.assertEqual(path.suffix, ".json")
            content = path.read_text(encoding="utf-8")
            self.assertIn("example.com", content)

    def test_trace_agent_summary_handles_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            llm = Mock()
            llm.chat.return_value = "summary"
            agent = TraceAgent(domain="example.com", trace_dir=Path(temp_dir), llm_client=llm)
            agent.recorder.events.append(
                TraceEvent(
                    ts="2026-07-10T00:00:00Z",
                    actor="agent",
                    action="action",
                    input=Path(temp_dir) / "input.json",
                    output={"path": Path(temp_dir) / "output.json"},
                    note=None,
                )
            )

            summary = agent._build_summary()

            self.assertEqual(summary, "summary")
            self.assertTrue(llm.chat.called)
            prompt = llm.chat.call_args.args[0][1]["content"]
            self.assertIn("input.json", prompt)
            self.assertIn("output.json", prompt)


class InvestigationTests(unittest.TestCase):
    def test_investigation_prefers_careers_candidate(self) -> None:
        agent = InvestigationAgent()
        candidates = [
            CandidatePage(url="https://example.com/", source="probe", title="Home"),
            CandidatePage(url="https://example.com/careers", source="search", title="Careers"),
        ]

        target = agent._select_target_candidate(candidates)

        self.assertIsNotNone(target)
        self.assertEqual(target.url, "https://example.com/careers")

    def test_best_container_prefers_outer_job_card(self) -> None:
        from bs4 import BeautifulSoup

        html = """
        <div class="cw-filter-joblist">
          <div class="cw-filter-joblist-left">
            <h3><a class="cw-3-title cw-bw" href="/jobs/1">Senior Project Manager</a></h3>
            <p class="filter-subhead cw-bw">Chennai, Tamil Nadu, India</p>
          </div>
          <div class="cw-filter-joblist-right">
            <span class="search-date-opened">07/31/2024</span>
          </div>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        anchor = soup.select_one("a.cw-3-title")

        best = InvestigationAgent()._best_container_for_anchor(anchor)

        self.assertIsNotNone(best)
        self.assertIn("cw-filter-joblist", " ".join(best.get("class") or []))
        self.assertNotIn("left", " ".join(best.get("class") or []))

    def test_discovery_prefers_firecrawl_before_search(self) -> None:
        from agents.discovery_agent import DiscoveryAgent

        agent = DiscoveryAgent()
        order: list[str] = []

        def probe(domain: str):
            order.append("probe")
            return []

        def search(domain: str):
            order.append("search")
            return []

        agent._probe_candidates = probe  # type: ignore[assignment]
        agent._search_candidates = search  # type: ignore[assignment]
        context = type("Context", (), {"domain": "example.com"})()

        agent.run(context)

        self.assertEqual(order[:2], ["probe", "search"])

    def test_investigation_payload_is_structured(self) -> None:
        finding = InvestigationFinding(
            source_type="spa",
            careers_url="https://example.com/careers",
            platform="example",
            api_url="https://example.com/api/jobs",
            jobs_found_during_investigation=True,
            job_count_detected=2,
            job_titles=["Senior Project Manager", "Product Management Intern"],
            listing_selector=".cw-filter-joblist",
            job_selector=".cw-filter-joblist",
            detail_selector=".cw-job-detail",
            recommended_strategy="beautifulsoup",
            json_paths=[["jobs", 0, "title"]],
            selectors={"listing": {"items_selector": "a"}},
            pagination={"type": "page"},
            listing={"items_selector": "a"},
            detail={"description_path": ["description"]},
        )

        payload = finding.planning_payload()

        self.assertNotIn("evidence", payload)
        self.assertEqual(payload["api_url"], "https://example.com/api/jobs")
        self.assertEqual(payload["json_paths"], [["jobs", 0, "title"]])
        self.assertEqual(payload["selectors"]["listing"]["items_selector"], "a")
        self.assertTrue(payload["jobs_found_during_investigation"])
        self.assertEqual(payload["job_count_detected"], 2)
        self.assertEqual(payload["job_titles"][0], "Senior Project Manager")
        self.assertEqual(payload["recommended_strategy"], "beautifulsoup")


class PlanningTests(unittest.TestCase):
    def test_planning_prefers_html_when_jobs_are_visible(self) -> None:
        investigation = InvestigationFinding(
            source_type="spa",
            careers_url="https://example.com/careers",
            platform="zoho_recruit",
            jobs_found_during_investigation=True,
            job_count_detected=2,
            job_titles=["Senior Project Manager", "Product Management Intern"],
            listing_selector=".cw-filter-joblist",
            job_selector=".cw-filter-joblist",
            detail_selector=".cw-job-detail",
            recommended_strategy="beautifulsoup",
            selectors={"listing": {"items_selector": ".cw-filter-joblist"}},
            pagination={"type": "none"},
            listing={"items_selector": ".cw-filter-joblist", "title_selector": "a.cw-3-title", "url_selector": "a.cw-3-title"},
            detail={"description_selector": ".cw-filter-joblist-left > p.cw-bw"},
        )
        context = type(
            "Context",
            (),
            {
                "domain": "example.com",
                "generated_dir": Path("generated"),
                "normalized_domain": "example.com",
                "investigation": investigation,
            },
        )()

        plan = PlanningAgent()._to_plan(
            context,
            {
                "strategy": "spa",
                "entry_url": investigation.careers_url,
                "output_path": "jobs.json",
                "pagination": {"type": "none"},
                "listing": {"items_selector": ".cw-filter-joblist"},
                "detail": {"description_selector": ".cw-job-detail"},
                "filters": {"country": "India", "country_code": "IN"},
                "notes": "",
            },
        )

        self.assertEqual(plan.strategy, "spa")
        self.assertTrue(plan.browser_required)
        self.assertEqual(plan.listing["items_selector"], ".cw-filter-joblist")

    def test_planning_keeps_investigation_listing_selector(self) -> None:
        investigation = InvestigationFinding(
            source_type="spa",
            careers_url="https://f22labs.zohorecruit.in/jobs/Careers",
            platform="zoho_recruit",
            jobs_found_during_investigation=True,
            job_count_detected=2,
            job_titles=["Senior Project Manager", "Product Management Intern"],
            listing_selector=".cw-filter-joblist",
            job_selector=".cw-filter-joblist",
            detail_selector=".cw-filter-joblist-left > p.cw-bw",
            recommended_strategy="beautifulsoup",
            selectors={"listing": {"items_selector": ".cw-filter-joblist", "title_selector": "a.cw-3-title", "url_selector": "a.cw-3-title"}},
            pagination={"type": "none"},
            listing={"items_selector": ".cw-filter-joblist", "title_selector": "a.cw-3-title", "url_selector": "a.cw-3-title", "location_selector": ".filter-subhead", "description_selector": ".cw-filter-joblist-left > p.cw-bw"},
            detail={"description_path": ["description"]},
        )
        context = type(
            "Context",
            (),
            {
                "domain": "f22labs.com",
                "generated_dir": Path("generated"),
                "normalized_domain": "f22labs.com",
                "investigation": investigation,
            },
        )()

        plan = PlanningAgent()._to_plan(
            context,
            {
                "strategy": "spa",
                "entry_url": investigation.careers_url,
                "output_path": "jobs.json",
                "pagination": {"type": "none"},
                "listing": {"items_selector": ".cw-filter-joblist > .cw-3", "title_selector": "a.cw-3-title", "url_selector": "a.cw-3-title"},
                "detail": {"description_selector": ".cw-filter-joblist-left > p.cw-bw"},
                "filters": {"country": "India", "country_code": "IN"},
                "notes": "",
            },
        )

        self.assertEqual(plan.listing["items_selector"], ".cw-filter-joblist")
        self.assertEqual(plan.listing["title_selector"], "a.cw-3-title")
        self.assertEqual(plan.detail["description_path"], ["description"])


class VerificationTests(unittest.TestCase):
    def test_verification_flags_investigation_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "jobs.jsonl"
            output.write_text("", encoding="utf-8")
            investigation = InvestigationFinding(
                source_type="spa",
                careers_url="https://example.com/careers",
                platform="zoho_recruit",
                jobs_found_during_investigation=True,
                job_count_detected=2,
                job_titles=["Senior Project Manager", "Product Management Intern"],
                recommended_strategy="beautifulsoup",
            )
            context = type(
                "Context",
                (),
                {
                    "output_path": output,
                    "investigation": investigation,
                    "verification": {},
                },
            )()

            report = VerificationAgent().run(context, type("Result", (), {"exit_code": 0, "stdout": "", "stderr": "", "duration_seconds": 0.1})())

            self.assertFalse(report.ok)
            self.assertTrue(any(issue.code == "investigation_mismatch" for issue in report.issues))


class RepairTests(unittest.TestCase):
    def test_repair_preserves_investigation_selectors_on_job_mismatch(self) -> None:
        from agents.repair_agent import RepairAgent

        class DummyLLM:
            def chat(self, messages, temperature=0.1):
                return json.dumps(
                    {
                        "strategy": "html",
                        "entry_url": "https://example.com/careers",
                        "output_path": "jobs.json",
                        "browser_required": True,
                        "pagination": {"type": "none"},
                        "listing": {"items_selector": ".wrong-selector"},
                        "detail": {"description_selector": ".wrong-detail"},
                        "filters": {"country": "India", "country_code": "IN"},
                        "notes": "guess",
                    }
                )

        investigation = InvestigationFinding(
            source_type="spa",
            careers_url="https://f22labs.zohorecruit.in/jobs/Careers",
            platform="zoho_recruit",
            jobs_found_during_investigation=True,
            job_count_detected=2,
            job_titles=["Senior Project Manager", "Product Management Intern"],
            listing_selector=".cw-filter-joblist",
            job_selector=".cw-filter-joblist",
            detail_selector=".cw-filter-joblist-left > p.cw-bw",
            recommended_strategy="beautifulsoup",
            selectors={"listing": {"items_selector": ".cw-filter-joblist", "title_selector": "a.cw-3-title", "url_selector": "a.cw-3-title"}},
            pagination={"type": "none"},
            listing={"items_selector": ".cw-filter-joblist", "title_selector": "a.cw-3-title", "url_selector": "a.cw-3-title", "location_selector": ".filter-subhead", "description_selector": ".cw-filter-joblist-left > p.cw-bw"},
            detail={"description_selector": ".cw-filter-joblist-left > p.cw-bw"},
        )
        plan = ScrapePlan(
            strategy="html",
            entry_url=investigation.careers_url,
            output_path="jobs.json",
            browser_required=False,
            pagination={"type": "none"},
            listing={"items_selector": ".cw-filter-joblist", "title_selector": "a.cw-3-title", "url_selector": "a.cw-3-title"},
            detail={"description_selector": ".cw-filter-joblist-left > p.cw-bw"},
            filters={"country": "India", "country_code": "IN"},
        )
        context = type(
            "Context",
            (),
            {
                "domain": "f22labs.com",
                "investigation": investigation,
                "plan": plan,
                "generated_dir": Path("generated"),
            },
        )()

        repaired = RepairAgent(llm_client=DummyLLM()).run(
            context,
            issues=[{"code": "investigation_mismatch", "message": "mismatch"}],
            execution_result=type("Result", (), {"stderr": "", "stdout": ""})(),
            generated_script="print('x')",
            verification_report={"validation": {"ok": False}},
        )

        self.assertIsNotNone(repaired)
        self.assertEqual(repaired.entry_url, investigation.careers_url)
        self.assertEqual(repaired.listing["items_selector"], ".cw-filter-joblist")
        self.assertEqual(repaired.detail["description_selector"], ".cw-filter-joblist-left > p.cw-bw")
        self.assertTrue(repaired.browser_required)


class ConfigTests(unittest.TestCase):
    def test_agent_profile_defaults(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            planning = _llm_profile(
                "PLANNING",
                default_provider="openai",
                default_model="gpt-5-mini",
                default_base_url="https://api.openai.com/v1",
                default_reasoning_effort="minimal",
            )
            discovery = _llm_profile(
                "DISCOVERY",
                default_provider="gemini",
                default_model="gemini-2.5-flash",
                default_base_url="https://generativelanguage.googleapis.com/v1beta",
            )

            self.assertEqual(planning.model, "gpt-5-mini")
            self.assertEqual(planning.reasoning_effort, "minimal")
            self.assertEqual(discovery.model, "gemini-2.5-flash")
            self.assertIsNone(discovery.reasoning_effort)


class LLMClientTests(unittest.TestCase):
    def test_openai_payload_includes_reasoning_effort(self) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                    }
                }
            ]
        }

        with patch("llm_client.requests.post", return_value=response) as mock_post:
            client = LLMClient(
                provider="openai",
                base_url="https://api.openai.com/v1",
                api_key="test-key",
                model="gpt-5-mini",
                reasoning_effort="minimal",
            )
            output = client.chat([{"role": "user", "content": "hello"}], temperature=0.1)

        self.assertEqual(output, "ok")
        self.assertTrue(mock_post.called)
        self.assertEqual(mock_post.call_args.kwargs["json"]["reasoning_effort"], "minimal")
        self.assertNotIn("temperature", mock_post.call_args.kwargs["json"])


if __name__ == "__main__":
    unittest.main()
