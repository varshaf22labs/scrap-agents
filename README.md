# Scrap Agents

An AI agent system that takes only a company domain, investigates the site, writes a standalone job scraper, runs it, verifies the output, and repairs the plan if needed.

This project is built around one idea:

**The agent writes the scraper. The generated scraper does the scraping.**

## What It Does

- Starts from a single domain like `f22labs.com`
- Finds likely careers pages
- Investigates the source type:
  - HTML
  - SPA
  - embedded JSON
  - API-style responses
- Generates a standalone Python scraper for that domain
- Runs the scraper locally
- Validates the JSONL output
- Repairs the plan and regenerates if validation fails
- Saves a full trace of the run

## Why This Exists

Most companies publish jobs across different platforms with no universal API.
This project is meant to handle that problem by using an agentic workflow that can adapt at runtime.

The goal is not to build one scraper for one site.
The goal is to build an agent that can write scrapers for new sites from scratch.

## Output Format

The generated scraper writes one job per line as JSONL.

Each job follows this schema:

- `title`
- `job_id`
- `location.city`
- `location.state`
- `location.country`
- `location.country_code`
- `url`
- `apply_url`
- `date_posted`
- `date_posted_text`
- `job_description`
- `employment_type`
- `work_type`
- `salary_range`

Missing fields are emitted as `null`.

## Core Rules

- Agentic decisions are made at runtime
- No per-domain hardcoded selector tables
- Generated scrapers do not call any LLM at runtime
- No regex-based field extraction in generated code
- Missing fields stay `null`
- Every run produces a trace
- The agent runs the generated scraper and verifies the result
- If validation fails, the agent repairs and regenerates

## Project Flow

1. **Discovery**
   - Search for likely careers pages and candidate URLs.

2. **Investigation**
   - Fetch the page.
   - Detect source type and platform clues.
   - Find selectors or JSON paths.
   - Estimate pagination behavior.

3. **Planning**
   - Turn investigation findings into a structured scrape plan.

4. **Code Generation**
   - Render a standalone scraper script from the plan.

5. **Execution**
   - Run the generated scraper locally.

6. **Verification**
   - Check job count, India filtering, schema shape, and output validity.

7. **Repair**
   - If validation fails, update the plan and regenerate.

## Project Structure

- `app.py` - orchestrates the full agent pipeline
- `agents/` - discovery, investigation, planning, code generation, execution, verification, repair, tracing
- `tools/` - search, firecrawl, playwright, validators, and helpers
- `generated/` - generated standalone scrapers
- `traces/` - run traces
- `tests/` - unit tests for the pipeline

## Example Run

```powershell
python app.py f22labs.com
```

JSON summary:

```powershell
python app.py f22labs.com --json
```

The generated scraper is written to:

```text
generated/<domain>/scraper.py
```

The output file is typically:

```text
output/jobs.jsonl
```

## Example Output

```json
{"title":"Senior Project Manager","job_id":"...","location":{"city":"Chennai","state":"Tamil Nadu","country":"India","country_code":"IN"},"url":"...","apply_url":"...","date_posted":null,"date_posted_text":null,"job_description":"...","employment_type":null,"work_type":null,"salary_range":null}
```

## Configuration

Use a root `.env` file for model and provider settings.

Common environment variables:

- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`
- `LLM_REASONING_EFFORT`
- `DISCOVERY_PROVIDER`
- `INVESTIGATION_PROVIDER`
- `PLANNING_PROVIDER`
- `CODEGEN_PROVIDER`
- `REPAIR_PROVIDER`
- `TRACE_PROVIDER`
- `SEARCH_PROVIDER`
- `SERPER_API_KEY`
- `SERPAPI_API_KEY`
- `FIRECRAWL_API_KEY`
- `REQUEST_TIMEOUT_SECONDS`

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m playwright install chromium
```

```

## Trace Files

Every run saves a trace file in `traces/` with:

- discovery output
- investigation evidence
- planning output
- generated code path
- execution result
- verification result
- repair attempts

## Notes

- The generated scraper is standalone and does not use an LLM at runtime.
- The agent is designed to generalize across different careers platforms.
- The system is optimized for India job extraction.

