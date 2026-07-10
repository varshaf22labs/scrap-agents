from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMProfile:
    provider: str
    base_url: str | None
    api_key: str | None
    model: str
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    agents_dir: Path
    tools_dir: Path
    prompts_dir: Path
    memory_dir: Path
    traces_dir: Path
    generated_dir: Path
    default_model: str
    llm_base_url: str | None
    llm_api_key: str | None
    search_provider: str
    serper_api_key: str | None
    serpapi_api_key: str | None
    firecrawl_api_key: str | None
    request_timeout_seconds: int
    env_file_loaded: bool
    discovery_llm: LLMProfile
    investigation_llm: LLMProfile
    planning_llm: LLMProfile
    codegen_llm: LLMProfile
    repair_llm: LLMProfile
    trace_llm: LLMProfile


def _load_env_file(path: Path) -> bool:
    if not path.exists():
        return False

    loaded = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded = True
    return loaded


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _llm_profile(
    prefix: str,
    *,
    default_provider: str,
    default_model: str,
    default_base_url: str | None,
    default_reasoning_effort: str | None = None,
) -> LLMProfile:
    provider = _first_env(f"{prefix}_PROVIDER", default=default_provider) or default_provider
    if provider.lower() == "gemini":
        base_url = _first_env(f"{prefix}_BASE_URL", "GEMINI_BASE_URL", default="https://generativelanguage.googleapis.com/v1beta")
        api_key = _first_env(f"{prefix}_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
        reasoning_effort = None
    else:
        base_url = _first_env(
            f"{prefix}_BASE_URL",
            "LLM_BASE_URL",
            "OPENAI_BASE_URL",
            "FREELLMAPI_BASE_URL",
            "GROQ_BASE_URL",
            "XAI_BASE_URL",
            default=default_base_url,
        )
        api_key = _first_env(
            f"{prefix}_API_KEY",
            "LLM_API_KEY",
            "OPENAI_API_KEY",
            "FREELLMAPI_API_KEY",
            "GROQ_API_KEY",
            "XAI_API_KEY",
        )
        reasoning_effort = _first_env(
            f"{prefix}_REASONING_EFFORT",
            "LLM_REASONING_EFFORT",
            default=default_reasoning_effort,
        )
    model = _first_env(f"{prefix}_MODEL", default=default_model) or default_model
    return LLMProfile(provider=provider, base_url=base_url, api_key=api_key, model=model, reasoning_effort=reasoning_effort)


def load_config() -> AppConfig:
    root = Path.cwd()
    env_file_loaded = _load_env_file(root / ".env") or _load_env_file(root / ".env.local")
    return AppConfig(
        root_dir=root,
        agents_dir=root / "agents",
        tools_dir=root / "tools",
        prompts_dir=root / "prompts",
        memory_dir=root / "memory",
        traces_dir=root / "traces",
        generated_dir=root / "generated",
        default_model=_first_env("LLM_MODEL", "OPENAI_MODEL", "GROK_MODEL", "XAI_MODEL", default="default"),
        llm_base_url=_first_env(
            "LLM_BASE_URL",
            "OPENAI_BASE_URL",
            "FREELLMAPI_BASE_URL",
            "GROK_BASE_URL",
            "XAI_BASE_URL",
        ),
        llm_api_key=_first_env(
            "LLM_API_KEY",
            "OPENAI_API_KEY",
            "FREELLMAPI_API_KEY",
            "GROK_API_KEY",
            "XAI_API_KEY",
        ),
        search_provider=_first_env("SEARCH_PROVIDER", default="serper") or "serper",
        serper_api_key=_first_env("SERPER_API_KEY", "SERPER_DEV_API_KEY"),
        serpapi_api_key=_first_env("SERPAPI_API_KEY", "SERP_API_KEY"),
        firecrawl_api_key=_first_env("FIRECRAWL_API_KEY"),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
        env_file_loaded=env_file_loaded,
        discovery_llm=_llm_profile("DISCOVERY", default_provider="gemini", default_model="gemini-2.5-flash", default_base_url="https://generativelanguage.googleapis.com/v1beta"),
        investigation_llm=_llm_profile("INVESTIGATION", default_provider="gemini", default_model="gemini-2.5-flash", default_base_url="https://generativelanguage.googleapis.com/v1beta"),
        planning_llm=_llm_profile(
            "PLANNING",
            default_provider="openai",
            default_model="gpt-5-mini",
            default_base_url="https://api.openai.com/v1",
            default_reasoning_effort="minimal",
        ),
        codegen_llm=_llm_profile(
            "CODEGEN",
            default_provider="openai",
            default_model="gpt-5-mini",
            default_base_url="https://api.openai.com/v1",
            default_reasoning_effort="minimal",
        ),
        repair_llm=_llm_profile(
            "REPAIR",
            default_provider="openai",
            default_model="gpt-5-mini",
            default_base_url="https://api.openai.com/v1",
            default_reasoning_effort="minimal",
        ),
        trace_llm=_llm_profile("TRACE", default_provider="gemini", default_model="gemini-2.5-flash", default_base_url="https://generativelanguage.googleapis.com/v1beta"),
    )


def ensure_directories(config: AppConfig) -> None:
    for path in [
        config.agents_dir,
        config.tools_dir,
        config.prompts_dir,
        config.memory_dir,
        config.traces_dir,
        config.generated_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
