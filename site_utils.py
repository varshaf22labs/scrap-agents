from __future__ import annotations

from urllib.parse import urlparse


COMMON_CAREERS_PATHS = [
    "/careers",
    "/career",
    "/jobs",
    "/job-openings",
    "/open-positions",
    "/join-us",
    "/work-with-us",
    "/global/en/careers",
    "/en/careers",
    "/company/careers",
    "/about/careers",
]


def normalize_domain(domain: str) -> str:
    cleaned = domain.strip()
    if "://" not in cleaned:
        cleaned = f"https://{cleaned}"
    parsed = urlparse(cleaned)
    return parsed.netloc.lower()


def origin_from_domain(domain: str) -> str:
    cleaned = domain.strip()
    if "://" not in cleaned:
        cleaned = f"https://{cleaned}"
    parsed = urlparse(cleaned)
    return f"{parsed.scheme}://{parsed.netloc}"


def build_common_urls(domain: str) -> list[str]:
    origin = origin_from_domain(domain)
    return [f"{origin}{path}" for path in COMMON_CAREERS_PATHS]


def location_text_mentions_india(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return "india" in lowered or "in " in lowered or lowered.endswith(", in")

