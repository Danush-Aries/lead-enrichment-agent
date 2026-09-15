"""Top-level orchestration: crawl each domain and extract a structured profile,
never letting one domain's failure take down the run."""

from __future__ import annotations

import logging

from .extractor import LLMBackend
from .fetcher import crawl_domain
from .schema import CompanyProfile

logger = logging.getLogger("lead_agent")


def run(
    domains: list[str],
    backend: LLMBackend,
    max_pages: int = 6,
    force_browser: bool = False,
    headed: bool = False,
) -> list[CompanyProfile]:
    profiles: list[CompanyProfile] = []
    for domain in domains:
        logger.info("Crawling %s", domain)
        try:
            pages = crawl_domain(domain, max_pages=max_pages, force_browser=force_browser, headed=headed)
        except Exception as exc:
            logger.warning("Crawl failed for %s: %s", domain, exc)
            profiles.append(CompanyProfile(domain=domain, error=f"Crawl failed: {exc}"))
            continue

        if not pages:
            logger.warning("No pages fetched for %s", domain)
            profiles.append(CompanyProfile(domain=domain, error="No pages could be fetched"))
            continue

        logger.info("Fetched %d page(s) for %s: %s", len(pages), domain, ", ".join(page.url for page in pages))
        try:
            profile = backend.extract(domain, pages)
        except Exception as exc:
            logger.warning("Extraction failed for %s: %s", domain, exc)
            profile = CompanyProfile(domain=domain, source_pages=[page.url for page in pages], error=f"Extraction failed: {exc}")

        profiles.append(profile)
        logger.info("Done: %s (%s)", domain, f"error: {profile.error}" if profile.error else "ok")

    return profiles
