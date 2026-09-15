"""Page fetcher: plain HTTP first, a real Chrome browser (via Playwright) for
pages that are blocked or rendered client-side, or for everything when
force_browser is set.

The browser profile defaults to a project-local `.chrome-profile/` so runs are
reproducible and never collide with a Chrome window that's already open.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

logger = logging.getLogger("lead_agent")

# Scored against a link's last path segment and (short) anchor text.
LINK_KEYWORDS = {
    "about": 3,
    "company": 3,
    "team": 3,
    "leadership": 3,
    "founders": 3,
    "people": 3,
    "story": 3,
    "contact": 2,
    "customers": 1,
    "pricing": 1,
}
SKIP_SECTIONS = {"blog", "docs", "changelog", "guides", "templates", "legal"}
# Tried after homepage links, for sites whose nav doesn't link these pages.
FALLBACK_PATHS = ["about", "about-us", "company", "team", "leadership", "contact", "contact-us"]
SOCIAL_DOMAINS = (
    "linkedin.com", "x.com", "twitter.com", "github.com", "youtube.com",
    "facebook.com", "instagram.com", "discord.com", "discord.gg",
)
# Single-segment social paths that are actions or listings, not profiles.
NON_PROFILE_PATHS = {
    "share", "sharer.php", "intent", "home", "watch", "playlist", "results",
    "shorts", "embed", "login", "signup", "hashtag", "search", "explore",
}

THIN_CONTENT_THRESHOLD = 400
# Pages still this small after a browser retry are app shells or soft 404s.
MIN_CONTENT_CHARS = 100
HTTP_TIMEOUT = 15.0
BROWSER_TIMEOUT_MS = 30_000
NETWORK_IDLE_TIMEOUT_MS = 5_000
ROBOTS_AGENT = "LeadEnrichmentAgent"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 LeadEnrichmentAgent/1.0"
)
DEFAULT_PROFILE_DIR = Path(__file__).resolve().parent.parent / ".chrome-profile"


@dataclass
class FetchedPage:
    url: str
    text: str
    method: str  # "http" or "browser"
    links: list[tuple[str, str]] = field(default_factory=list)  # (absolute href, anchor text)

    def contact_links(self) -> list[tuple[str, str]]:
        """mailto:/tel: links and social *profile* links (not posts or repos), as (href, anchor)."""
        return [
            (href, " ".join(anchor.split())[:80])
            for href, anchor in self.links
            if href.startswith(("mailto:", "tel:")) or _is_social_profile(href)
        ]


def _host(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _page_key(url: str) -> str:
    return _host(url) + urlparse(url).path.rstrip("/")


def _is_social_profile(href: str) -> bool:
    host = _host(href)
    if not any(host == domain or host.endswith("." + domain) for domain in SOCIAL_DOMAINS):
        return False
    segments = [s for s in urlparse(href).path.split("/") if s]
    if not segments or segments[0].lower() in NON_PROFILE_PATHS:
        return False
    if host.endswith("linkedin.com"):
        return len(segments) == 2 and segments[0] in ("company", "in", "school", "showcase")
    if host.endswith("youtube.com"):
        return len(segments) == 1 or (len(segments) == 2 and segments[0] in ("c", "channel", "user"))
    return len(segments) == 1


def _parse_page(url: str, html: str, method: str) -> FetchedPage | None:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or href.startswith(("javascript:", "#")):
            continue
        if not href.startswith(("mailto:", "tel:")):
            href = urljoin(url, href)
        links.append((href, anchor.get_text(" ", strip=True)))

    # Links are collected first so nav/footer contact links survive this —
    # header/footer/nav are site-wide chrome, never page-specific content, so
    # dropping them here cuts a lot of the "Home | Product | Pricing | ..."
    # menu boilerplate that would otherwise burn LLM tokens for no signal.
    for tag in soup(["script", "style", "noscript", "svg", "nav", "header", "footer"]):
        tag.decompose()
    lines = (line.strip() for line in soup.get_text("\n").splitlines())
    text = "\n".join(line for line in lines if line)
    if not text:
        return None
    return FetchedPage(url=url, text=text, method=method, links=links)


def _in_skipped_section(url: str) -> bool:
    segments = [s for s in urlparse(url).path.lower().split("/") if s]
    return bool(segments) and segments[0] in SKIP_SECTIONS


def _is_soft_404(page: FetchedPage, not_found: FetchedPage | None) -> bool:
    """True when a page is near-identical to what the site serves for a made-up URL."""
    if not_found is None:
        return False
    a, b = page.text, not_found.text
    if abs(len(a) - len(b)) > 0.2 * max(len(a), len(b)):
        return False
    return SequenceMatcher(None, a, b).ratio() > 0.9


def _rank_links(links: list[tuple[str, str]], site: str) -> list[str]:
    best: dict[str, tuple[int, str]] = {}
    for href, anchor in links:
        if not href.startswith("http") or _host(href) != site or _in_skipped_section(href):
            continue
        segments = [s for s in urlparse(href).path.lower().split("/") if s]
        if not segments or len(segments) > 3:
            continue
        label = anchor.lower() if len(anchor.split()) <= 3 else ""
        tokens = set(re.split(r"[^a-z]+", f"{segments[-1]} {label}"))
        score = max((weight for keyword, weight in LINK_KEYWORDS.items() if keyword in tokens), default=0)
        key = _page_key(href)
        if score > best.get(key, (0, ""))[0]:
            best[key] = (score, href.split("#")[0])
    ranked = sorted(best.values(), key=lambda item: (-item[0], len(item[1])))
    return [href for _, href in ranked]


def _load_robots(base_url: str, client: httpx.Client) -> RobotFileParser | None:
    try:
        resp = client.get(f"{base_url}/robots.txt")
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    parser = RobotFileParser()
    parser.parse(resp.text.splitlines())
    return parser


class BrowserSession:
    """Starts Chrome on first use and reuses it for the rest of the crawl."""

    def __init__(self, headed: bool = False):
        self.headed = headed
        self._playwright = None
        self._context = None
        self._unavailable = False

    def __enter__(self) -> BrowserSession:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def fetch(self, url: str) -> FetchedPage | None:
        context = self._get_context()
        if context is None:
            return None
        tab = None
        try:
            tab = context.new_page()
            response = tab.goto(url, timeout=BROWSER_TIMEOUT_MS, wait_until="domcontentloaded")
            if response is not None and response.status >= 400:
                logger.info("Browser got HTTP %s for %s", response.status, url)
                return None
            try:
                tab.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT_MS)
            except PlaywrightError:
                pass  # analytics beacons and websockets keep some sites from ever going idle
            return _parse_page(tab.url, tab.content(), "browser")
        except PlaywrightError as exc:
            logger.warning("Browser fetch failed for %s: %s", url, exc)
            return None
        finally:
            if tab is not None:
                tab.close()

    def _get_context(self):
        if self._context is not None or self._unavailable:
            return self._context
        profile_dir = os.environ.get("CHROME_PROFILE_DIR") or str(DEFAULT_PROFILE_DIR)
        options = {"headless": not self.headed, "user_agent": USER_AGENT}
        try:
            self._playwright = sync_playwright().start()
            try:
                self._context = self._playwright.chromium.launch_persistent_context(
                    profile_dir, channel="chrome", **options
                )
            except PlaywrightError:
                logger.info("Chrome not found, using Playwright's bundled Chromium")
                self._context = self._playwright.chromium.launch_persistent_context(profile_dir, **options)
            logger.info("Browser started (profile: %s)", profile_dir)
        except Exception as exc:
            logger.warning("Browser unavailable, continuing with HTTP results only: %s", exc)
            self._unavailable = True
            self.close()
        return self._context

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None


def _fetch(
    url: str,
    client: httpx.Client,
    browser: BrowserSession,
    robots: RobotFileParser | None,
    force_browser: bool,
) -> FetchedPage | None:
    if robots is not None and not robots.can_fetch(ROBOTS_AGENT, url):
        logger.info("Skipping %s (disallowed by robots.txt)", url)
        return None
    page = browser.fetch(url) if force_browser else _fetch_http_first(url, client, browser)
    if page is None or len(page.text) < MIN_CONTENT_CHARS:
        return None
    return page


def _fetch_http_first(url: str, client: httpx.Client, browser: BrowserSession) -> FetchedPage | None:
    try:
        resp = client.get(url)
    except httpx.HTTPError as exc:
        logger.info("HTTP fetch failed for %s (%s), trying the browser", url, exc)
        return browser.fetch(url)
    if resp.status_code in (403, 429, 503):
        logger.info("HTTP %s for %s, likely bot protection, trying the browser", resp.status_code, url)
        return browser.fetch(str(resp.url))
    if resp.status_code >= 400 or "html" not in resp.headers.get("content-type", ""):
        return None

    final_url = str(resp.url)
    page = _parse_page(final_url, resp.text, "http")
    if page is None or len(page.text) < THIN_CONTENT_THRESHOLD:
        # Near-empty static HTML usually means the page is rendered client-side.
        page = browser.fetch(final_url) or page
    return page


def crawl_domain(
    domain: str,
    max_pages: int = 6,
    force_browser: bool = False,
    headed: bool = False,
) -> list[FetchedPage]:
    """Fetch the homepage plus the most relevant company pages linked from it."""
    base_url = (domain if domain.startswith("http") else f"https://{domain}").rstrip("/")
    pages: list[FetchedPage] = []
    seen: set[str] = set()

    with (
        httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
            transport=httpx.HTTPTransport(retries=2),
        ) as client,
        BrowserSession(headed) as browser,
    ):
        robots = _load_robots(base_url, client)

        home = _fetch(base_url, client, browser, robots, force_browser)
        if home is None:
            return []
        pages.append(home)
        seen.update({_page_key(base_url), _page_key(home.url)})

        # Many sites answer unknown paths with a 200 "not found" page; fingerprint it.
        not_found = _fetch(f"{base_url}/{uuid.uuid4().hex[:16]}", client, browser, robots, force_browser)

        candidates = _rank_links(home.links, _host(home.url)) + [f"{base_url}/{path}" for path in FALLBACK_PATHS]
        for url in dict.fromkeys(candidates):
            if len(pages) >= max_pages:
                break
            requested = _page_key(url)
            if requested in seen:
                continue
            seen.add(requested)

            page = _fetch(url, client, browser, robots, force_browser)
            if page is None:
                continue
            final = _page_key(page.url)
            if final != requested:
                if final in seen or _in_skipped_section(page.url):
                    continue  # redirected to a page we already have, or somewhere irrelevant
                seen.add(final)
            if _is_soft_404(page, not_found):
                logger.info("Skipping %s (soft 404)", page.url)
                continue
            pages.append(page)

    return pages
