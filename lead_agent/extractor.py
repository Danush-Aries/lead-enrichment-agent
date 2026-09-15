"""LLM-backed structured extraction: crawled pages -> CompanyProfile.

Two backends, selected via LLM_PROVIDER (or auto-detected from whichever API
key is set): native Anthropic, or OpenRouter's OpenAI-compatible API, which
can route to free-tier models.
"""

from __future__ import annotations

import json
import logging
import os
import signal
from contextlib import contextmanager
from typing import Protocol

import openai

from .fetcher import FetchedPage
from .schema import CompanyProfile

logger = logging.getLogger("lead_agent")

MODEL_DEADLINE_SECONDS = 50


@contextmanager
def _wall_clock_deadline(seconds: int):
    """Hard cutoff via SIGALRM, since a congested free-tier gateway can trickle
    keep-alive bytes that reset httpx's read timeout and hang far past it."""
    if not hasattr(signal, "SIGALRM"):  # Windows: no per-call deadline, rely on the client timeout
        yield
        return

    def _on_alarm(signum, frame):
        raise TimeoutError(f"exceeded {seconds}s wall-clock deadline")

    try:
        previous = signal.signal(signal.SIGALRM, _on_alarm)
    except ValueError:  # not the main thread — signal-based deadlines aren't available here
        yield
        return
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

MAX_CHARS_PER_PAGE = 6000

SYSTEM_PROMPT = (
    "You extract structured company information from raw website text for a "
    "lead-enrichment pipeline. Only report facts present in the provided text. "
    "Leave a field empty/null rather than guessing or inferring information that "
    "isn't stated. Never fabricate emails, phone numbers, or URLs. Only attach a "
    "LinkedIn URL to a person when it appears in the provided links and the link "
    "text or URL slug clearly identifies that person. Set data_confidence to your "
    "own honest estimate (0.0-1.0) of how complete and reliable this specific "
    "extraction is — e.g. low if the pages barely mention leadership or contact "
    "info, high if the about/contact pages were thorough and unambiguous."
)

TOOL_NAME = "record_company_profile"
TOOL_DESCRIPTION = "Record the extracted company profile in a structured form."

# Filled in by the pipeline, not the model.
PIPELINE_FIELDS = ("domain", "source_pages", "fetch_method", "llm_model", "tokens_used", "estimated_cost_usd", "error")

# All confirmed free + tool-calling + structured_outputs-capable at time of
# writing. Free-tier availability shifts day to day, so this is deliberately
# a longer chain — an early success short-circuits the rest.
DEFAULT_OPENROUTER_MODELS = (
    "nvidia/nemotron-3-super-120b-a12b:free,"
    "nex-agi/nex-n2.5-pro:free,"
    "openrouter/free,"
    "nex-agi/nex-n2.5-mini:free,"
    "dots-studio/dots-3-note-preview:free"
)


def _inline_refs(node, defs: dict):
    """Recursively replace {"$ref": "#/$defs/X"} with X's own schema. Some
    OpenRouter providers' grammar-constrained decoders (observed: Nex AGI)
    reject $ref/$defs outright with a compile error, so a fully self-contained
    schema is more broadly compatible than the $ref-based one Pydantic emits
    by default for nested models like ContactInfo/LeadershipMember."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            resolved = _inline_refs(defs[ref.rsplit("/", 1)[-1]], defs)
            extra = {k: _inline_refs(v, defs) for k, v in node.items() if k != "$ref"}
            return {**resolved, **extra}
        return {k: _inline_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(item, defs) for item in node]
    return node


def _build_tool_schema() -> dict:
    schema = CompanyProfile.model_json_schema()
    for name in PIPELINE_FIELDS:
        schema.get("properties", {}).pop(name, None)
    if "required" in schema:
        schema["required"] = [name for name in schema["required"] if name not in PIPELINE_FIELDS]
    return _inline_refs(schema, schema.get("$defs", {}))


def _build_prompt(domain: str, pages: list[FetchedPage]) -> str:
    sections = [f"--- PAGE: {page.url} ---\n{page.text[:MAX_CHARS_PER_PAGE]}" for page in pages]

    contact: dict[str, str] = {}
    for page in pages:
        for href, anchor in page.contact_links():
            contact.setdefault(href.rstrip("/"), anchor)
    if contact:
        lines = [f"{anchor} -> {href}" if anchor else href for href, anchor in contact.items()]
        sections.append("--- CONTACT AND SOCIAL LINKS FOUND IN PAGE MARKUP ---\n" + "\n".join(lines))

    return f"Domain: {domain}\n\nExtract a company profile from the following crawled pages.\n\n" + "\n\n".join(sections)


def _build_profile(
    domain: str,
    pages: list[FetchedPage],
    tool_input: dict,
    model: str,
    tokens_used: int | None = None,
    estimated_cost_usd: float | None = None,
) -> CompanyProfile:
    try:
        data = {key: value for key, value in dict(tool_input).items() if key not in PIPELINE_FIELDS}
        return CompanyProfile.model_validate(
            {
                **data,
                "domain": domain,
                "source_pages": [page.url for page in pages],
                "fetch_method": "+".join(sorted({page.method for page in pages})),
                "llm_model": model,
                "tokens_used": tokens_used,
                "estimated_cost_usd": estimated_cost_usd,
            }
        )
    except Exception as exc:  # malformed model output shouldn't crash the run
        return CompanyProfile(domain=domain, source_pages=[page.url for page in pages], error=f"Invalid model output: {exc}")


class LLMBackend(Protocol):
    def extract(self, domain: str, pages: list[FetchedPage]) -> CompanyProfile: ...


class AnthropicBackend:
    def __init__(self, client, model: str | None = None):
        self.client = client
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

    def extract(self, domain: str, pages: list[FetchedPage]) -> CompanyProfile:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_prompt(domain, pages)}],
            tools=[{"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "input_schema": _build_tool_schema()}],
            tool_choice={"type": "tool", "name": TOOL_NAME},
        )
        tool_use = next((block for block in response.content if block.type == "tool_use"), None)
        if tool_use is None:
            return CompanyProfile(domain=domain, source_pages=[page.url for page in pages], error="Model did not return structured output")
        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        # Not hardcoding a $/token rate here — Anthropic's pricing can change and
        # a stale guess baked into the code is worse than no number at all.
        return _build_profile(domain, pages, tool_use.input, response.model, tokens_used=tokens_used)


class OpenRouterBackend:
    """Tries each configured model in order, moving on when one is rate-limited,
    errors out, or doesn't return a usable tool call — free models do all three."""

    def __init__(self, client, models: list[str] | None = None):
        self.client = client
        configured = os.environ.get("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODELS
        self.models = models or [name.strip() for name in configured.split(",") if name.strip()]

    def extract(self, domain: str, pages: list[FetchedPage]) -> CompanyProfile:
        prompt = _build_prompt(domain, pages)
        tools = [
            {
                "type": "function",
                "function": {"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "parameters": _build_tool_schema()},
            }
        ]
        failures = []
        for model in self.models:
            logger.info("Extracting %s with %s", domain, model)
            try:
                with _wall_clock_deadline(MODEL_DEADLINE_SECONDS):
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                        tools=tools,
                        tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
                    )
            except (openai.APIError, TimeoutError) as exc:
                logger.warning("%s failed for %s: %s", model, domain, exc)
                failures.append(f"{model}: {type(exc).__name__}")
                continue

            choices = response.choices or []
            tool_calls = choices[0].message.tool_calls if choices else None
            if not tool_calls:
                said_instead = (choices[0].message.content or "").strip()[:200] if choices else ""
                logger.warning("%s returned no tool call for %s%s", model, domain, f" (said: {said_instead!r})" if said_instead else "")
                failures.append(f"{model}: no tool call returned")
                continue
            try:
                arguments = json.loads(tool_calls[0].function.arguments)
            except json.JSONDecodeError:
                logger.warning("%s returned malformed tool call arguments for %s", model, domain)
                failures.append(f"{model}: malformed tool call arguments")
                continue

            tokens_used = response.usage.total_tokens if response.usage else None
            # ":free" is a contractual guarantee of $0 cost, not an estimate. For a
            # paid model we'd need its per-token rate — not hardcoded here since
            # OpenRouter's paid catalog changes; left as None rather than a guess.
            cost = 0.0 if model.endswith(":free") else None
            profile = _build_profile(domain, pages, arguments, response.model or model, tokens_used=tokens_used, estimated_cost_usd=cost)
            if profile.error is None:
                logger.info("%s succeeded for %s", model, domain)
                return profile
            logger.warning("%s produced an invalid profile for %s: %s", model, domain, profile.error)
            failures.append(f"{model}: {profile.error}")

        return CompanyProfile(
            domain=domain,
            source_pages=[page.url for page in pages],
            error="All models failed: " + "; ".join(failures),
        )


def build_backend() -> LLMBackend:
    """Picks a backend from LLM_PROVIDER, or from whichever API key is present
    (Anthropic wins if both are set)."""
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if not provider:
        provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openrouter"

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (required for LLM_PROVIDER=anthropic)")
        from anthropic import Anthropic

        return AnthropicBackend(Anthropic(api_key=api_key))

    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (required for LLM_PROVIDER=openrouter)")
        # Free models can hang or queue for minutes under load. A short per-call
        # timeout with no SDK-level retries lets us fail fast onto the next model
        # in the list instead of retrying the same overloaded one for ages.
        client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=45, max_retries=0)
        return OpenRouterBackend(client)

    raise RuntimeError(f"Unknown LLM_PROVIDER: {provider!r} (expected 'anthropic' or 'openrouter')")
