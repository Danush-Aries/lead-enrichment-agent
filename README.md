# Lead Enrichment Agent

Crawls a company's own website and produces a structured profile: overview,
target audience, contact info, and leadership — using an LLM for extraction
instead of brittle regex/CSS-selector scraping.

## How it works

1. **Fetch** (`lead_agent/fetcher.py`) — fetches the homepage, then follows
   its actual nav links to find the most relevant company pages (about, team,
   leadership, contact, ...), ranked by keyword match rather than guessing
   fixed paths. Plain HTTP is tried first; a real headless Chrome instance
   (via Playwright) kicks in only when a page is blocked (403/429/503),
   fails outright, or comes back too thin to be real content (a common sign
   of a JS-rendered single-page app). It also fingerprints each site's "not
   found" page (by requesting a random nonexistent path) so pages that
   return HTTP 200 with a soft-404 body — common on SPA-style sites — get
   filtered out instead of being fed to the LLM as if they were real content.
2. **Extract** (`lead_agent/extractor.py`) — sends the crawled page text to
   an LLM with a forced tool/function call whose input schema is generated
   directly from the `CompanyProfile` Pydantic model, so the output is
   guaranteed to match the schema or the run reports a clear error. Supports
   two backends: native Anthropic, or OpenRouter (OpenAI-compatible API,
   useful for routing to free-tier models) — see Setup below. The OpenRouter
   backend tries a short list of free models in order and moves on when one
   is rate-limited, hangs, or declines to call the tool — free-tier models
   are noticeably less reliable than a paid Claude/GPT call, so a single
   model isn't enough to depend on. A hard wall-clock deadline per attempt
   (`MODEL_DEADLINE_SECONDS` in `extractor.py`) guards against a gateway that
   trickles keep-alive bytes and would otherwise hang well past the client's
   own timeout.
3. **Orchestrate** (`lead_agent/agent.py`) — runs the above per domain,
   catching and recording per-domain failures instead of crashing the whole
   batch.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium   # only needed if you don't have Chrome installed
cp .env.example .env          # then fill in ANTHROPIC_API_KEY *or* OPENROUTER_API_KEY
```

**LLM provider**: set `ANTHROPIC_API_KEY` (console.anthropic.com) for native
Claude, or `OPENROUTER_API_KEY` (openrouter.ai) to route through OpenRouter's
free-tier models (a comma-separated fallback list, see
`DEFAULT_OPENROUTER_MODELS` in `extractor.py`; override with `OPENROUTER_MODEL`
to use specific model(s) instead). If both keys are set, Anthropic wins unless
`LLM_PROVIDER` says otherwise. Anthropic is the more reliable option — the
OpenRouter path exists for running this without a paid API key, at the cost
of occasionally falling through several free models before one responds.

## Usage

```bash
python main.py --domains postman.com supabase.com vapi.ai --output output.json
```

Add `-v` for progress logging. Output is a JSON array of `CompanyProfile`
objects (see `lead_agent/schema.py`), one per domain, written to the path
given by `--output`.

## Design decisions worth calling out

- **LinkedIn is never scraped directly.** `leadership[].linkedin_url` is only
  populated when a LinkedIn link is already present on the company's own
  site. Scraping LinkedIn itself violates its ToS and risks account bans —
  out of scope for this agent by design, not an oversight.
- **Browser fallback uses real Chrome, not bundled headless Chromium**, via
  Playwright's `channel="chrome"`. Falls back to bundled Chromium
  automatically if Chrome isn't installed. The profile directory defaults to
  a project-local `.chrome-profile/` folder (gitignored) so the agent is
  reproducible on any machine and never fights with an already-open personal
  Chrome window. See `.env.example` for pointing it at a real profile
  instead.
- **Failures are data, not crashes.** A domain that can't be fetched or
  fails extraction gets a profile with `error` set rather than aborting the
  batch — check `output.json` for any `"error"` fields after a run.
- **No LinkedIn scraping, no data invented.** The system prompt explicitly
  tells the model to leave a field null rather than guess — verified in
  practice on vapi.ai, which has no public email and gets `"emails": []`
  rather than a fabricated address.

## Project structure

```
main.py                  # CLI entrypoint
lead_agent/
  fetcher.py              # HTTP + browser-fallback crawling
  extractor.py             # LLM structured extraction
  agent.py                 # per-domain orchestration
  schema.py                 # Pydantic output schema
output.json                # sample output (3 domains)
```
