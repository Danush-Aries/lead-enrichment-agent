# Lead Enrichment Agent

Give it a company domain. It crawls the site and uses an LLM to pull out a
structured profile: overview, target audience, contact info, and leadership —
as JSON.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium   # skip if Chrome is already installed
cp .env.example .env          # add ANTHROPIC_API_KEY or OPENROUTER_API_KEY
```

## Run

```bash
python main.py --domains postman.com supabase.com vapi.ai -v
```

Output goes to `output.json` — one entry per domain (fields in `lead_agent/schema.py`).

## How it works

- **Crawl** — follows real nav links (about/team/contact/pricing) instead of
  guessing paths. Skips soft-404 pages. Falls back to real Chrome for
  JS-heavy or bot-blocked sites.
- **Extract** — LLM call with a forced tool schema (Pydantic), so output
  always matches the schema. Anthropic or OpenRouter (free-tier fallback
  chain across 5 models).
- **Resilient** — one domain failing never crashes the batch; it just gets
  `"error"` set and the run continues.

## Notes

- Never scrapes LinkedIn directly — only uses LinkedIn links already present
  on the company's own site.
- Never invents data — leaves a field empty rather than guessing (verified:
  vapi.ai has no public email, output correctly shows `[]`, not a fake one).
- Every profile includes `data_confidence` (0.0–1.0) and
  `tokens_used` / `estimated_cost_usd`.

## Structure

```
main.py               entrypoint
lead_agent/
  fetcher.py           crawling
  extractor.py         LLM extraction
  agent.py             orchestration
  schema.py            output schema
output.json            sample output (3 domains)
```
