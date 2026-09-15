#!/usr/bin/env python3
"""CLI entrypoint for the lead-enrichment agent.

Usage:
    python main.py --domains postman.com supabase.com vapi.ai --output output.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from dotenv import load_dotenv

from lead_agent.agent import run
from lead_agent.extractor import build_backend

DEFAULT_DOMAINS = ["postman.com", "supabase.com", "vapi.ai"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous lead-enrichment agent")
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS, help="Domains to crawl")
    parser.add_argument("--output", default="output.json", help="Path to write results JSON")
    parser.add_argument("--max-pages", type=int, default=6, help="Max pages to crawl per domain")
    parser.add_argument("--browser", action="store_true", help="Fetch every page with Chrome instead of plain HTTP")
    parser.add_argument("--headed", action="store_true", help="Show the browser window (with --browser, or on fallback)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.getLogger("lead_agent").setLevel(logging.INFO if args.verbose else logging.WARNING)

    try:
        backend = build_backend()
    except RuntimeError as exc:
        print(f"ERROR: {exc}. Copy .env.example to .env and fill in an API key.", file=sys.stderr)
        return 1

    profiles = run(
        args.domains,
        backend,
        max_pages=args.max_pages,
        force_browser=args.browser,
        headed=args.headed,
    )

    output = [p.model_dump() for p in profiles]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    succeeded = sum(1 for p in profiles if not p.error)
    print(f"Wrote {len(profiles)} profile(s) to {args.output} ({succeeded} succeeded, {len(profiles) - succeeded} failed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
