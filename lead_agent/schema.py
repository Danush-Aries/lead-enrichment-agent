"""Structured output schema for company lead-enrichment extraction."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    emails: list[str] = Field(default_factory=list, description="Verified or discovered email addresses")
    phone_numbers: list[str] = Field(default_factory=list)
    social_links: list[str] = Field(default_factory=list, description="LinkedIn/Twitter/etc company profile URLs found on the site")
    contact_page_url: Optional[str] = None


class LeadershipMember(BaseModel):
    name: str
    title: Optional[str] = None
    linkedin_url: Optional[str] = Field(
        default=None,
        description="LinkedIn URL only if explicitly linked from the company's own site — never guessed or scraped from LinkedIn directly",
    )


class CompanyProfile(BaseModel):
    domain: str
    company_name: Optional[str] = None
    overview: Optional[str] = Field(default=None, description="1-3 sentence summary of what the company does")
    target_audience: Optional[str] = Field(default=None, description="Who the company's product/service is aimed at")
    industry: Optional[str] = None
    contact: ContactInfo = Field(default_factory=ContactInfo)
    leadership: list[LeadershipMember] = Field(default_factory=list)
    source_pages: list[str] = Field(default_factory=list, description="URLs actually crawled to produce this profile")
    fetch_method: Optional[str] = Field(default=None, description="'http', 'browser', or 'browser+http' — how the pages were retrieved")
    llm_model: Optional[str] = Field(default=None, description="Model that produced the extraction")
    error: Optional[str] = Field(default=None, description="Set when extraction failed for this domain; other fields may be partial")
