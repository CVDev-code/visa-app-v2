import json
import os
from typing import Dict, Optional

from openai import OpenAI


def _get_secret(name: str):
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


SYSTEM = """You extract structured metadata from USCIS O-1 evidence PDFs.
Return ONLY valid JSON.
If a field is not found, return an empty string for that field.
"""

USER = """Extract metadata from the following document text.

Return JSON with keys:
- source_url
- venue_name
- performance_date
- org_name
- salary_amount

Guidelines:
- source_url: a URL visible in the document (prefer the publication URL).
- performance_date: the date of the performance/event (as written).
- venue_name: venue / organisation where performance occurs.
- org_name: organisation relevant to recognition (criterion 6).
- salary_amount: any explicit compensation figure (e.g. $10,000).

DOCUMENT TEXT:
{text}
"""


def autodetect_metadata(document_text: str) -> Dict:
    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    model = _get_secret("OPENAI_MODEL") or "gpt-4.1-mini"
    client = OpenAI(api_key=api_key)

    prompt = USER.format(text=document_text[:20000])  # keep costs controlled

    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = getattr(resp, "output_text", "") or ""
        data = json.loads(raw) if raw else {}
    except Exception:
        data = {}

    # normalize
    out = {
        "source_url": str(data.get("source_url", "") or ""),
        "venue_name": str(data.get("venue_name", "") or ""),
        "performance_date": str(data.get("performance_date", "") or ""),
        "org_name": str(data.get("org_name", "") or ""),
        "salary_amount": str(data.get("salary_amount", "") or ""),
    }
    return out
