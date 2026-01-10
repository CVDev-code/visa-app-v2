import csv
import io
import json
import os
import re
from typing import Dict, Optional

from openai import OpenAI


# -----------------------------
# Secrets helper
# -----------------------------
def _get_secret(name: str):
    # Works on Streamlit Cloud (st.secrets) and locally (.env / env vars)
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


# ============================================================
# PART A: Mode B helpers (CSV + merge + basic first-page signals)
# ============================================================

URL_REGEX = re.compile(r"(https?://[^\s)>\]]+)", re.IGNORECASE)


def extract_first_page_signals(first_page_text: str) -> Dict:
    """
    Lightweight non-AI extraction from page-1 text.
    Returns dict with any fields we can guess reliably.
    """
    text = first_page_text or ""
    url = ""
    m = URL_REGEX.search(text)
    if m:
        url = m.group(1).strip().rstrip(".,);]")

    return {
        "source_url": url or None,
        "venue_name": None,
        "performance_date": None,
        "salary_amount": None,
        "org_name": None,
    }


def make_csv_template(filenames: list[str]) -> bytes:
    """
    Produces a CSV template for bulk metadata entry.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["filename", "source_url", "venue_name", "performance_date", "org_name", "salary_amount"])
    for fn in filenames:
        writer.writerow([fn, "", "", "", "", ""])
    return buf.getvalue().encode("utf-8")


def parse_metadata_csv(csv_bytes: bytes) -> Dict[str, Dict]:
    """
    Parses uploaded CSV and returns:
      { filename: {source_url, venue_name, performance_date, org_name, salary_amount} }
    """
    text = csv_bytes.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    required = {"filename", "source_url", "venue_name", "performance_date", "org_name", "salary_amount"}
    if not required.issubset(set(reader.fieldnames or [])):
        raise ValueError(f"CSV must include headers: {sorted(required)}")

    out: Dict[str, Dict] = {}
    for row in reader:
        fn = (row.get("filename") or "").strip()
        if not fn:
            continue
        out[fn] = {
            "source_url": (row.get("source_url") or "").strip() or None,
            "venue_name": (row.get("venue_name") or "").strip() or None,
            "performance_date": (row.get("performance_date") or "").strip() or None,
            "org_name": (row.get("org_name") or "").strip() or None,
            "salary_amount": (row.get("salary_amount") or "").strip() or None,
        }
    return out


def merge_metadata(
    filename: str,
    auto: Optional[Dict] = None,
    global_defaults: Optional[Dict] = None,
    csv_data: Optional[Dict[str, Dict]] = None,
    overrides: Optional[Dict] = None,
) -> Dict:
    """
    Merge priority (highest wins):
      overrides > csv row > global defaults > auto
    """
    auto = auto or {}
    global_defaults = global_defaults or {}
    overrides = overrides or {}
    row = (csv_data or {}).get(filename, {}) if csv_data else {}

    def pick(key: str):
        return (
            overrides.get(key)
            or row.get(key)
            or global_defaults.get(key)
            or auto.get(key)
            or None
        )

    return {
        "source_url": pick("source_url"),
        "venue_name": pick("venue_name"),
        "performance_date": pick("performance_date"),
        "org_name": pick("org_name"),
        "salary_amount": pick("salary_amount"),
    }


# ============================================================
# PART B: AI metadata auto-detect (Fixed for OpenAI v1.0+)
# ============================================================

_AUTODETECT_SYSTEM = (
    "You extract structured metadata from USCIS O-1 evidence PDFs. "
    "Return ONLY valid JSON. If a field is not found, return an empty string."
)

_AUTODETECT_USER = """Extract metadata from the following document text.

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
    """
    Optional AI step: extracts metadata candidates from document text.
    """
    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    # Corrected model name: gpt-4o-mini
    model = _get_secret("OPENAI_MODEL") or "gpt-4o-mini"
    client = OpenAI(api_key=api_key)

    prompt = _AUTODETECT_USER.format(text=(document_text or "")[:20000])

    try:
        # Standard Chat Completion API call
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _AUTODETECT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        
        # Corrected way to access the response content
        raw = resp.choices[0].message.content
        data = json.loads(raw) if raw else {}
    except Exception as e:
        # Print the error to Streamlit logs for debugging
        print(f"Metadata Extraction Error: {e}")
        data = {}

    def s(key: str) -> str:
        val = data.get(key, "")
        return str(val or "").strip()

    return {
        "source_url": s("source_url"),
        "venue_name": s("venue_name"),
        "performance_date": s("performance_date"),
        "org_name": s("org_name"),
        "salary_amount": s("salary_amount"),
    }
