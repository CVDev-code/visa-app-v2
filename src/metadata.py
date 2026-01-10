import csv
import io
import re
from typing import Dict, Optional, List


# URL detection (page 1)
URL_REGEX = re.compile(r"(https?://[^\s\)\]\}<>\"']+)", re.IGNORECASE)

# Date detection (page 1) – broad but practical
DATE_REGEX = re.compile(
    r"\b("
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"                     # 12/01/2026 or 12-01-2026
    r"|"
    r"\d{4}[/-]\d{1,2}[/-]\d{1,2}"                       # 2026-01-12
    r"|"
    r"\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}"  # 12 Jan 2026
    r"|"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}" # Jan 12, 2026
    r")\b",
    re.IGNORECASE,
)

# Money detection (page 1)
MONEY_REGEX = re.compile(
    r"\b("
    r"(?:USD|GBP|EUR)\s*\d+(?:,\d{3})*(?:\.\d{2})?"      # USD 12000
    r"|"
    r"[$£€]\s*\d+(?:,\d{3})*(?:\.\d{2})?"                # $12,000.00 / £5,000
    r")\b"
)


CSV_COLUMNS = ["filename", "source_url", "venue_name", "performance_date", "salary_amount"]


def extract_first_page_signals(first_page_text: str) -> Dict[str, Optional[str]]:
    """
    Lightweight heuristics for page-1 metadata detection.
    Returns best-guess strings (or None).
    """
    text = first_page_text or ""

    url = None
    m = URL_REGEX.search(text)
    if m:
        url = m.group(1).rstrip(".,;")

    date = None
    m = DATE_REGEX.search(text)
    if m:
        date = m.group(0).strip()

    money = None
    m = MONEY_REGEX.search(text)
    if m:
        money = m.group(0).strip()

    return {
        "source_url": url,
        "performance_date": date,
        "salary_amount": money,
        # venue_name is usually not reliably auto-detected; we mainly use defaults/CSV/override
        "venue_name": None,
    }


def make_csv_template(filenames: List[str]) -> bytes:
    """
    Build a CSV template that users can optionally fill in Excel.
    """
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for fn in filenames:
        writer.writerow(
            {
                "filename": fn,
                "source_url": "",
                "venue_name": "",
                "performance_date": "",
                "salary_amount": "",
            }
        )
    return output.getvalue().encode("utf-8")


def parse_metadata_csv(file_bytes: bytes) -> Dict[str, Dict[str, str]]:
    """
    CSV must contain a 'filename' column.
    Returns mapping: filename -> metadata dict (only non-empty fields).
    """
    decoded = file_bytes.decode("utf-8-sig")  # handles Excel BOM
    reader = csv.DictReader(io.StringIO(decoded))

    if not reader.fieldnames or "filename" not in reader.fieldnames:
        raise ValueError("CSV must include a 'filename' column")

    out: Dict[str, Dict[str, str]] = {}
    for row in reader:
        filename = (row.get("filename") or "").strip()
        if not filename:
            continue

        cleaned = {}
        for k in CSV_COLUMNS:
            if k == "filename":
                continue
            v = (row.get(k) or "").strip()
            if v:
                cleaned[k] = v

        out[filename] = cleaned

    return out


def merge_metadata(
    filename: str,
    auto: Dict[str, Optional[str]],
    global_defaults: Dict[str, Optional[str]],
    csv_data: Optional[Dict[str, Dict[str, str]]] = None,
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Optional[str]]:
    """
    Priority (highest wins):
      1) Per-PDF overrides
      2) CSV row
      3) Global defaults
      4) Auto-detected
    """
    merged: Dict[str, Optional[str]] = dict(auto or {})

    # Global defaults override auto
    if global_defaults:
        for k, v in global_defaults.items():
            if v:
                merged[k] = v

    # CSV override defaults
    if csv_data and filename in csv_data:
        for k, v in csv_data[filename].items():
            if v:
                merged[k] = v

    # Per-PDF overrides override everything
    if overrides:
        for k, v in overrides.items():
            if v:
                merged[k] = v

    return merged
