import io
import zipfile

import streamlit as st
from dotenv import load_dotenv

from src.pdf_text import (
    extract_text_from_pdf_bytes,
    extract_first_page_text_from_pdf_bytes,
)
from src.metadata import (
    extract_first_page_signals,
    make_csv_template,
    parse_metadata_csv,
    merge_metadata,
)
from src.openai_terms import suggest_ovisa_quotes
from src.pdf_highlighter import annotate_pdf_bytes
from src.prompts import CRITERIA

load_dotenv()
st.set_page_config(page_title="O-1 PDF Highlighter", layout="wide")

st.title("O-1 PDF Highlighter")
st.caption(
    "Upload PDFs → choose criteria → approve/reject quotes → export criterion-specific highlighted PDFs"
)

# -------------------------
# Case setup (sidebar)
# -------------------------
with st.sidebar:
    st.header("Case setup")
    beneficiary_name = st.text_input("Beneficiary full name", value="")
    variants_raw = st.text_input("Name variants (comma-separated)", value="")
    beneficiary_variants = [v.strip() for v in variants_raw.split(",") if v.strip()]

    st.subheader("Manual metadata (optional but improves outputs)")
    source_url = st.text_input(
        "Source URL (exact text as shown in PDF page 1)", value=""
    )
    venue_name = st.text_input(
        "Venue / organisation name (for criteria 2/4)", value=""
    )
    org_name = st.text_input("Organisation name (for criterion 6)", value="")
    performance_date = st.text_input(
        "Performance date (exact text as shown in PDF)", value=""
    )
    salary_amount = st.text_input(
        "Salary amount (for criterion 7) – e.g. $10,000", value=""
    )

    st.subheader("Select O-1 criteria to extract")
    default_criteria = ["2_past", "2_future", "3", "4_past", "4_future"]
    selected_criteria_ids: list[str] = []
    for cid, desc in CRITERIA.items():
        checked = st.checkbox(f"({cid}) {desc}", value=(cid in default_criteria))
        if checked:
            selected_criteria_ids.append(cid)

    st.divider()
    st.caption("Tip: Tick only the criteria you want to build evidence for in this batch.")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload at least one PDF to begin.")
    st.stop()

if not beneficiary_name.strip():
    st.warning("Enter the beneficiary full name in the sidebar to improve extraction accuracy.")
    st.stop()

if not selected_criteria_ids:
    st.warning("Tick at least one O-1 criterion in the sidebar.")
    st.stop()

st.divider()

# -------------------------
# Session state
# -------------------------
if "ai_by_file" not in st.session_state:
    st.session_state["ai_by_file"] = {}
if "approval" not in st.session_state:
    st.session_state["approval"] = {}
if "csv_metadata" not in st.session_state:
    st.session_state["csv_metadata"] = None
if "overrides_by_file" not in st.session_state:
    st.session_state["overrides_by_file"] = {}
if "meta_by_file" not in st.session_state:
    st.session_state["meta_by_file"] = {}

# -------------------------
# Metadata (Mode B)
# -------------------------
st.subheader("🧾 Metadata (optional but improves highlighting)")

with st.expander("Batch defaults (apply to all PDFs)", expanded=True):
    default_source_url = st.text_input("Default source URL (optional)", value="")
    default_venue_name = st.text_input("Default venue / organization name (optional)", value="")
    default_performance_date = st.text_input("Default performance date (optional)", value="")
    default_salary_amount = st.text_input("Default salary amount (optional)", value="")

global_defaults = {
    "source_url": default_source_url.strip() or None,
    "venue_name": default_venue_name.strip() or None,
    "performance_date": default_performance_date.strip() or None,
    "salary_amount": default_salary_amount.strip() or None,
}

with st.expander("CSV metadata (optional bulk mode)", expanded=False):
    filenames = [f.name for f in uploaded_files]
    template_bytes = make_csv_template(filenames)
    st.download_button("⬇️ Download CSV template", data=template_bytes, file_name="o1_metadata_template.csv", mime="text/csv")
    csv_file = st.file_uploader("Upload filled CSV (optional)", type=["csv"], key="metadata_csv_uploader")
    if csv_file is not None:
        try:
            st.session_state["csv_metadata"] = parse_metadata_csv(csv_file.getvalue())
            applied = len([fn for fn in filenames if fn in st.session_state["csv_metadata"]])
            st.success(f"CSV loaded. Rows matched to {applied}/{len(filenames)} uploaded PDFs.")
        except Exception as e:
            st.session_state["csv_metadata"] = None
            st.error(f"Could not parse CSV: {e}")

csv_data = st.session_state.get("csv_metadata")

for f in uploaded_files:
    first_page_text = extract_first_page_text_from_pdf_bytes(f.getvalue())
    auto = extract_first_page_signals(first_page_text)
    overrides = st.session_state["overrides_by_file"].get(f.name, {})
    resolved = merge_metadata(f.name, auto, global_defaults, csv_data, overrides)
    st.session_state["meta_by_file"][f.name] = resolved

    with st.expander(f"Metadata overrides for: {f.name}", expanded=False):
        o = dict(overrides)
        o["source_url"] = st.text_input("Source URL override", value=o.get("source_url", "") or (resolved.get("source_url") or ""), key=f"url_{f.name}").strip()
        o["venue_name"] = st.text_input("Venue override", value=o.get("venue_name", "") or (resolved.get("venue_name") or ""), key=f"venue_{f.name}").strip()
        o["performance_date"] = st.text_input("Date override", value=o.get("performance_date", "") or (resolved.get("performance_date") or ""), key=f"date_{f.name}").strip()
        o["salary_amount"] = st.text_input("Salary override", value=o.get("salary_amount", "") or (resolved.get("salary_amount") or ""), key=f"money_{f.name}").strip()
        st.session_state["overrides_by_file"][f.name] = {k: v for k, v in o.items() if v}

st.divider()

# -------------------------
# Step 1: Generate AI quotes
# -------------------------
st.subheader("1️⃣ Generate criterion-tagged quote candidates (AI)")
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    run_ai = st.button("Generate for all PDFs", type="primary")
with colB:
    if st.button("Clear results"):
        st.session_state["ai_by_file"] = {}
        st.session_state["approval"] = {}
        st.rerun()

if run_ai:
    with st.spinner("Generating quote candidates…"):
        for f in uploaded_files:
            text = extract_text_from_pdf_bytes(f.getvalue())
            data = suggest_ovisa_quotes(text, beneficiary_name, beneficiary_variants, selected_criteria_ids)
            st.session_state["ai_by_file"][f.name] = data
            if f.name not in st.session_state["approval"]:
                st.session_state["approval"][f.name] = {}
            for cid in selected_criteria_ids:
                items = data.get("by_criterion", {}).get(cid, [])
                st.session_state["approval"][f.name][cid] = {it["quote"]: True for it in items}
    st.success("Done.")

st.divider()

# -------------------------
# Step 2: Approve/Reject
# -------------------------
st.subheader("2️⃣ Approve / Reject quotes by criterion")
for f in uploaded_files:
    st.markdown(f"## 📄 {f.name}")
    data = st.session_state["ai_by_file"].get(f.name)
    if not data: continue
    for cid in selected_criteria_ids:
        items = data.get("by_criterion", {}).get(cid, [])
        with st.expander(f"Criterion ({cid}): {CRITERIA.get(cid, '')}"):
            if not items:
                st.write("No candidates found.")
                continue
            approvals = st.session_state["approval"].get(f.name, {}).get(cid, {})
            for i, it in enumerate(items):
                q = it["quote"]
                approvals[q] = st.checkbox(f"[{it.get('strength', 'medium')}] {q}", value=approvals.get(q, True), key=f"chk_{f.name}_{cid}_{i}")
            st.session_state["approval"][f.name][cid] = approvals

st.divider()

# -------------------------
# Step 3: Export
# -------------------------
st.subheader("3️⃣ Export highlighted PDFs")

def build_annotated_pdf_bytes(pdf_bytes: bytes, quotes: list[str], criterion_id: str, filename: str):
    resolved = st.session_state["meta_by_file"].get(filename, {}) or {}
    meta = {
        "source_url": resolved.get("source_url") or source_url,
        "venue_name": resolved.get("venue_name") or venue_name,
        "org_name": resolved.get("org_name") or org_name,
        "performance_date": resolved.get("performance_date") or performance_date,
        "salary_amount": resolved.get("salary_amount") or salary_amount,
        "beneficiary_name": beneficiary_name,
        "beneficiary_variants": beneficiary_variants,
    }
    # Matches the fixed signature in highlighter.py
    return annotate_pdf_bytes(pdf_bytes, quotes, criterion_id=criterion_id, meta=meta)

if st.button("Export ALL selected criteria as ZIP", type="primary"):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for f in uploaded_files:
            if f.name not in st.session_state["approval"]: continue
            for cid in selected_criteria_ids:
                apps = st.session_state["approval"][f.name].get(cid, {})
                approved = [q for q, ok in apps.items() if ok]
                if approved:
                    out_bytes, report = build_annotated_pdf_bytes(f.getvalue(), approved, cid, f.name)
                    zf.writestr(f.name.replace(".pdf", f"_crit-{cid}.pdf"), out_bytes)
    st.download_button("⬇️ Download ZIP", data=zip_buffer.getvalue(), file_name="o1_highlights.zip")

for f in uploaded_files:
    if f.name not in st.session_state["ai_by_file"]: continue
    st.markdown(f"### 📄 {f.name}")
    for cid in selected_criteria_ids:
        apps = st.session_state["approval"][f.name].get(cid, {})
        approved = [q for q, ok in apps.items() if ok]
        if approved:
            if st.button(f"Generate PDF: Crit {cid}", key=f"gen_{f.name}_{cid}"):
                out_bytes, report = build_annotated_pdf_bytes(f.getvalue(), approved, cid, f.name)
                st.download_button(f"⬇️ Download {cid}", data=out_bytes, file_name=f"{f.name}_crit{cid}.pdf", key=f"dl_{f.name}_{cid}")
