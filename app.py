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

    st.subheader("Manual metadata (optional)")
    source_url_input = st.text_input(
        "Source URL", value="", help="Global fallback if not detected in PDF"
    )
    venue_name_input = st.text_input(
        "Venue / Organization", value="", help="Global fallback for criteria 2/4"
    )
    org_name_input = st.text_input("Org Name (Crit 6)", value="")
    performance_date_input = st.text_input("Performance Date", value="")
    salary_amount_input = st.text_input("Salary amount", value="")

    st.subheader("Select O-1 criteria")
    default_criteria = ["2_past", "2_future", "3", "4_past", "4_future"]
    selected_criteria_ids: list[str] = []
    for cid, desc in CRITERIA.items():
        checked = st.checkbox(f"({cid}) {desc}", value=(cid in default_criteria))
        if checked:
            selected_criteria_ids.append(cid)

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload at least one PDF to begin.")
    st.stop()

if not beneficiary_name.strip():
    st.warning("Enter the beneficiary full name in the sidebar.")
    st.stop()

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
# Metadata Processing
# -------------------------
st.subheader("🧾 Metadata Management")

global_defaults = {
    "source_url": source_url_input.strip() or None,
    "venue_name": venue_name_input.strip() or None,
    "performance_date": performance_date_input.strip() or None,
    "salary_amount": salary_amount_input.strip() or None,
    "org_name": org_name_input.strip() or None,
}

# CSV Logic
with st.expander("CSV metadata (bulk mode)", expanded=False):
    filenames = [f.name for f in uploaded_files]
    st.download_button("⬇️ Download CSV template", data=make_csv_template(filenames), file_name="o1_metadata.csv", mime="text/csv")
    csv_file = st.file_uploader("Upload filled CSV", type=["csv"], key="csv_up")
    if csv_file:
        st.session_state["csv_metadata"] = parse_metadata_csv(csv_file.getvalue())

# Resolve per-file metadata
for f in uploaded_files:
    first_page_text = extract_first_page_text_from_pdf_bytes(f.getvalue())
    auto = extract_first_page_signals(first_page_text)
    
    overrides = st.session_state["overrides_by_file"].get(f.name, {})
    resolved = merge_metadata(
        filename=f.name,
        auto=auto,
        global_defaults=global_defaults,
        csv_data=st.session_state["csv_metadata"],
        overrides=overrides,
    )
    st.session_state["meta_by_file"][f.name] = resolved

    with st.expander(f"Edit Metadata: {f.name}"):
        o = dict(overrides)
        o["venue_name"] = st.text_input("Venue Override", value=resolved.get("venue_name") or "", key=f"v_{f.name}")
        st.session_state["overrides_by_file"][f.name] = {k: v for k, v in o.items() if v}
        st.json(resolved)

# -------------------------
# Step 1: AI Analysis
# -------------------------
st.divider()
if st.button("1️⃣ Generate Quotes (AI)", type="primary"):
    with st.spinner("Analyzing PDFs..."):
        for f in uploaded_files:
            text = extract_text_from_pdf_bytes(f.getvalue())
            data = suggest_ovisa_quotes(
                document_text=text,
                beneficiary_name=beneficiary_name,
                beneficiary_variants=beneficiary_variants,
                selected_criteria_ids=selected_criteria_ids
            )
            st.session_state["ai_by_file"][f.name] = data
            st.session_state["approval"][f.name] = {
                cid: {it["quote"]: True for it in data.get("by_criterion", {}).get(cid, [])}
                for cid in selected_criteria_ids
            }
    st.success("Analysis complete.")

# -------------------------
# Step 2: Approval UI
# -------------------------
st.divider()
st.subheader("2️⃣ Review & Approve")
for f in uploaded_files:
    if f.name not in st.session_state["ai_by_file"]: continue
    st.markdown(f"### {f.name}")
    by_criterion = st.session_state["ai_by_file"][f.name].get("by_criterion", {})
    
    for cid in selected_criteria_ids:
        items = by_criterion.get(cid, [])
        if not items: continue
        with st.expander(f"Criterion {cid} - {CRITERIA[cid]}"):
            for it in items:
                q = it["quote"]
                st.session_state["approval"][f.name][cid][q] = st.checkbox(
                    f"[{it['strength']}] {q}", 
                    value=st.session_state["approval"][f.name][cid].get(q, True),
                    key=f"cb_{f.name}_{cid}_{q}"
                )

# -------------------------
# Step 3: Export
# -------------------------
st.divider()
st.subheader("3️⃣ Export")

def build_annotated_pdf_bytes(pdf_bytes: bytes, quotes: list[str], criterion_id: str, filename: str):
    # This now matches the 3-argument signature in your src/pdf_highlighter.py
    # pdf_bytes, quote_terms, meta
    meta = st.session_state["meta_by_file"].get(filename, {}).copy()
    meta["beneficiary_name"] = beneficiary_name
    
    # Check if criterion_id is needed in your version of annotate_pdf_bytes
    # If your src/pdf_highlighter.py was updated to accept it, use:
    # return annotate_pdf_bytes(pdf_bytes, quotes, criterion_id=criterion_id, meta=meta)
    # Based on the file content you provided for pdf_highlighter.py, it takes 3:
    return annotate_pdf_bytes(pdf_bytes, quotes, meta=meta)

if st.button("Generate & Download ZIP"):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for f in uploaded_files:
            if f.name not in st.session_state["approval"]: continue
            for cid in selected_criteria_ids:
                approvals = st.session_state["approval"][f.name].get(cid, {})
                approved = [q for q, ok in approvals.items() if ok]
                if approved:
                    out_bytes = build_annotated_pdf_bytes(f.getvalue(), approved, cid, f.name)
                    zf.writestr(f"{f.name}_crit_{cid}.pdf", out_bytes)
    
    st.download_button("⬇️ Download ZIP", data=zip_buffer.getvalue(), file_name="o1_highlights.zip")
