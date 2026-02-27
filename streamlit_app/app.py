from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cme_core import extract, ingest, rank  # noqa: E402
from cme_core.models import AnalysisOptions  # noqa: E402
from streamlit_app.ui_components import (  # noqa: E402
    filter_topics,
    render_export_buttons,
    render_topic_details,
    topic_rows,
)

APP_KEY = "analysis_result"
URL_PREVIEW_KEY = "url_preview_document"
URL_INPUT_KEY = "url_input"


def main() -> None:
    st.set_page_config(page_title="NeuroCME High-Yield Coach", layout="wide")
    st.title("NeuroCME High-Yield Coach")
    st.caption("Educational use only. This app summarizes user-provided text and does not provide medical advice.")

    with st.sidebar:
        st.subheader("Controls")
        specialty_focus = st.selectbox("Specialty focus", ["Neuro ICU", "General ICU", "ECMO"])
        desired_depth = st.selectbox("Desired depth", ["boards", "fellowship", "attending"])
        output_type = st.selectbox("Output type", ["outline", "pearls", "flashcards"])
        options = AnalysisOptions(
            specialty_focus=specialty_focus,
            desired_depth=desired_depth,
            output_type=output_type,
        )
        st.markdown("Only plain HTML fetch is supported for URLs. Uploads are processed in memory and not stored by default.")

    pdf_tab, url_tab = st.tabs(["Upload PDF", "Paste URL"])
    with pdf_tab:
        render_pdf_tab(options)
    with url_tab:
        render_url_tab(options)

    result = st.session_state.get(APP_KEY)
    if result:
        render_results(result["document"], result["topics"], output_type)


def render_pdf_tab(options: AnalysisOptions) -> None:
    st.subheader("PDF Ingest")
    uploaded_file = st.file_uploader("Upload a PDF review, guideline, or chapter", type=["pdf"])
    if uploaded_file is not None:
        st.write(f"Selected file: `{uploaded_file.name}`")
        if st.button("Analyze PDF", type="primary", width="content"):
            try:
                document = ingest.ingest_pdf_bytes(uploaded_file.getvalue(), source_name=uploaded_file.name)
                topics = analyze_document(document, options)
                st.session_state[APP_KEY] = {"document": document, "topics": topics}
                st.success("PDF analyzed.")
            except Exception as exc:
                st.error(f"PDF analysis failed: {exc}")


def render_url_tab(options: AnalysisOptions) -> None:
    st.subheader("URL Ingest")
    default_url = st.session_state.get(URL_INPUT_KEY, "")
    url = st.text_input("Open-access article URL", value=default_url, key=URL_INPUT_KEY)
    fetch_col, analyze_col = st.columns(2)
    if fetch_col.button("Fetch preview", width="stretch"):
        if not url.strip():
            st.warning("Enter a URL first.")
        else:
            try:
                document = ingest.ingest_url(url.strip())
                st.session_state[URL_PREVIEW_KEY] = document
                st.success("Fetched URL preview.")
            except Exception as exc:
                st.error(f"URL fetch failed: {exc}")
    if analyze_col.button("Analyze URL", type="primary", width="stretch"):
        if not url.strip():
            st.warning("Enter a URL first.")
        else:
            try:
                document = st.session_state.get(URL_PREVIEW_KEY)
                if document is None or document.source_ref != url.strip():
                    document = ingest.ingest_url(url.strip())
                    st.session_state[URL_PREVIEW_KEY] = document
                topics = analyze_document(document, options)
                st.session_state[APP_KEY] = {"document": document, "topics": topics}
                st.success("URL analyzed.")
            except Exception as exc:
                st.error(f"URL analysis failed: {exc}")
    preview = st.session_state.get(URL_PREVIEW_KEY)
    if preview:
        st.markdown(f"**Preview title**: {preview.title}")
        st.markdown(f"**Paragraphs captured**: {len(preview.paragraphs)}")
        for paragraph in preview.paragraphs[:3]:
            st.markdown(f"- `{paragraph.anchor.label}`: {paragraph.text[:220]}...")


def analyze_document(document, options: AnalysisOptions):
    chunks = extract.extract_chunks(document)
    return rank.rank_document(document=document, chunks=chunks, options=options)


def render_results(document, topics, output_type: str) -> None:
    st.divider()
    st.subheader("Results")
    st.markdown(f"**Document**: {document.title}")
    priorities = st.multiselect("Filter priority", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
    levels = st.multiselect(
        "Filter level",
        ["BASIC", "INTERMEDIATE", "ADVANCED", "EXPERT"],
        default=["BASIC", "INTERMEDIATE", "ADVANCED", "EXPERT"],
    )
    filtered_topics = filter_topics(topics, priorities=priorities, levels=levels)
    st.dataframe(topic_rows(filtered_topics), hide_index=True, width="stretch")
    render_topic_details(filtered_topics, output_type=output_type)
    render_export_buttons(document, filtered_topics)


if __name__ == "__main__":
    main()
