from __future__ import annotations

from pathlib import Path

from cme_core import extract, ingest


ROOT = Path(__file__).resolve().parents[1]


def test_ingest_pdf_extracts_page_anchors() -> None:
    document = ingest.ingest_pdf_path(ROOT / "sample_data" / "sample_page.pdf")
    chunks = extract.extract_chunks(document)

    assert document.source_type == "pdf"
    assert document.metadata["page_count"] >= 1
    assert len(document.paragraphs) >= 3
    assert document.paragraphs[0].anchor.page == 1
    assert chunks[0].anchors[0].page == 1
