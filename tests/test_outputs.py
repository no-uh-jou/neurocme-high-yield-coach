from __future__ import annotations

from pathlib import Path

from cme_core import extract, ingest, outputs, rank
from cme_core.models import AnalysisOptions


ROOT = Path(__file__).resolve().parents[1]


def test_exports_include_topic_content() -> None:
    html = (ROOT / "sample_data" / "sample_article.html").read_text(encoding="utf-8")
    document = ingest.document_from_html(html=html, url="https://example.test/sample")
    chunks = extract.extract_chunks(document)
    topics = rank.rank_document(document=document, chunks=chunks, options=AnalysisOptions())

    json_payload = outputs.export_topics_json(document, topics)
    csv_payload = outputs.export_topics_csv(topics)
    markdown_payload = outputs.export_topics_markdown(document, topics)
    anki_payload = outputs.export_anki_tsv(topics)

    assert "Status Epilepticus" in json_payload
    assert "topic,priority,level,score,anchors,rationale" in csv_payload
    assert "# Sample Neurocritical Care Article" in markdown_payload
    assert "front\tback\tanchor\tcard_type" in anki_payload
